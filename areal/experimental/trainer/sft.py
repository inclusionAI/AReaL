from __future__ import annotations

import os

import torch.distributed as dist
from datasets import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import (
    SFTConfig,
    TrainDatasetConfig,
    TrainEngineConfig,
    ValidDatasetConfig,
)
from areal.api.io_struct import FinetuneSpec, StepInfo
from areal.engine.sft.lm_engine import FSDPLMEngine, MegatronLMEngine
from areal.platforms import current_platform
from areal.utils import logging, perf_tracer, seeding, stats_tracker
from areal.utils.data import (
    broadcast_tensor_container,
    cycle_dataloader,
    pad_sequences_to_tensors,
    tensor_container_to,
)
from areal.utils.dataloader import create_dataloader
from areal.utils.device import log_gpu_stats
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_processor_and_tokenizer
from areal.utils.perf_tracer import Category
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger

logger = logging.getLogger(__name__)


class SFTTrainer:
    def __init__(
        self,
        config: SFTConfig,
        train_dataset: Dataset,
        valid_dataset: Dataset | None = None,
    ):
        rank = int(os.getenv("RANK", "0"))
        # Configure performance tracer
        if config.perf_tracer is not None:
            perf_tracer.configure(config.perf_tracer, rank=rank)

        self.config = config
        self.processor, self.tokenizer = load_hf_processor_and_tokenizer(
            config.tokenizer_path
        )

        # Set seed.
        seeding.set_random_seed(config.seed, key=f"trainer{rank}")

        # Parse allocation mode.
        self.allocation_mode = AllocationMode.from_str(config.allocation_mode)

        # Create models.
        self.actor = self._create_actor(config.model)

        # Create dataloaders
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.train_dataloader = self._create_dataloader(
            train_dataset,
            dataset_config=self.config.train_dataset,
            rank=self.actor.data_parallel_rank,
            world_size=self.actor.data_parallel_world_size,
        )
        self.valid_dataloader = None
        if self.config.valid_dataset is not None and valid_dataset is not None:
            self.valid_dataloader = self._create_dataloader(
                valid_dataset,
                dataset_config=self.config.valid_dataset,
                rank=self.actor.data_parallel_rank,
                world_size=self.actor.data_parallel_world_size,
            )

        ft_spec = FinetuneSpec(
            total_train_epochs=config.total_train_epochs,
            dataset_size=len(self.train_dataloader) * config.train_dataset.batch_size,
            train_batch_size=config.train_dataset.batch_size,
        )

        # Initialize models
        self.parallel_strategy = self.allocation_mode.train
        assert self.parallel_strategy is not None
        engine_init_kwargs = {"addr": None, "ft_spec": ft_spec}
        self.actor.initialize(**engine_init_kwargs)

        # Set up evaluation
        self.evaluator = Evaluator(config.evaluator, ft_spec)

        # Set up save as HF model
        self.saver = Saver(config.saver, ft_spec)
        self.recover_handler = RecoverHandler(config.recover, ft_spec)

        # Set up statistics logging (wandb, tensoboard, etc.)
        self.stats_logger = StatsLogger(config, ft_spec)

        # Set up checkpointing for recover
        self.recover_info = self.recover_handler.load(
            self.actor,
            self.saver,
            self.evaluator,
            self.stats_logger,
            self.train_dataloader,
        )

    def train(self):
        config = self.config
        start_step = (
            self.recover_info.last_step_info.next().global_step
            if self.recover_info is not None
            else 0
        )

        total_epochs = config.total_train_epochs
        steps_per_epoch = len(self.train_dataloader)
        max_steps = total_epochs * steps_per_epoch

        global_step = 0
        data_generator = cycle_dataloader(self.train_dataloader)
        for global_step in range(start_step, max_steps):
            if (
                config.total_train_steps is not None
                and global_step >= config.total_train_steps
            ):
                break
            epoch = global_step // steps_per_epoch
            step = global_step % steps_per_epoch

            batch = next(data_generator)

            with (
                stats_tracker.record_timing("to_device"),
                perf_tracer.trace_scope(
                    "train.to_device",
                    category=Category.IO,
                    args={"global_step": global_step},
                ),
            ):
                # NOTE: data are identical across model+context parallel group
                batch = tensor_container_to(batch, current_platform.current_device())

            with (
                stats_tracker.record_timing("bcast"),
                perf_tracer.trace_scope(
                    "train.bcast",
                    category=Category.COMM,
                    args={"global_step": global_step},
                ),
            ):
                batch = broadcast_tensor_container(
                    batch,
                    src_rank=self.actor.current_data_parallel_head(),
                    group=self.actor.context_and_model_parallel_group,
                )

            with (
                stats_tracker.record_timing("train_step"),
                perf_tracer.trace_scope(
                    "train.sft_step",
                    category=Category.COMPUTE,
                    args={"global_step": global_step},
                ),
            ):
                self.actor.train_lm(batch)
                self.actor.step_lr_scheduler()
                log_gpu_stats("after train step")

            self.actor.set_version(global_step + 1)

            with (
                stats_tracker.record_timing("save"),
                perf_tracer.trace_scope(
                    "train.save",
                    category=Category.IO,
                    args={"global_step": global_step},
                ),
            ):
                self._save_hf(epoch=epoch, epoch_step=step, global_step=global_step)

            with (
                stats_tracker.record_timing("checkpoint_for_recover"),
                perf_tracer.trace_scope(
                    "train.checkpoint",
                    category=Category.IO,
                    args={"global_step": global_step},
                ),
            ):
                self._save_recover_checkpoint(
                    epoch=epoch, epoch_step=step, global_step=global_step
                )

            with (
                stats_tracker.record_timing("eval"),
                perf_tracer.trace_scope(
                    "train.eval",
                    category=Category.COMPUTE,
                    args={"global_step": global_step},
                ),
            ):
                self._evaluate(
                    epoch=epoch,
                    epoch_step=step,
                    global_step=global_step,
                )

            with perf_tracer.trace_scope(
                "train.log_stats",
                category=Category.INSTR,
                args={"global_step": global_step},
            ):
                self._export_and_commit_stats(
                    epoch=epoch, epoch_step=step, global_step=global_step
                )

            perf_tracer.save(step=global_step)

    def close(self):
        self.stats_logger.close()
        self.actor.destroy()
        perf_tracer.save(force=True)

    def _create_dataloader(
        self,
        dataset: Dataset,
        dataset_config: TrainDatasetConfig | ValidDatasetConfig,
        rank: int,
        world_size: int,
    ) -> StatefulDataLoader:
        return create_dataloader(
            dataset,
            rank=rank,
            world_size=world_size,
            dataset_config=dataset_config,
            collate_fn=pad_sequences_to_tensors,
        )

    def _create_actor(self, actor_config: TrainEngineConfig):
        if self.allocation_mode.train_backend == "fsdp":
            actor = FSDPLMEngine(config=actor_config)
        elif self.allocation_mode.train_backend == "megatron":
            actor = MegatronLMEngine(config=actor_config)
        else:
            raise ValueError(
                f"Invalid backend: {self.allocation_mode.train_backend}, expected fsdp or megatron"
            )
        actor.create_process_group(parallel_strategy=self.allocation_mode.train)
        return actor

    def _save_hf(self, epoch: int, epoch_step: int, global_step: int):
        # Save as HF models for evaluation
        self.saver.save(
            self.actor,
            epoch,
            epoch_step,
            global_step,
            tokenizer=self.tokenizer,
            processor=self.processor,
        )
        dist.barrier(device_ids=[self.actor.device.index])
        current_platform.synchronize()

    def _save_recover_checkpoint(self, epoch: int, epoch_step: int, global_step: int):
        # Save recoverable checkpoints
        to_save = dict(default=self.actor)
        step_info = StepInfo(
            global_step=global_step,
            epoch=epoch,
            epoch_step=epoch_step,
            steps_per_epoch=len(self.train_dataloader),
        )
        self.recover_handler.dump(
            to_save,
            step_info,
            self.saver,
            self.evaluator,
            self.stats_logger,
            self.train_dataloader,
            tokenizer=self.tokenizer,
            processor=self.processor,
        )

        dist.barrier(device_ids=[self.actor.device.index])
        current_platform.synchronize()

    def _evaluate_fn(self):
        for data in self.valid_dataloader:
            data = tensor_container_to(data, current_platform.current_device())
            data = broadcast_tensor_container(
                data,
                src_rank=self.actor.current_data_parallel_head(),
                group=self.actor.context_and_model_parallel_group,
            )
            self.actor.evaluate_lm(data)
        dist.barrier(device_ids=[self.actor.device.index])
        current_platform.synchronize()

    def _evaluate(
        self,
        epoch: int,
        epoch_step: int,
        global_step: int,
    ):
        if self.valid_dataloader is None:
            return
        self.evaluator.evaluate(
            self._evaluate_fn,
            epoch,
            epoch_step,
            global_step,
        )
        dist.barrier(device_ids=[self.actor.device.index])
        current_platform.synchronize()

    def _export_and_commit_stats(self, epoch: int, epoch_step: int, global_step: int):
        # Upload statistics to the logger (e.g., wandb)
        stats = self.actor.export_stats()
        self.stats_logger.commit(epoch, epoch_step, global_step, stats)

        dist.barrier(device_ids=[self.actor.device.index])
        current_platform.synchronize()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        if exc_type is not None:
            raise exc_value
