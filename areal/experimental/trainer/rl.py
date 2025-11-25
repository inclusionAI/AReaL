import os
from copy import deepcopy

import torch.distributed as dist
from datasets import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import (
    GRPOConfig,
    InferenceEngineConfig,
    PPOActorConfig,
)
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import FinetuneSpec, StepInfo, WeightUpdateMeta
from areal.api.workflow_api import RolloutWorkflow
from areal.engine.ppo.actor import FSDPPPOActor, MegatronPPOActor
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.engine.vllm_remote import RemotevLLMEngine
from areal.platforms import current_platform
from areal.utils import logging, seeding, stats_tracker
from areal.utils.dataloader import create_dataloader
from areal.utils.device import log_gpu_stats
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger

logger = logging.getLogger("GRPOTrainer")


class GRPOTrainer:
    def __init__(
        self,
        config: GRPOConfig,
        train_dataset: Dataset,
        valid_dataset: Dataset | None = None,
        tokenizer: PreTrainedTokenizerFast | None = None,
    ):
        self.config = config
        rank = int(os.getenv("RANK", "0"))

        self.tokenizer = tokenizer
        if self.tokenizer is None:
            self.tokenizer = load_hf_tokenizer(config.tokenizer_path)

        seeding.set_random_seed(config.seed, key=f"trainer{rank}")
        self.allocation_mode = AllocationMode.from_str(config.allocation_mode)

        self.parallel_strategy = self.allocation_mode.train
        assert self.parallel_strategy is not None

        self.inference_backend = self.allocation_mode.gen_backend
        if self.inference_backend not in ["sglang", "vllm"]:
            raise ValueError(
                f"Invalid inference generation backend: {self.inference_backend}, expected sglang or vllm"
            )

        self.train_backend = self.allocation_mode.train_backend
        if self.train_backend not in ["fsdp", "megatron"]:
            raise ValueError(
                f"Invalid backend: {self.train_backend}, expected fsdp or megatron"
            )
        # Create actor
        self.actor = self._create_train_engine(config.actor)

        # Create dataloaders
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.train_dataloader = self._create_dataloader(train_dataset, split="train")
        self.valid_dataloader = self._create_dataloader(valid_dataset, split="test")

        self.ft_spec = FinetuneSpec(
            total_train_epochs=config.total_train_epochs,
            dataset_size=len(self.train_dataloader) * config.train_dataset.batch_size,
            train_batch_size=config.train_dataset.batch_size,
        )

        # Initialize inference engine
        self.rollout = self._init_inference_engine(config.rollout, is_eval=False)
        self.eval_rollout = self._init_inference_engine(config.rollout, is_eval=True)

        # Initialize train engine
        engine_init_kwargs = {"addr": None, "ft_spec": self.ft_spec}
        if self.train_backend == "megatron":
            engine_init_kwargs["parallel_strategy"] = self.parallel_strategy
            engine_init_kwargs["seed"] = config.seed
        self.actor.initialize(**engine_init_kwargs)

        # Prepare weight update meta and connect to inference engine
        if self.train_backend == "megatron":
            self.weight_update_meta = WeightUpdateMeta.from_megatron_xccl(
                self.allocation_mode,
                nccl_group_name=self.actor.weight_update_group_name,
            )
        else:
            self.weight_update_meta = WeightUpdateMeta.from_fsdp_xccl(
                self.allocation_mode
            )
        self.actor.connect_engine(self.rollout, self.weight_update_meta)

        self.ref = None
        if config.actor.kl_ctl > 0 and config.ref is not None:
            self.ref = self._create_train_engine(config.ref)
            self.ref.initialize(**engine_init_kwargs)

        # Prepare save, stats logger, evaluator, and recover handler
        self.saver = Saver(config.saver, self.ft_spec)
        self.stats_logger = StatsLogger(config, self.ft_spec)
        self.evaluator = Evaluator(config.evaluator, self.ft_spec)

        self.recover_handler = RecoverHandler(config.recover, self.ft_spec)
        self.recover_info = self.recover_handler.load(
            self.actor,
            self.saver,
            self.evaluator,
            self.stats_logger,
            self.train_dataloader,
            inference_engine=self.rollout,
            weight_update_meta=self.weight_update_meta,
        )

    def train(self, workflow: RolloutWorkflow, eval_workflow: RolloutWorkflow):
        """
        Train the model using GRPO algorithm, with custom rollout workflow.

        Args:
            workflow: Rollout workflow for training.
            eval_workflow: Rollout workflow for evaluation.
        """
        config = self.config
        start_step = (
            self.recover_info.last_step_info.next().global_step
            if self.recover_info is not None
            else 0
        )

        total_epochs = config.total_train_epochs
        steps_per_epoch = len(self.train_dataloader)
        max_steps = total_epochs * steps_per_epoch

        for global_step in range(start_step, max_steps):
            epoch = global_step // steps_per_epoch
            step = global_step % steps_per_epoch
            step_info = StepInfo(
                global_step=global_step,
                epoch=epoch,
                epoch_step=step,
                steps_per_epoch=steps_per_epoch,
            )

            with stats_tracker.record_timing("rollout"):
                batch = self.actor.prepare_batch(
                    self.train_dataloader,
                    granularity=self.actor.config.group_size,
                    workflow=workflow,
                    should_accept_fn=lambda sample: True,
                )

            if config.actor.recompute_logprob or config.actor.use_decoupled_loss:
                with stats_tracker.record_timing("recompute_logp"):
                    logp = self.actor.compute_logp(batch)
                    batch["prox_logp"] = logp
                    log_gpu_stats("recompute logp")

            if self.ref is not None:
                with stats_tracker.record_timing("ref_logp"):
                    batch["ref_logp"] = self.ref.compute_logp(batch)
                    log_gpu_stats("ref logp")

            with stats_tracker.record_timing("compute_advantage"):
                self.actor.compute_advantages(batch)
                log_gpu_stats("compute advantages")

            with stats_tracker.record_timing("train_step"):
                self.actor.ppo_update(batch)
                self.actor.step_lr_scheduler()
                log_gpu_stats("ppo update")

            # pause inference for updating weights, save, and evaluation
            self.rollout.pause()

            with stats_tracker.record_timing("update_weights"):
                self.actor.update_weights(self.weight_update_meta)

                self.actor.set_version(global_step + 1)
                self.rollout.set_version(global_step + 1)
                self.eval_rollout.set_version(global_step + 1)

            with stats_tracker.record_timing("save"):
                self.saver.save(
                    self.actor, epoch, step, global_step, tokenizer=self.tokenizer
                )

            with stats_tracker.record_timing("checkpoint_for_recover"):
                self.recover_handler.dump(
                    self.actor,
                    step_info,
                    self.saver,
                    self.evaluator,
                    self.stats_logger,
                    self.train_dataloader,
                    tokenizer=self.tokenizer,
                )

            current_platform.synchronize()
            dist.barrier(group=self.actor.cpu_group)

            with stats_tracker.record_timing("eval"):

                def evaluate_fn():
                    if self.valid_dataloader is None:
                        # skip evaluation if no validation dataset is provided
                        if self.actor.is_data_parallel_head():
                            logger.info(
                                "No validation dataset found, skipping evaluation"
                            )
                        return

                    if self.actor.is_data_parallel_head():
                        cnt = 0
                        for data in self.valid_dataloader:
                            for item in data:
                                self.eval_rollout.submit(item, eval_workflow)
                                cnt += 1
                        self.eval_rollout.wait(cnt, timeout=None)
                    current_platform.synchronize()
                    dist.barrier(group=self.actor.cpu_group)

                self.evaluator.evaluate(
                    evaluate_fn,
                    epoch,
                    step,
                    global_step,
                )

            current_platform.synchronize()
            dist.barrier(group=self.actor.cpu_group)

            # Upload statistics to the logger (e.g., wandb)
            stats = stats_tracker.export_all(
                reduce_group=self.actor.data_parallel_group
            )
            self.stats_logger.commit(epoch, step, global_step, stats)

            current_platform.synchronize()
            dist.barrier(group=self.actor.cpu_group)

            # Resume rollout
            self.rollout.resume()

    def close(self):
        self.stats_logger.close()
        self.eval_rollout.destroy()
        self.rollout.destroy()
        if self.ref is not None:
            self.ref.destroy()
        self.actor.destroy()

    def _create_dataloader(
        self, dataset: Dataset | None, split: str = "train"
    ) -> StatefulDataLoader | None:
        if getattr(self, "actor", None) is None:
            raise ValueError("Train engine is not created")

        if dataset is None:
            return None

        dataset_config = (
            self.config.train_dataset if split == "train" else self.config.valid_dataset
        )
        return create_dataloader(
            dataset,
            rank=self.actor.data_parallel_rank,
            world_size=self.actor.data_parallel_world_size,
            dataset_config=dataset_config,
        )

    def _create_train_engine(
        self, actor_config: PPOActorConfig
    ) -> FSDPPPOActor | MegatronPPOActor:
        # Initialize train engine
        if self.train_backend == "fsdp":
            actor = FSDPPPOActor(config=actor_config)
        elif self.train_backend == "megatron":
            actor = MegatronPPOActor(config=actor_config)
        else:
            raise ValueError(
                f"Invalid backend: {self.train_backend}, expected fsdp or megatron"
            )
        actor.create_process_group(parallel_strategy=self.parallel_strategy)
        return actor

    def _init_inference_engine(
        self, rollout_config: InferenceEngineConfig, is_eval: bool = False
    ) -> InferenceEngine:
        # Initialize inference engine
        if self.inference_backend == "sglang":
            engine = RemoteSGLangEngine(deepcopy(rollout_config))
        elif self.inference_backend == "vllm":
            engine = RemotevLLMEngine(deepcopy(rollout_config))
        else:
            raise ValueError(
                f"Invalid backend: {self.inference_backend}, expected sglang or vllm"
            )

        if is_eval:
            # NOTE: eval does not have any offpolicyness control
            engine.config.max_head_offpolicyness = int(1e12)
        kwargs = (
            {}
            if is_eval
            else {"train_data_parallel_size": self.parallel_strategy.dp_size}
        )
        engine.initialize(**kwargs)
        return engine

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        if exc_type is not None:
            raise exc_value
