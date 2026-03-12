"""Trainer for self-distillation (SDPO).

Orchestrates: rollout collection, student/teacher forward, distillation loss, weight update,
and checkpointing.
"""

from __future__ import annotations

import os
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import torch.distributed as dist

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import (
    InferenceEngineConfig,
    SchedulingStrategy,
    SchedulingStrategyType,
    SelfDistillActorConfig,
    SelfDistillConfig,
    SGLangConfig,
    vLLMConfig,
)
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import FinetuneSpec, StepInfo, WeightUpdateMeta
from areal.api.scheduler_api import Scheduler
from areal.api.workflow_api import WorkflowLike
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.engine.vllm_remote import RemotevLLMEngine
from areal.infra import (
    LocalScheduler,
    RayScheduler,
    RolloutController,
    SlurmScheduler,
    current_platform,
)
from areal.utils import logging, perf_tracer, seeding, stats_tracker
from areal.utils.environ import is_single_controller
from areal.utils.hf_utils import load_hf_processor_and_tokenizer
from areal.utils.perf_tracer import Category
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger

if TYPE_CHECKING:
    from areal.engine.fsdp_engine import FSDPDistillActor
    from areal.engine.megatron_engine import MegatronDistillActor
    from areal.experimental.engine.archon_engine import ArchonDistillActor

logger = logging.getLogger("SelfDistillationTrainer")


class _EmptyDataLoader:
    """Minimal dataloader for self-distill mode that yields empty dicts."""

    def __init__(self, batch_size: int = 1, steps_per_epoch: int = 1):
        self.batch_size = batch_size
        self._steps_per_epoch = steps_per_epoch

    def __len__(self) -> int:
        return self._steps_per_epoch

    def __iter__(self):
        while True:
            yield [{} for _ in range(self.batch_size)]

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state_dict: dict) -> None:  # noqa: ARG002
        pass


class SelfDistillationTrainer:
    """Trainer for Self-Distilled Policy Optimization (SDPO).

    Orchestrates the self-distillation training loop:
    1. Collect extended trajectories from the distillation buffer
    2. Compute student and teacher log-probs
    3. Compute self-distillation loss (KL divergence variant)
    4. Update weights and sync to inference engine
    5. Checkpoint

    Parameters
    ----------
    config : SelfDistillConfig
        Top-level experiment configuration.
    """

    def __init__(
        self, config: SelfDistillConfig, train_dataset=None, valid_dataset=None
    ):
        rank = int(os.getenv("RANK", "0"))
        if is_single_controller():
            logging.setup_file_logging(StatsLogger.get_log_path(config.stats_logger))

        self.config = config
        self.processor, self.tokenizer = load_hf_processor_and_tokenizer(
            config.tokenizer_path
        )
        self.scheduler = None
        if is_single_controller():
            self.scheduler = self._init_scheduler()

        # Set seed.
        seeding.set_random_seed(config.seed, key=f"distill_trainer{rank}")

        # Parse allocation mode.
        self.allocation_mode = AllocationMode.from_str(config.allocation_mode)

        self._amend_xccl_weight_update_envvar()

        # Create actor (no critic, no ref).
        self.actor = self._create_actor(config.actor)

        # Dataloader: use provided dataset for offline mode, or empty loader for online.
        if train_dataset is not None:
            from areal.utils.dataloader import create_dataloader

            self.train_dataloader = create_dataloader(
                train_dataset,
                rank=self.actor.data_parallel_rank,
                world_size=self.actor.data_parallel_world_size,
                dataset_config=config.train_dataset,
            )
        else:
            if config.total_train_steps is None:
                raise ValueError(
                    "total_train_steps must be set when no train_dataset is provided."
                )
            steps_per_epoch = config.total_train_steps // config.total_train_epochs
            if steps_per_epoch < 1:
                raise ValueError(
                    f"total_train_steps ({config.total_train_steps}) must be >= "
                    f"total_train_epochs ({config.total_train_epochs})."
                )
            self.train_dataloader = _EmptyDataLoader(
                batch_size=config.batch_size,
                steps_per_epoch=steps_per_epoch,
            )
        self.valid_dataset = valid_dataset

        ft_spec = FinetuneSpec(
            total_train_epochs=config.total_train_epochs,
            dataset_size=len(self.train_dataloader) * config.batch_size,
            train_batch_size=config.batch_size,
        )

        self.parallel_strategy = self.allocation_mode.train
        assert self.parallel_strategy is not None
        engine_init_kwargs = {
            "addr": None,
            "ft_spec": ft_spec,
            "alloc_mode": self.allocation_mode,
        }
        self.actor.initialize(**engine_init_kwargs, role="actor")

        # Initialize inference engine
        self.rollout = self._init_rollout(config.rollout, is_eval=False)

        # Prepare weight update meta
        if self.config.actor.weight_update_mode == "disk":
            disk_kwargs = {
                "experiment_name": config.experiment_name,
                "trial_name": config.trial_name,
                "file_root": config.cluster.fileroot,
                "name": "default",
                "clear_checkpoint_after_load": True,
            }
            self.weight_update_meta = WeightUpdateMeta.from_disk(**disk_kwargs)
        elif self.config.actor.weight_update_mode == "xccl":
            if self.allocation_mode.train_backend == "megatron":
                self.weight_update_meta = WeightUpdateMeta.from_megatron_xccl(
                    self.allocation_mode
                )
            else:
                self.weight_update_meta = WeightUpdateMeta.from_fsdp_xccl(
                    allocation_mode=self.allocation_mode
                )
        else:
            raise ValueError(
                f"Invalid weight update mode: {self.config.actor.weight_update_mode}"
            )
        self.actor.connect_engine(self.rollout, self.weight_update_meta)

        # Set up save, recover, stats
        self.saver = Saver(config.saver, ft_spec)
        self.recover_handler = RecoverHandler(config.recover, ft_spec)
        self.stats_logger = StatsLogger(config, ft_spec)

        # Proxy worker state
        self._proxy_started = False

        # Load recover checkpoint
        self.recover_info = self.recover_handler.load(
            self.actor,
            self.saver,
            None,  # no evaluator
            self.stats_logger,
            self.train_dataloader,
            inference_engine=self.rollout,
            weight_update_meta=self.weight_update_meta,
        )

    def train(
        self,
        workflow: WorkflowLike | None = None,
        workflow_kwargs: dict[str, Any] | None = None,
        total_epochs: int | None = None,
    ):
        """Run the self-distillation training loop.

        Parameters
        ----------
        workflow : WorkflowLike, optional
            The workflow for rollout.  If None, online mode with
            ``self_distill`` proxy workflow is assumed.
        workflow_kwargs : dict, optional
            Extra kwargs for workflow construction.
        total_epochs : int, optional
            Override ``config.total_train_epochs``.
        """
        config = self.config
        start_step = (
            self.recover_info.last_step_info.next().global_step
            if self.recover_info is not None
            else 0
        )

        if total_epochs is None:
            total_epochs = config.total_train_epochs
        steps_per_epoch = len(self.train_dataloader)
        max_steps = total_epochs * steps_per_epoch

        # Start proxy workers
        self._ensure_proxy_started()

        for global_step in range(start_step, max_steps):
            if (
                config.total_train_steps is not None
                and global_step >= config.total_train_steps
            ):
                break
            epoch = global_step // steps_per_epoch
            step = global_step % steps_per_epoch

            # Rollout
            with (
                stats_tracker.record_timing("rollout"),
                perf_tracer.trace_scope(
                    "distill_train.rollout",
                    category=Category.COMPUTE,
                    args={"global_step": global_step},
                ),
            ):
                rollout_batch = self.actor.prepare_batch(
                    self.train_dataloader,
                    workflow=workflow,
                    workflow_kwargs=workflow_kwargs,
                    group_size=config.gconfig.n_samples,
                )

            # Self-distillation update
            self.saver.maybe_wait_for_staging()

            with (
                stats_tracker.record_timing("reorg_batch"),
                perf_tracer.trace_scope(
                    "distill_train.reorg_batch",
                    category=Category.MISC,
                    args={"global_step": global_step},
                ),
            ):
                batch = self.actor.reorg_batch(rollout_batch)
            with (
                stats_tracker.record_timing("compute_teacher_logp"),
                perf_tracer.trace_scope(
                    "distill_train.compute_teacher_logp",
                    category=Category.COMPUTE,
                    args={"global_step": global_step},
                ),
            ):
                teacher_logp = self.actor.compute_teacher_logp(batch)
                batch['teacher_logp'] = teacher_logp
                
            with (
                stats_tracker.record_timing("distill_update"),
                perf_tracer.trace_scope(
                    "distill_train.self_distill_update",
                    category=Category.COMPUTE,
                    args={"global_step": global_step},
                ),
            ):
                self.actor.self_distill_update(rollout_batch)
                self.actor.step_lr_scheduler()
                self.actor.get_device_stats().log("distill update")

            # Weight update
            self.rollout.pause()

            with (
                stats_tracker.record_timing("update_weights"),
                perf_tracer.trace_scope(
                    "distill_train.update_weights",
                    category=Category.COMM,
                    args={"global_step": global_step},
                ),
            ):
                new_version = global_step + 1
                versioned_meta = self.weight_update_meta.with_version(new_version)
                self.actor.update_weights(versioned_meta)
                self.actor.set_version(new_version)
                self.rollout.set_version(new_version)

            # Save
            with (
                stats_tracker.record_timing("save"),
                perf_tracer.trace_scope(
                    "distill_train.save",
                    category=Category.IO,
                    args={"global_step": global_step},
                ),
            ):
                self._save_hf(epoch=epoch, epoch_step=step, global_step=global_step)

            with (
                stats_tracker.record_timing("checkpoint_for_recover"),
                perf_tracer.trace_scope(
                    "distill_train.checkpoint",
                    category=Category.IO,
                    args={"global_step": global_step},
                ),
            ):
                self._save_recover_checkpoint(
                    epoch=epoch, epoch_step=step, global_step=global_step
                )

            # Log stats
            with perf_tracer.trace_scope(
                "distill_train.log_stats",
                category=Category.INSTR,
                args={"global_step": global_step},
            ):
                self._export_and_commit_stats(
                    epoch=epoch, epoch_step=step, global_step=global_step
                )

            # Resume
            self.actor.clear_batches(rollout_batch)
            self.rollout.resume()

    # Private helpers

    def _save_hf(self, epoch: int, epoch_step: int, global_step: int):
        self.saver.save(
            self.actor,
            epoch,
            epoch_step,
            global_step,
            tokenizer=self.tokenizer,
            processor=self.processor,
        )
        if not self.saver.is_async:
            dist.barrier(group=self.actor.cpu_group)
            current_platform.synchronize()

    def _save_recover_checkpoint(self, epoch: int, epoch_step: int, global_step: int):
        to_save: dict = dict(default=self.actor)
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
            None,  # no evaluator
            self.stats_logger,
            self.train_dataloader,
            tokenizer=self.tokenizer,
            processor=self.processor,
        )
        dist.barrier(group=self.actor.cpu_group)
        current_platform.synchronize()

    def _export_and_commit_stats(self, epoch: int, epoch_step: int, global_step: int):
        stats = self.actor.export_stats()
        stats.update(self.rollout.export_stats())
        self.stats_logger.commit(epoch, epoch_step, global_step, stats)
        dist.barrier(group=self.actor.cpu_group)
        current_platform.synchronize()

    def _init_scheduler(self) -> Scheduler:
        cfg = self.config.scheduler
        if cfg.type == "local":
            return LocalScheduler(exp_config=self.config)
        elif cfg.type == "ray":
            return RayScheduler(exp_config=self.config)
        elif cfg.type == "slurm":
            return SlurmScheduler(exp_config=self.config)
        raise NotImplementedError(f"Unknown scheduler type: {cfg.type}")

    def _create_actor(
        self, actor_config: SelfDistillActorConfig
    ) -> FSDPDistillActor | MegatronDistillActor | ArchonDistillActor:
        if self.allocation_mode.train_backend == "fsdp":
            from areal.engine.fsdp_engine import FSDPDistillActor

            actor_cls = FSDPDistillActor
        elif self.allocation_mode.train_backend == "megatron":
            from areal.engine.megatron_engine import MegatronDistillActor

            actor_cls = MegatronDistillActor
        elif self.allocation_mode.train_backend == "archon":
            from areal.experimental.engine.archon_engine import ArchonDistillActor

            actor_cls = ArchonDistillActor
        else:
            raise ValueError(f"Invalid backend: {self.allocation_mode.train_backend}")
        if is_single_controller():
            actor = actor_cls.as_controller(actor_config, self.scheduler)
        else:
            actor = actor_cls(config=actor_config)
        actor.create_process_group(parallel_strategy=self.allocation_mode.train)
        return actor

    def _init_rollout(
        self,
        rollout_config: InferenceEngineConfig,
        is_eval: bool = False,
    ) -> InferenceEngine | RolloutController:
        config = deepcopy(rollout_config)
        if is_eval:
            config.max_head_offpolicyness = int(1e12)
            config.scheduling_strategy = SchedulingStrategy(
                type=SchedulingStrategyType.colocation, target="rollout"
            )
            for spec in config.scheduling_spec:
                spec.gpu = 0

        if self.allocation_mode.gen_backend == "sglang":
            engine_cls = RemoteSGLangEngine
            server_args = SGLangConfig.build_args(
                sglang_config=self.config.sglang,
                tp_size=self.allocation_mode.gen.tp_size,
                base_gpu_id=0,
            )
        elif self.allocation_mode.gen_backend == "vllm":
            engine_cls = RemotevLLMEngine
            server_args = vLLMConfig.build_args(
                vllm_config=self.config.vllm,
                tp_size=self.allocation_mode.gen.tp_size,
                pp_size=self.allocation_mode.gen.pp_size,
            )
        else:
            raise ValueError(f"Invalid backend: {self.allocation_mode.gen_backend}")

        if not is_single_controller():
            engine = engine_cls(config)
            engine.initialize(
                train_data_parallel_size=self.allocation_mode.train.dp_size
            )
            return engine

        controller = engine_cls.as_controller(config, self.scheduler)
        init_kwargs = dict(
            role="rollout",
            alloc_mode=self.allocation_mode,
            server_args=server_args,
        )
        controller.initialize(**init_kwargs)
        return controller

    def _amend_xccl_weight_update_envvar(self):
        if not is_single_controller():
            return
        if self.allocation_mode.gen_backend != "sglang":
            return
        for spec in self.config.actor.scheduling_spec:
            spec.env_vars["NCCL_CUMEM_ENABLE"] = "0"
            spec.env_vars["NCCL_NVLS_ENABLE"] = "0"

    def _ensure_proxy_started(self) -> None:
        """Start proxy workers when using online (proxy) mode."""
        if self._proxy_started:
            return

        # Offline mode with real dataset: no proxy needed.
        if not isinstance(self.train_dataloader, _EmptyDataLoader):
            self._proxy_started = True
            return

        if not is_single_controller():
            raise NotImplementedError("Proxy workers not supported in SPMD mode")

        if self.config.scheduler.type == "ray":
            raise NotImplementedError("Proxy workers not supported with RayScheduler")

        assert isinstance(self.rollout, RolloutController)

        logger.info("Initializing proxy workers for self-distillation")
        self.rollout.start_proxy()
        self.rollout.start_proxy_gateway()
        logger.info("Proxy gateway available at %s", self.rollout.proxy_gateway_addr)
        self._proxy_started = True

    def close(self):
        """Clean up all resources."""
        self.saver.finalize()
        self.stats_logger.close()
        self.rollout.destroy()
        self.actor.destroy()
        perf_tracer.save(force=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            logger.error(f"Training failed with exception: {exc_value}", exc_info=True)
        self.close()
        if exc_type is not None:
            raise
