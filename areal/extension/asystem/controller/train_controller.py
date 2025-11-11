"""ASystem TrainController implementation.

This module provides the ASystem-specific TrainController that inherits from
the base TrainController and overrides the initialize method.
"""

import asyncio
import torch

from torch import Tensor
from collections.abc import Callable
from typing import Any
from areal.extension.asystem.api.cli_args import TrainEngineConfig
from areal.api.engine_api import TrainEngine
from areal.api.io_struct import FinetuneSpec
from areal.api.scheduler_api import Job, Scheduler
from areal.controller.train_controller import TrainController as BaseTrainController
from areal.extension.asystem.controller.util import execute_parallel_tasks, calc_metrics
from areal.extension.asystem.remote_hybrid_train_worker import RemoteMegatronInitConfig
from areal.utils import logging, stats_tracker
from areal.controller.batch import DistributedBatch
from areal.api.io_struct import AllocationMode, SaveLoadMeta, WeightUpdateMeta

logger = logging.getLogger("TrainController")


class TrainController(BaseTrainController):
    """ASystem-specific TrainController.

    This controller inherits from the base TrainController and overrides
    the initialize method to provide ASystem-specific initialization behavior.
    """

    def __init__(
        self,
        train_engine: type[TrainEngine],
        config: TrainEngineConfig,
        scheduler: Scheduler,
    ):
        """Initialize the ASystem TrainController.

        Parameters
        ----------
        train_engine : type[TrainEngine]
            The engine class (not instance) to instantiate on each worker
        config : TrainEngineConfig
            Configuration for training engines
        scheduler : Scheduler
            Scheduler for worker management
        """
        super().__init__(train_engine, config, scheduler)

    def initialize(
        self,
        role: str,
        alloc_mode: AllocationMode,
        ft_spec: FinetuneSpec,
        **kwargs,
    ):
        """Initialize environments for distributed training and load models.

        This method is overridden to provide ASystem-specific initialization behavior.
        Currently, it passes without performing any initialization.

        Parameters
        ----------
        role : str
            Role identifier for the workers
        alloc_mode : AllocationMode
            Allocation mode configuration for distributed setup
        ft_spec : FinetuneSpec
            Finetune specification for model initialization
        **kwargs
            Additional keyword arguments passed to engine initialization
        """
        self.logger = logging.getLogger("[TrainController]")

        # Store configuration
        self.parallel_strategy = alloc_mode.train
        self._worker_role = role
        self.alloc_mode = alloc_mode
        self.world_size = self.alloc_mode.train.world_size
        self.dp_size = self.alloc_mode.train.dp_size
        self.tp_size = self.alloc_mode.train.tp_size
        self.pp_size = self.alloc_mode.train.pp_size
        self.group_size = kwargs.get("group_size")
        self.enable_colocate_mode = kwargs.get("enable_colocate_mode")
        self.storage_prefix = kwargs.get("storage_prefix")

        # Create job for scheduler
        job = Job(
            replicas=alloc_mode.train.world_size,
            tasks=list(self.config.scheduling_specs),
            scheduling_strategy=self.config.scheduling_strategy,
            role=self._worker_role,
        )

        # Create workers via scheduler
        self.logger.info("Creating workers via scheduler...")
        worker_ids = self.scheduler.create_workers(job=job)
        self.logger.info(f"Workers created: {worker_ids}")

        # Wait for workers to be ready
        self.logger.info("Waiting for workers to be ready...")
        self.workers = self.scheduler.get_workers(role=job.role)
        self.logger.info(f"Workers ready: {[w.id for w in self.workers]}")

        # Get engine class path for dynamic import on workers
        engine_class = self.train_engine
        engine_path = f"{engine_class.__module__}.{engine_class.__name__}"

        # Create and initialize engines on workers
        asyncio.run(self._async_create_engines(engine_path))
        asyncio.run(self._async_initialize(job, ft_spec, **kwargs))

        self.logger.info("TrainController initialization complete")

    async def _async_initialize(self, job: Job, ft_spec: FinetuneSpec, **kwargs):
        # Initialize engines
        self.logger.info("Calling engine initialization...")
        init_configs = self._build_engine_initialize_config(
            enable_colocate_mode=job.scheduling_strategy.type == "colocation"
        )

        assert len(init_configs) == len(self.workers)

        tasks = [
            self.scheduler.async_call_engine(
                worker.id, "initialize", init_config, _should_bcast=False
            )
            for worker, init_config in zip(self.workers, init_configs)
        ]

        self.rank_info = {}
        try:
            gather_results = await asyncio.gather(*tasks, return_exceptions=False)
        except Exception as e:
            self.logger.error(f"Initialization failed with error: {e}")
            raise RuntimeError(f"Failed to initialize workers, error: {e}")

        for worker_index, result in enumerate(gather_results):
            self.rank_info[worker_index] = result
            self.logger.info(f"Worker {worker_index} succeeded: {result}")

        self.logger.info("All engines are initialized!")

    def _build_engine_initialize_config(
        self, enable_colocate_mode: bool
    ) -> list[RemoteMegatronInitConfig]:
        server_addrs = [
            f"{worker.ip}:{worker.engine_ports[0]}" for worker in self.workers
        ]
        return [
            RemoteMegatronInitConfig(
                server_addrs=server_addrs,
                global_rank=index,
                world_size=self.alloc_mode.train.world_size,
                enable_colocate_mode=enable_colocate_mode,
            )
            for index, worker in enumerate(self.workers)
        ]

    def train_batch(
        self,
        input_: DistributedBatch,
        loss_fn: Callable[[torch.Tensor, dict[str, Any]], torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
    ) -> dict[str, float]:
        self.logger.info(f"start to train_batch")
        with (stats_tracker.record_timing("train_batch_data_split"), ):
            batches = input_.chunk_by_ffd(self.group_size, self.dp_size)

        calc_metrics(batches)

        tasks = [
            self.scheduler.async_call_engine(
                worker.id, "train_batch", batches[self.rank_info[index]["dp_rank"]], _should_bcast=False
            )
            for index, worker in enumerate(self.workers)
        ]

        try:
            results = asyncio.run(asyncio.gather(*tasks, return_exceptions=False))
        except KeyboardInterrupt:
            raise
        except Exception as e:
            raise RuntimeError(f"train_batch failed, error: {e}")

        for worker_result in results:
            if len(worker_result) > 1:
                for minibatch in worker_result:
                    stats_tracker.scalar(**minibatch)
            else:
                stats_tracker.scalar(**worker_result[0])

        return {}

    def compute_logp(self, input_: DistributedBatch) -> Tensor:
        """Update the model with a batch of data and a loss function."""
        logger.info(f"start to compute_logp")
        with (
            stats_tracker.record_timing("compute_logp_data_split"),
        ):
            batches = input_.chunk(self.dp_size)
            tasks = [
                self.scheduler.async_call_engine(
                    worker.id, "compute_logprobs", batches[self.rank_info[index]["dp_rank"]], _should_bcast=False
                )
                for index, worker in enumerate(self.workers)
            ]

        try:
            results = asyncio.run(asyncio.gather(*tasks, return_exceptions=False))
        except KeyboardInterrupt:
            raise
        except Exception as e:
            raise RuntimeError(f"compute_logp failed, error: {e}")

        # cat tensor from dp head with padding
        tensors_from_dp_heads = results[: self.dp_size]
        if not tensors_from_dp_heads:
            return torch.tensor([])

        # Find max length in dim 1
        max_len = max(t.shape[1] for t in tensors_from_dp_heads)
        max_len_all = max(t.shape[1] for t in results)
        assert max_len_all == max_len
        # Pad all tensors to max length
        padded_tensors = []
        for t in tensors_from_dp_heads:
            pad_size = max_len - t.shape[1]
            padded = torch.nn.functional.pad(t, (0, pad_size), value=0.0)
            padded_tensors.append(padded)

        # Concatenate along batch dimension
        concatenated_result = torch.cat(padded_tensors, dim=0)
        return concatenated_result

    def upload_weights(self, meta: WeightUpdateMeta):
        """Upload weights to the inference engine."""
        self.logger.info("begin upload_weights")
        execute_parallel_tasks(self.workers, self.scheduler, "upload_weights", meta)
        self.logger.info("finished upload_weights")

    def save(self, meta: SaveLoadMeta):
        """Save model weights (and optimizer states) for later use."""
        execute_parallel_tasks(self.workers, self.scheduler, "save", meta)

    def load(self, meta: SaveLoadMeta):
        """Load model weights and optimizer states from a file."""
        execute_parallel_tasks(self.workers, self.scheduler, "load", meta)

    def notify_event(self, event: str, global_step: int) -> None:
        """Notify workers about training start/end events.

        Args:
            event: "train_start" or "train_end"
            global_step: Current global step
        """
        execute_parallel_tasks(self.workers, self.scheduler, "notify_event", event, global_step)
        return None
