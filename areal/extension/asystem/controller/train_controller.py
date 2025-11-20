"""ASystem TrainController implementation.

This module provides the ASystem-specific TrainController that inherits from
the base TrainController and overrides the initialize method.
"""

import asyncio
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from areal.api.engine_api import TrainEngine
from areal.api.io_struct import (
    AllocationMode,
    FinetuneSpec,
    SaveLoadMeta,
    WeightUpdateMeta,
)
from areal.api.scheduler_api import Job, Scheduler
from areal.controller.batch import DistributedBatch
from areal.controller.train_controller import TrainController as BaseTrainController
from areal.extension.asystem.api.cli_args import TrainEngineConfig
from areal.extension.asystem.remote_hybrid_train_worker import RemoteMegatronInitConfig
from areal.utils import logging, stats_tracker

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
        self.workers = self.scheduler.get_workers(role=job.role, timeout=1200)
        self.logger.info(f"Workers ready: {[w.id for w in self.workers]}")

        # Get engine class path for dynamic import on workers
        engine_class = self.train_engine
        engine_path = f"{engine_class.__module__}.{engine_class.__name__}"

        # Create and initialize engines on workers
        asyncio.run(self._async_create_engines(engine_path))
        asyncio.run(self._async_initialize(job, ft_spec, **kwargs))

        seen_dp_ranks = set()
        self.workers_is_dp_head = []
        for index in range(len(self.workers)):
            rank_info = self.rank_info[index]
            dp_rank = rank_info["dp_rank"]
            is_dp_head = dp_rank not in seen_dp_ranks
            self.workers_is_dp_head.append(is_dp_head)
            if is_dp_head:
                seen_dp_ranks.add(dp_rank)
        self.logger.info("TrainController initialization complete")

    async def _async_initialize(self, job: Job, ft_spec: FinetuneSpec, **kwargs):
        # Initialize engines
        self.logger.info("Calling engine initialization...")
        init_configs = self._build_engine_initialize_config(
            enable_colocate_mode=kwargs.get("enable_colocate_mode", False)
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
        self.logger.info("start to train_batch")
        results = self._custom_function_call("train_batch", input_, _should_bcast=False)
        for worker_result in results:  # TODO: Get the last pp_rank data from each dp
            if len(worker_result) > 1:
                for minibatch in worker_result:
                    stats_tracker.scalar(**minibatch)
            else:
                stats_tracker.scalar(**worker_result[0])
        self.logger.info(f"train_batch finished, results={results}")
        return {}

    def compute_logp(self, input_: DistributedBatch) -> Tensor:
        """Update the model with a batch of data and a loss function."""
        logger.info("start to compute_logp")
        with (
            stats_tracker.record_timing("compute_logp_data_split"),
        ):
            batches = input_.chunk(self.dp_size)
            tasks = [
                self.scheduler.async_call_engine(
                    worker.id,
                    "compute_logprobs",
                    batches[self.rank_info[index]["dp_rank"]],
                    _should_bcast=False,
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
        """Upload weights to the inference engine - thread-safe for ThreadPoolExecutor calls."""
        self.logger.info("begin upload_weights")
        self._execute_async_task_on_workers("upload_weights", meta)
        self.logger.info("finished upload_weights")

    def save(self, meta: SaveLoadMeta):
        """Save model weights (and optimizer states) for later use."""
        self.logger.info("begin save")
        self._execute_async_task_on_workers("save", meta)
        self.logger.info("finished save")

    def load(self, meta: SaveLoadMeta):
        """Load model weights and optimizer states from a file."""
        return self._execute_async_task_on_workers("load", meta=meta)

    def notify_event(self, event: str, global_step: int) -> None:
        """Notify workers about training start/end events.

        Args:
            event: "train_start" or "train_end"
            global_step: Current global step
        """
        self.logger.info(f"begin notify_event global_step: {global_step}")
        self._execute_async_task_on_workers("notify_event", event, global_step)
        self.logger.info(f"finished notify_event global_step: {global_step}")
        return None

    def _custom_function_call(self, method: str, *args, **kwargs):
        dp_split_args, dp_split_kwargs = self._dispatch_inputs(*args, **kwargs)
        results = self._run_async_task(
            self._call_with_dispatched_inputs(method, dp_split_args, dp_split_kwargs)
        )
        # Only remain data from DP head.
        # results = [r for idx, r in enumerate(results) if self.workers_is_dp_head[idx]]
        return results

    def _align_batches_with_dp(
        self, input_: DistributedBatch, rebalance=True
    ) -> list[DistributedBatch]:
        """Split DistributedBatch across DP groups.

        Returns a list of batches, one for each DP head worker.
        """
        # Handle empty batch by replicating to all DP groups
        if len(input_.get_data()) == 0:
            return [input_] * self.alloc_mode.train.dp_size

        # NOTE: group normalization should be done in workflow
        if rebalance:
            inputs = input_.chunk_by_ffd(self.group_size, self.alloc_mode.train.dp_size)
        else:
            inputs = input_.chunk(self.alloc_mode.train.dp_size)
        return inputs

    async def _call_with_dispatched_inputs(
        self,
        method: str,
        dp_split_args: list[list[Any]],
        dp_worker_kwargs: list[dict[str, Any]],
    ):
        # Call all workers.
        # ONLY DP head workers get their data slice.
        # Other workers will get data by broadcasting in RPC server.
        tasks = []
        for idx, worker in enumerate(self.workers):
            rank_info = self.rank_info[idx]
            dp_rank = rank_info["dp_rank"]
            # Get this worker's slice of each argument
            worker_args = [splits[dp_rank] for splits in dp_split_args]
            worker_kwargs = {
                k: splits[dp_rank] for k, splits in dp_worker_kwargs.items()
            }

            worker_args = [
                arg.get_data() if isinstance(arg, DistributedBatch) else arg
                for arg in worker_args
            ]
            worker_kwargs = {
                k: v.get_data() if isinstance(v, DistributedBatch) else v
                for k, v in worker_kwargs.items()
            }

            tasks.append(
                self.scheduler.async_call_engine(
                    worker.id,
                    method,
                    *worker_args,
                    **worker_kwargs,
                )
            )
        return await asyncio.gather(*tasks)

    def _execute_async_task_on_workers(self, method_name: str, *args, **kwargs):
        def _run_async_in_thread():
            """Run async code in a thread-safe manner."""
            # Always create a new event loop for this thread to avoid conflicts
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:

                async def _async_exec_func():
                    try:
                        self.logger.info(
                            f"Executing {method_name} on {len(self.workers)} workers"
                        )
                        tasks = [
                            self.scheduler.async_call_engine(
                                worker.id,
                                method_name,
                                *args,
                                **kwargs,
                                _should_bcast=False,
                            )
                            for worker in self.workers
                        ]
                        results = await asyncio.gather(*tasks, return_exceptions=True)

                        # Check for exceptions in results
                        for i, result in enumerate(results):
                            if isinstance(result, Exception):
                                self.logger.error(
                                    f"Worker {self.workers[i].id} failed to execute {method_name}: {result}"
                                )
                            else:
                                self.logger.info(
                                    f"Worker {self.workers[i].id} successfully executed {method_name}"
                                )

                        # Re-raise if any exceptions occurred
                        for result in results:
                            if isinstance(result, Exception):
                                raise result

                        return results
                    except Exception as e:
                        self.logger.error(
                            f"Failed to execute {method_name} on workers: {e}"
                        )
                        raise e

                return loop.run_until_complete(_async_exec_func())
            finally:
                # Always close the loop we created
                if not loop.is_closed():
                    loop.close()
                # Clear the event loop for this thread
                asyncio.set_event_loop(None)

        return _run_async_in_thread()
