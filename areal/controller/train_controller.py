import asyncio
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp
import torch.distributed as dist
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.alloc_mode import ParallelStrategy
from areal.api.cli_args import TrainEngineConfig
from areal.api.engine_api import TrainEngine
from areal.api.io_struct import (
    AllocationMode,
    FinetuneSpec,
    SaveLoadMeta,
    WeightUpdateMeta,
)
from areal.api.scheduler_api import Job, Scheduler, Worker
from areal.api.workflow_api import RolloutWorkflow
from areal.controller.rollout_controller import RolloutController
from areal.platforms import current_platform
from areal.scheduler.rpc.rtensor import RTensor
from areal.utils import logging, name_resolve, names, stats_tracker
from areal.utils.network import find_free_ports

logger = logging.getLogger(__name__)


class TrainController:
    """Controller for managing distributed training across multiple workers.

    This class orchestrates the lifecycle of training workers, handles data
    distribution across data-parallel groups, and provides a unified interface
    for training operations. It manages worker creation, engine initialization,
    and coordinates method calls across distributed workers.

    The controller automatically handles:
    - Worker creation and lifecycle management via scheduler
    - Data splitting across data-parallel groups
    - Result merging from multiple workers
    - Distributed training configuration (MASTER_ADDR, MASTER_PORT)
    """

    def __init__(
        self,
        train_engine: type[TrainEngine],
        config: TrainEngineConfig,
        scheduler: Scheduler,
    ):
        self.train_engine = train_engine
        self.config = config
        self.scheduler = scheduler

        self.alloc_mode: AllocationMode
        self.workers: list[Worker] = []
        # Boolean list indicating which workers are data-parallel heads
        # Only DP head workers receive data slices; others get data via broadcast
        self.workers_is_dp_head: list[bool] = []
        self.parallel_strategy: ParallelStrategy | None = None

        self._worker_role: str = "default"

        self.rollout: RolloutController = None
        self.weight_update_group_initialized = False

    def create_process_group(self, parallel_strategy: ParallelStrategy | None = None):
        """Placeholder method for process group creation.

        This is a dummy method maintained for API compatibility. The actual
        process group creation happens during `initialize()` when engines are
        initialized on workers.

        Parameters
        ----------
        parallel_strategy : ParallelStrategy | None, optional
            Parallel strategy configuration (currently unused), by default None
        """
        port = find_free_ports(1)[0]
        dist.init_process_group(
            backend="gloo", init_method=f"tcp://localhost:{port}", rank=0, world_size=1
        )

    @property
    def data_parallel_rank(self) -> int:
        return 0

    @property
    def data_parallel_world_size(self) -> int:
        return 1

    def is_data_parallel_head(self) -> bool:
        return True

    @property
    def cpu_group(self):
        return None

    def initialize(
        self,
        role: str,
        alloc_mode: AllocationMode,
        ft_spec: FinetuneSpec,
        **kwargs,
    ):
        """Initialize environments for distributed training and load models.

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
        # Store configuration
        self._worker_role = role
        self.alloc_mode = alloc_mode

        self.parallel_strategy = alloc_mode.train

        # Create job specification for scheduler
        # Convert scheduling_spec tuple to list for scheduler compatibility
        # The scheduler will handle task replication across workers if needed
        job = Job(
            replicas=alloc_mode.train.world_size,
            tasks=list(self.config.scheduling_spec),
            scheduling_strategy=self.config.scheduling_strategy,
            role=self._worker_role,
        )

        # Create workers via scheduler
        logger.info("Creating workers via scheduler...")
        worker_ids = self.scheduler.create_workers(job=job)
        logger.info(f"Workers created: {worker_ids}")

        # Wait for workers to be ready
        logger.info("Waiting for workers to be ready...")
        self.workers = self.scheduler.get_workers(role=job.role)
        logger.info(f"Workers ready: {[w.id for w in self.workers]}")

        # Determine distributed training master address and port from rank 0 worker
        # These are used for PyTorch distributed initialization across workers
        # Prefer engine_ports[1] if available, fallback to worker_ports[1]
        rank0_worker = self.workers[0]
        if rank0_worker.engine_ports:
            self._master_port = int(rank0_worker.engine_ports[1])
        else:
            self._master_port = int(rank0_worker.worker_ports[1])
        self._master_addr = rank0_worker.ip

        logger.info(
            f"Distributed training: MASTER_ADDR={self._master_addr}, MASTER_PORT={self._master_port}"
        )

        # Construct engine class import path for dynamic loading on workers
        # Workers will import and instantiate the engine class using this path
        engine_class = self.train_engine
        engine_path = f"{engine_class.__module__}.{engine_class.__name__}"

        # Create and initialize engines on workers
        self._run_async_task(self._async_create_engines(engine_path))
        self._run_async_task(self._async_initialize_engines(ft_spec, **kwargs))

        # Identify DP head workers
        self._identify_dp_heads()
        logger.info("TrainController initialization complete")

    def _run_async_task(self, task):
        """Run an async task synchronously."""
        return asyncio.run(task)

    async def _async_create_engines(self, engine_path: str):
        """Create engine instances on all workers. Sets distributed env vars before creation."""
        logger.info("Creating engines on workers...")

        async def _setup_worker(worker: Worker, rank: int):
            env = {
                "RANK": str(rank),
                "WORLD_SIZE": str(len(self.workers)),
                "MASTER_ADDR": str(self._master_addr),
                "MASTER_PORT": str(self._master_port),
                "LOCAL_RANK": "0",  # NOTE: local rank is always 0 while each process use only one GPU
            }
            await self.scheduler.set_worker_env(worker.id, env)
            await self.scheduler.create_engine(
                worker_id=worker.id,
                engine=engine_path,
                config=self.config,
            )

        tasks = [
            _setup_worker(worker, rank) for rank, worker in enumerate(self.workers)
        ]
        await asyncio.gather(*tasks)
        logger.info("Engines created on all workers!")

    async def _async_initialize_engines(self, ft_spec: FinetuneSpec, **kwargs):
        """Initialize engines: create process groups, then load models and setup optimizers."""
        logger.info("Calling engine initialization...")
        # Phase 1: Create process groups for distributed training
        tasks = [
            self.scheduler.async_call_engine(
                worker_id=worker.id,
                method="create_process_group",
                parallel_strategy=self.parallel_strategy,
                should_broadcast=False,
            )
            for worker in self.workers
        ]
        await asyncio.gather(*tasks)
        # Phase 2: Initialize engines (load models, setup optimizers, etc.)
        tasks = [
            self.scheduler.async_call_engine(
                worker_id=worker.id,
                method="initialize",
                ft_spec=ft_spec,
                should_broadcast=False,
                **kwargs,
            )
            for worker in self.workers
        ]
        await asyncio.gather(*tasks)
        logger.info("All engines are initialized!")

    def _identify_dp_heads(self):
        """Query workers to identify DP heads. Stores result in self.workers_is_dp_head."""
        logger.info("Identifying DP head workers...")

        async def _get_dp_head():
            tasks = [
                self.scheduler.async_call_engine(
                    worker_id=worker.id, method="is_data_parallel_head"
                )
                for worker in self.workers
            ]
            return await asyncio.gather(*tasks)

        self.workers_is_dp_head = self._run_async_task(_get_dp_head())

    def destroy(self):
        """Destroy the controller and release GPU memory of models.

        Cleans up all resources including workers, engines, and internal state.
        """
        logger.info("Destroying TrainController...")

        # First destroy engines to release GPU memory
        if self.workers:
            logger.info("Destroying engines on all workers...")
            try:

                async def _destroy_all_engines():
                    tasks = [
                        self.scheduler.async_call_engine(worker.id, "destroy")
                        for worker in self.workers
                    ]
                    await asyncio.gather(*tasks, return_exceptions=True)

                self._run_async_task(_destroy_all_engines())
                logger.info("Engines destroyed")
            except Exception as e:
                logger.error(f"Error destroying engines: {e}")

        # Then delete workers via scheduler
        try:
            logger.info("Deleting all workers...")
            self.scheduler.delete_workers(role=self._worker_role)
            logger.info("Workers deleted")
        except Exception as e:
            logger.error(f"Error deleting workers: {e}")

        # Clear worker lists
        self.workers.clear()
        self.workers_is_dp_head.clear()

        if dist.is_initialized():
            dist.destroy_process_group()
        logger.info("TrainController destroyed")

    def _custom_function_call(self, method: str, *args, **kwargs):
        """Dispatch method call to workers: split batches, replicate args, merge results."""
        dp_split_args, dp_split_kwargs, group_indices = self._dispatch_inputs(
            *args, **kwargs
        )
        results = self._run_async_task(
            self._call_with_dispatched_inputs(method, dp_split_args, dp_split_kwargs)
        )
        # Filter to only keep results from DP head workers
        results = [r for idx, r in enumerate(results) if self.workers_is_dp_head[idx]]
        merged = self._merge_results(results, group_indices)
        return merged

    async def _async_custom_function_call(self, method: str, *args, **kwargs):
        """Async version of _custom_function_call."""
        dp_split_args, dp_split_kwargs, group_indices = self._dispatch_inputs(
            *args, **kwargs
        )
        results = await self._call_with_dispatched_inputs(
            method, dp_split_args, dp_split_kwargs
        )
        # Filter to only keep results from DP head workers
        results = [r for idx, r in enumerate(results) if self.workers_is_dp_head[idx]]
        return self._merge_results(results, group_indices)

    def _dispatch_inputs(self, *args, **kwargs):
        """Split RTensors across DP groups, replicate other args."""
        results, group_indices = RTensor.data_parallel_dispatch(
            (args, kwargs), dp_size=self.parallel_strategy.dp_size
        )
        # results is list of (args_tuple, kwargs_dict) pairs, one per DP group
        # Transpose to match _call_with_dispatched_inputs expectations:
        # dp_split_args[arg_idx][dp_idx] = value for arg_idx-th arg on dp_idx-th group
        # dp_worker_kwargs[key][dp_idx] = value for key kwarg on dp_idx-th group

        dp_size = len(results)
        num_args = len(args)

        # Transpose args: from list of tuples to list of lists
        dp_split_args = [
            [results[dp_idx][0][arg_idx] for dp_idx in range(dp_size)]
            for arg_idx in range(num_args)
        ]

        # Transpose kwargs: from list of dicts to dict of lists
        dp_worker_kwargs = {}
        if kwargs:
            for key in kwargs.keys():
                dp_worker_kwargs[key] = [
                    results[dp_idx][1][key] for dp_idx in range(dp_size)
                ]

        return dp_split_args, dp_worker_kwargs, group_indices

    async def _call_with_dispatched_inputs(
        self,
        method: str,
        dp_split_args: list[list[Any]],
        dp_worker_kwargs: list[dict[str, Any]],
    ):
        """Call method on all workers. DP heads get data slices, others get empty args (broadcast via RPC)."""
        tasks = []
        dp_idx = 0
        for idx, worker in enumerate(self.workers):
            if self.workers_is_dp_head[idx]:
                # Get this DP head worker's slice of each argument
                worker_args = [splits[dp_idx] for splits in dp_split_args]
                worker_kwargs = {
                    k: splits[dp_idx] for k, splits in dp_worker_kwargs.items()
                }
                dp_idx += 1
            else:
                # Non-DP-head workers get empty arguments
                # They will receive data via broadcast in RPC server
                worker_args = []
                worker_kwargs = {}

            tasks.append(
                self.scheduler.async_call_engine(
                    worker.id,
                    method,
                    *worker_args,
                    **worker_kwargs,
                )
            )
        return await asyncio.gather(*tasks)

    def _merge_results(self, results, group_indices):
        """Merge RTensor results from DP heads using RTensor.merge()."""
        return RTensor.data_parallel_merge(results, group_indices)

    def export_stats(self):
        """Export training statistics from all workers.

        Collects statistics from all workers. The statistics are assumed to be
        already aggregated and synchronized (e.g., via all-reduce operations),
        so only the first result is returned.

        Returns
        -------
        dict[str, Any]
            Training statistics dictionary
        """
        # Statistics have been aggregated and synchronized across workers
        # All results should be identical, so return the first one
        stats = stats_tracker.export_all()
        stats.update(self._custom_function_call("export_stats"))
        return stats

    # ==================== ENGINE RPC WRAPPERS ====================
    # Note: Methods like train_batch, forward, etc. are not implemented here.
    # They are expected to be called directly via _custom_function_call in
    # specific training scenarios (PPO, SFT, etc.) where the appropriate
    # loss functions and data processing are handled.
    def train(self, mode: bool = True):
        """Set the engine to training mode.

        Parameters
        ----------
        mode : bool, optional
            Whether to set the engine to training mode, by default True

        Returns
        -------
        TrainController
            Returns self for method chaining
        """
        self._custom_function_call("train", mode)
        return self

    def eval(self):
        """Set the engine to evaluation mode.

        This is a convenience method that calls `self.train(False)`.

        Returns
        -------
        TrainController
            Returns self for method chaining
        """
        return self.train(False)

    def set_version(self, version: int):
        """Set the current weight version in the training engine.

        Parameters
        ----------
        version : int
            The weight version number to set
        """
        self._custom_function_call("set_version", version)

    def get_version(self) -> int:
        """Get the current weight version in the training engine.

        Returns
        -------
        int
            The current weight version number
        """
        return self._custom_function_call("get_version")

    def save(self, meta: SaveLoadMeta):
        """Save model weights and optimizer states for later use.

        Parameters
        ----------
        meta : SaveLoadMeta
            Metadata containing information about where and how to save
        """
        self._custom_function_call("save", meta)

    def load(self, meta: SaveLoadMeta):
        """Load model weights and optimizer states from a file.

        Parameters
        ----------
        meta : SaveLoadMeta
            Metadata containing information about where and how to load
        """
        self._custom_function_call("load", meta)

    def step_lr_scheduler(self):
        """Step the learning rate scheduler.

        Since PPO uses minibatch updates, this method should be called periodically
        (e.g., once per PPO step). It is separated from train_batch to allow
        for more flexible learning rate scheduling.
        """
        self._custom_function_call("step_lr_scheduler")

    def connect_engine(self, rollout: RolloutController, meta: WeightUpdateMeta):
        if self.rollout is not None and self.rollout != rollout:
            logger.warning(
                f"Connected rollout controller changed from {self.rollout} to {rollout}."
            )
        self.rollout = rollout

        if (
            meta.type == current_platform.communication_backend
            and not self.weight_update_group_initialized
        ):
            self._init_weight_update_from_distributed(meta)
            self.weight_update_group_initialized = True

    def get_device_stats(self):
        return self._custom_function_call("get_device_stats")

    def prepare_batch(
        self,
        dataloader: StatefulDataLoader,
        workflow: str,
        workflow_kwargs: dict[str, Any],
        should_accept_fn: str | None = None,
        granularity: int | None = None,
    ) -> dict[str, Any]:
        if granularity is not None:
            logger.warning("For now, granularity takes no effect in train controller.")
        return self.rollout.prepare_batch(
            dataloader=dataloader,
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            should_accept_fn=should_accept_fn,
        )

    def rollout_batch(
        self,
        data: list[dict[str, Any]],
        workflow: RolloutWorkflow | type[RolloutWorkflow] | str,
        workflow_kwargs: dict[str, Any],
        should_accept_fn: str | None = None,
        granularity: int | None = None,
    ) -> dict[str, Any]:
        if granularity is not None:
            logger.warning("For now, granularity takes no effect in train controller.")
        return self.rollout.rollout_batch(
            data=data,
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            should_accept_fn=should_accept_fn,
        )

    def _init_weight_update_from_distributed(self, meta: WeightUpdateMeta):
        raise NotImplementedError()

    def _update_weights_from_distributed(self, meta: WeightUpdateMeta):
        raise NotImplementedError()

    def _update_weights_from_disk(self, meta: WeightUpdateMeta):
        # Update all LocalInfEngine's local weight
        self.save(
            SaveLoadMeta(
                path=meta.path,
                weight_format="hf",
                with_optim=False,
                tokenizer=None,
                processor=None,
            )
        )
        has_model_files = any(child.is_file() for child in Path(meta.path).iterdir())
        assert has_model_files, f"No model files found in {meta.path} after saving."

        update_name = names.update_weights_from_disk(
            self.config.experiment_name,
            self.config.trial_name,
            self.get_version(),
        )
        name_resolve.add(
            update_name,
            str(datetime.now().timestamp()),
            keepalive_ttl=120,
            replace=True,
        )

        meta.clear_checkpoint_after_load = False
        self._run_async_task(self.rollout.update_weights_from_disk(meta))
        shutil.rmtree(meta.path, ignore_errors=True)

    def _check_rollout_engine_connected(self):
        """Validate that rollout engine has been connected via connect_engine()."""
        if self.rollout is None:
            raise RuntimeError(
                "Rollout engine not connected. Call connect_engine()"
                " before using rollout/update_weight methods."
            )

    def update_weights(self, meta: WeightUpdateMeta):
        self._check_rollout_engine_connected()
        if meta.type == current_platform.communication_backend:
            assert self.weight_update_group_initialized
            self._update_weights_from_distributed(meta)
        elif meta.type == "disk":
            self._update_weights_from_disk(meta)
        else:
            raise ValueError(f"Unknown weight update type {meta.type}")

    async def _async_clear_batches(self, *targets: dict[str, RTensor]):
        """Extract shard IDs and call /data/clear on each worker."""
        shards_by_node = RTensor.collect_shards(targets)

        if not shards_by_node:
            return

        async def clear_node(node_addr, shard_ids):
            async with aiohttp.ClientSession() as session:
                async with session.delete(
                    f"http://{node_addr}/data/clear", json={"shard_ids": shard_ids}
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        logger.info(
                            f"Cleared {result.get('cleared_count', 0)} shards on {node_addr}"
                        )

        await asyncio.gather(
            *[clear_node(addr, sids) for addr, sids in shards_by_node.items()],
            return_exceptions=True,
        )

    def clear_batches(self, *targets: dict[str, RTensor]):
        """Clear distributed batch shards from workers to free memory."""
        self._run_async_task(self._async_clear_batches(*targets))
