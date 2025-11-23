import asyncio
from typing import Any

import torch

from areal.api.alloc_mode import ParallelStrategy
from areal.api.cli_args import TrainEngineConfig
from areal.api.controller_api import DistributedBatch
from areal.api.engine_api import TrainEngine
from areal.api.io_struct import (
    AllocationMode,
    FinetuneSpec,
    SaveLoadMeta,
)
from areal.api.scheduler_api import Job, Scheduler, Worker
from areal.controller.batch import DistributedBatchMemory
from areal.utils import logging

logger = logging.getLogger("TrainController")


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

        self._worker_role: str
        self.logger = None

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
        pass

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
        self.logger = logging.getLogger("[TrainController]")

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
        self.logger.info("Creating workers via scheduler...")
        worker_ids = self.scheduler.create_workers(job=job)
        self.logger.info(f"Workers created: {worker_ids}")

        # Wait for workers to be ready
        self.logger.info("Waiting for workers to be ready...")
        self.workers = self.scheduler.get_workers(role=job.role)
        self.logger.info(f"Workers ready: {[w.id for w in self.workers]}")

        # Determine distributed training master address and port from rank 0 worker
        # These are used for PyTorch distributed initialization across workers
        # Prefer engine_ports[1] if available, fallback to worker_ports[1]
        rank0_worker = self.workers[0]
        if rank0_worker.engine_ports:
            self._master_port = int(rank0_worker.engine_ports[1])
        else:
            self._master_port = int(rank0_worker.worker_ports[1])
        self._master_addr = rank0_worker.ip

        self.logger.info(
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

        self.logger.info("TrainController initialization complete")

    def _run_async_task(self, task):
        """Run an async task in a separate thread with a new event loop.

        This is a helper method to execute async operations synchronously.
        It creates a new event loop in a thread pool executor to avoid blocking
        the main thread.

        Parameters
        ----------
        task : coroutine
            The async task (coroutine) to execute

        Returns
        -------
        Any
            The result of the async task
        """

        def _run_in_thread():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(task)
            finally:
                new_loop.close()

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor() as executor:
            future = executor.submit(_run_in_thread)
            return future.result()

    async def _async_create_engines(self, engine_path: str):
        """Create engine instances on all workers with distributed training configuration.

        Parameters
        ----------
        engine_path : str
            Full import path to the engine class (e.g., "areal.engine.fsdp_engine.FSDPEngine")

        Note
        ----
        The rank, world_size, master_addr, and master_port parameters are passed to
        the RPC server, which sets them as environment variables for PyTorch distributed
        initialization. This allows each worker to properly initialize its distributed
        process group.
        """
        self.logger.info("Creating engines on workers...")
        tasks = []
        for rank, worker in enumerate(self.workers):
            tasks.append(
                self.scheduler.create_engine(
                    worker_id=worker.id,
                    engine=engine_path,
                    config=self.config,
                    # These parameters are set as environment variables in RPC server
                    # for PyTorch distributed initialization
                    rank=rank,
                    world_size=len(self.workers),
                    master_addr=self._master_addr,
                    master_port=self._master_port,
                )
            )
        await asyncio.gather(*tasks)
        self.logger.info("Engines created on all workers!")

    async def _async_initialize_engines(self, ft_spec: FinetuneSpec, **kwargs):
        """Initialize engines on all workers in two phases.

        First creates process groups for distributed training, then initializes
        the engines with model loading and optimizer setup.

        Parameters
        ----------
        ft_spec : FinetuneSpec
            Finetune specification containing model and training configuration
        **kwargs
            Additional keyword arguments passed to engine.initialize()
        """
        self.logger.info("Calling engine initialization...")
        # Phase 1: Create process groups for distributed training
        tasks = [
            self.scheduler.async_call_engine(
                worker_id=worker.id,
                method="create_process_group",
                parallel_strategy=self.parallel_strategy,
                _should_bcast=False,
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
                _should_bcast=False,
                **kwargs,
            )
            for worker in self.workers
        ]
        await asyncio.gather(*tasks)
        self.logger.info("All engines are initialized!")

    def _identify_dp_heads(self):
        """Identify which workers are data-parallel heads.

        Queries each worker to determine if it is a DP head. Only DP head workers
        receive data slices directly; non-head workers get data via broadcast from
        their corresponding DP head.

        The result is stored in `self.workers_is_dp_head`, a boolean list where
        `True` indicates the worker at that index is a DP head.
        """
        self.logger.info("Identifying DP head workers...")

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
        self.logger.info("Destroying TrainController...")

        # Delete workers via scheduler
        try:
            self.scheduler.delete_workers(role=self._worker_role)
            self.logger.info("Workers deleted")
        except Exception as e:
            self.logger.error(f"Error deleting workers: {e}")

        # Clear worker lists
        self.workers.clear()
        self.workers_is_dp_head.clear()

        self.logger.info("TrainController destroyed")

    def _custom_function_call(self, method: str, *args, **kwargs):
        """Dispatch method call to appropriate workers based on input type.

        This is the main entry point for calling engine methods across workers.
        It handles:
        1. Splitting DistributedBatch arguments across data-parallel groups
        2. Replicating non-batch arguments to all DP heads
        3. Calling the method on all workers (only DP heads receive data)
        4. Filtering results to only include DP head responses
        5. Merging results from DP heads

        Parameters
        ----------
        method : str
            Name of the engine method to call
        *args
            Positional arguments (DistributedBatch will be split, others replicated)
        **kwargs
            Keyword arguments (DistributedBatch will be split, others replicated)

        Returns
        -------
        Any
            Merged result from DP head workers
        """
        dp_split_args, dp_split_kwargs = self._dispatch_inputs(*args, **kwargs)
        results = self._run_async_task(
            self._call_with_dispatched_inputs(method, dp_split_args, dp_split_kwargs)
        )
        # Filter to only keep results from DP head workers
        results = [r for idx, r in enumerate(results) if self.workers_is_dp_head[idx]]
        return self._merge_results(results, method)

    async def _async_custom_function_call(self, method: str, *args, **kwargs):
        """Async version of _custom_function_call.

        See `_custom_function_call` for detailed documentation.
        This async version is used internally when already in an async context.

        Parameters
        ----------
        method : str
            Name of the engine method to call
        *args
            Positional arguments
        **kwargs
            Keyword arguments

        Returns
        -------
        Any
            Merged result from DP head workers
        """
        dp_split_args, dp_split_kwargs = self._dispatch_inputs(*args, **kwargs)
        results = await self._call_with_dispatched_inputs(
            method, dp_split_args, dp_split_kwargs
        )
        # Filter to only keep results from DP head workers
        results = [r for idx, r in enumerate(results) if self.workers_is_dp_head[idx]]
        return self._merge_results(results, method)

    def _dispatch_inputs(self, *args, **kwargs):
        """Split or replicate inputs across data-parallel groups.

        DistributedBatch arguments are split across DP groups (one slice per DP head).
        Non-batch arguments are replicated to all DP heads.

        Parameters
        ----------
        *args
            Positional arguments to dispatch
        **kwargs
            Keyword arguments to dispatch

        Returns
        -------
        tuple[list[list[Any]], dict[str, list[Any]]]
            A tuple of (split_args, split_kwargs) where each element is a list
            of values, one per DP head worker
        """
        split_args = []
        for arg in args:
            if isinstance(arg, DistributedBatch):
                # Split across DP groups
                split_args.append(self._align_batches_with_dp(arg, rebalance=True))
            else:
                # Replicate to all DP heads
                split_args.append([arg] * self.parallel_strategy.dp_size)

        split_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, DistributedBatch):
                split_kwargs[k] = self._align_batches_with_dp(v, rebalance=True)
            else:
                split_kwargs[k] = [v] * self.parallel_strategy.dp_size
        return split_args, split_kwargs

    async def _call_with_dispatched_inputs(
        self,
        method: str,
        dp_split_args: list[list[Any]],
        dp_worker_kwargs: list[dict[str, Any]],
    ):
        """Call engine method on all workers with dispatched inputs.

        Only DP head workers receive their data slice directly. Non-head workers
        receive empty arguments and will get data via broadcast in the RPC server
        from their corresponding DP head.

        Parameters
        ----------
        method : str
            Name of the engine method to call
        dp_split_args : list[list[Any]]
            Arguments split per DP head (outer list is per-argument, inner list is per-DP-head)
        dp_worker_kwargs : list[dict[str, Any]]
            Keyword arguments split per DP head

        Returns
        -------
        list[Any]
            Results from all workers (will be filtered to DP heads later)

        Note
        ----
        DistributedBatch objects are converted to dicts for RPC serialization.
        TODO: Consider passing only metadata instead of full tensors to reduce
        network overhead for large batches.
        """
        tasks = []
        dp_idx = 0
        for idx, worker in enumerate(self.workers):
            if self.workers_is_dp_head[idx]:
                # Get this DP head worker's slice of each argument
                worker_args = [splits[dp_idx] for splits in dp_split_args]
                worker_kwargs = {
                    k: splits[dp_idx] for k, splits in dp_worker_kwargs.items()
                }

                # Convert DistributedBatch to dict for RPC serialization
                # TODO: Consider passing metadata instead of full tensors to reduce
                # network overhead, especially for large batches
                worker_args = [
                    arg.get_data() if isinstance(arg, DistributedBatch) else arg
                    for arg in worker_args
                ]
                worker_kwargs = {
                    k: v.get_data() if isinstance(v, DistributedBatch) else v
                    for k, v in worker_kwargs.items()
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

    def _merge_results(self, results, method):
        """Merge results from DP head workers based on result type.

        Handles different result types:
        - torch.Tensor: Pads to max sequence length and concatenates along batch dimension
        - dict with attention_mask: Uses DistributedBatchMemory.concat for proper padding
        - dict without attention_mask: Concatenates tensors along batch dimension
        - Other types: Assumes already synchronized, returns first result

        Parameters
        ----------
        results : list[Any]
            Results from DP head workers (all should be of the same type)
        method : str
            Method name (currently unused, kept for potential future use)

        Returns
        -------
        Any
            Merged result from all DP head workers

        Note
        ----
        TODO: Consider implementing a more general data conversion strategy that
        can handle arbitrary nested structures and tensor types.
        """
        first_result = results[0]

        if isinstance(first_result, torch.Tensor):
            # Pad tensors to max sequence length and concatenate along batch dimension
            # Assumes tensor shape is [batch_size, seq_len, ...]
            max_length = max(tensor.shape[1] for tensor in results)
            n_dim = first_result.ndim
            padded_tensors = []
            for tensor in results:
                # Pad format: (pad_left, pad_right) for each dimension from right to left
                # For 2D: (pad_left_seq, pad_right_seq, pad_left_batch, pad_right_batch)
                pad_mode = (
                    (0,) * (2 * (n_dim - 2))
                    + (0, max_length - tensor.shape[1])  # Pad sequence dimension
                    + (0, 0)  # No padding for batch dimension
                )
                padded_tensor = torch.nn.functional.pad(tensor, pad_mode, value=0.0)
                padded_tensors.append(padded_tensor)
            return torch.cat(padded_tensors, dim=0)

        if isinstance(first_result, dict):
            if len(first_result) == 0:
                return DistributedBatchMemory.from_dict({})

            k = next(iter(first_result.keys()))
            if isinstance(first_result[k], torch.Tensor):
                # Check if this looks like a proper batch (has attention_mask)
                # If so, use DistributedBatchMemory.concat which handles padding correctly
                if "attention_mask" in first_result:
                    return DistributedBatchMemory.concat(
                        [DistributedBatchMemory.from_dict(r) for r in results]
                    )
                else:
                    # Simple tensor dict - concatenate tensors along batch dimension
                    merged = {}
                    for key in first_result.keys():
                        if isinstance(first_result[key], torch.Tensor):
                            merged[key] = torch.cat([r[key] for r in results], dim=0)
                        else:
                            # Non-tensor values are assumed to be identical across workers
                            merged[key] = first_result[key]
                    return DistributedBatchMemory.from_dict(merged)

        # For non-tensor, non-dict results, assume they are already synchronized
        # (e.g., scalar statistics that have been all-reduced)
        return first_result

    def _align_batches_with_dp(
        self, input_: DistributedBatch, rebalance=True
    ) -> list[DistributedBatch]:
        """Split DistributedBatch across data-parallel groups.

        Splits the input batch into multiple sub-batches, one for each DP head worker.
        Empty batches are replicated to all DP groups.

        Parameters
        ----------
        input_ : DistributedBatch
            The batch to split across DP groups
        rebalance : bool, optional
            If True, uses chunk_by_ffd for fair distribution based on sequence lengths.
            If False, uses simple chunking. Default is True.

        Returns
        -------
        list[DistributedBatch]
            List of batches, one for each DP head worker

        Note
        ----
        Group normalization (e.g., for reward normalization) should be handled
        in the workflow layer before calling this method, as this method only
        handles data splitting.
        """
        # Handle empty batch by replicating to all DP groups
        if len(input_.get_data()) == 0:
            return [input_] * self.alloc_mode.train.dp_size

        if rebalance:
            # Use fair distribution based on sequence lengths (first-fit-decreasing)
            inputs = input_.chunk_by_ffd(1, self.alloc_mode.train.dp_size)
        else:
            # Simple sequential chunking
            inputs = input_.chunk(self.alloc_mode.train.dp_size)
        return inputs

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

        async def _call_all():
            tasks = [
                self.scheduler.async_call_engine(worker.id, "export_stats")
                for worker in self.workers
            ]
            return await asyncio.gather(*tasks)

        results = self._run_async_task(_call_all())
        # Statistics have been aggregated and synchronized across workers
        # All results should be identical, so return the first one
        return results[0]

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

    # ==================== SFT RPC WRAPPERS ====================
    def train_lm(
        self,
        input_: DistributedBatch,
        *args,
        **kwargs,
    ) -> dict[str, float]:
        """Train language model across workers.

        Parameters
        ----------
        input_ : DistributedBatch
            The distributed input data for language model training
        *args
            Additional positional arguments passed to the engine
        **kwargs
            Additional keyword arguments passed to the engine

        Returns
        -------
        Dict[str, float]
            Scalar statistics after training
        """
        return self._custom_function_call("train_lm", input_, *args, **kwargs)

    def evaluate_lm(
        self,
        input_: DistributedBatch,
        *args,
        **kwargs,
    ) -> torch.Tensor | None:
        """Evaluate language model across workers.

        Parameters
        ----------
        input_ : DistributedBatch
            The distributed input data for language model evaluation
        *args
            Additional positional arguments passed to the engine
        **kwargs
            Additional keyword arguments passed to the engine

        Returns
        -------
        torch.Tensor or None
            A scalar loss or None
        """
        return self._custom_function_call("evaluate_lm", input_, *args, **kwargs)
