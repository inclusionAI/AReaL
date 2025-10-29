import asyncio
from collections.abc import Callable
from typing import Any

import torch

from areal.api.alloc_mode import ParallelStrategy
from areal.api.cli_args import TrainEngineConfig
from areal.api.controller_api import DistributedBatch
from areal.api.engine_api import InferenceEngine, TrainEngine
from areal.api.io_struct import (
    AllocationMode,
    FinetuneSpec,
    SaveLoadMeta,
    WeightUpdateMeta,
)
from areal.api.scheduler_api import Job, Scheduler, ScheduleStrategy, Worker
from areal.controller.batch import DistributedBatchMemory
from areal.utils import logging

logger = logging.getLogger("TrainController")


class TrainController:
    """A centralized controller that manages multiple distributed TrainEngine workers.

    TrainController serves as a high-level orchestrator for distributed training across
    multiple concurrent workers, each running TrainEngine instances. It provides a
    unified interface for coordinating training operations while abstracting away the
    complexities of inter-worker communication and data distribution.

    Key differences from TrainEngine:
        - Operates at a higher abstraction level, managing multiple engine instances
        - Does not directly perform collective communications (no rank and process group APIs)
        - Uses `DistributedBatch` for data that spans multiple workers
        - Provides centralized coordination for distributed training workflows

    The controller handles workload distribution, synchronization, and aggregation
    of results from the underlying TrainEngine workers, enabling scalable and
    efficient distributed training.

    Parameters
    ----------
    train_engine : type[TrainEngine]
        The engine class (not instance) to instantiate on each worker
    config : TrainEngineConfig
        Configuration for training engines
    scheduler : Scheduler
        Scheduler for worker management
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

        self.group_size: int
        self.alloc_mode: AllocationMode
        self.workers: list[Worker] = []
        self.dp_head_workers: list[Worker] = []  # Only DP head workers
        self.engine_dp_ranks: list[int] = []  # DP rank of each DP head worker

        self._worker_role: str
        self.logger = None

    def create_process_group(self, parallel_strategy: ParallelStrategy | None = None):
        """Initialize PyTorch distributed communication groups.

        Parameters
        ----------
        parallel_strategy : ParallelStrategy, optional
            The parallel strategy configuration for distributed training, by default None
        """
        assert self.workers is not None, "Workers are not created"
        self.custom_function_call("create_process_group", parallel_strategy)

    def initialize(
        self,
        role: str,
        alloc_mode: AllocationMode,
        ft_spec: FinetuneSpec,
        schedule_strategy: ScheduleStrategy,
        **kwargs,
    ):
        """Initialize environments for distributed training and load models.

        This method should be called after `create_process_group`.

        Parameters
        ----------
        role : str
            Role identifier for the workers
        alloc_mode : AllocationMode
            Allocation mode configuration for distributed setup
        ft_spec : FinetuneSpec
            Finetune specification for model initialization
        schedule_strategy : ScheduleStrategy
            Strategy for scheduling workers
        **kwargs
            Additional keyword arguments passed to engine initialization
        """
        self.logger = logging.getLogger("[TrainController]")

        # Store configuration
        self._worker_role = role
        self.alloc_mode = alloc_mode
        # todo: group size is a sampling parameter and an attribute of the data, should be moved to DistributedBatch
        self.group_size = kwargs.get("group_size", 1)

        # Create job for scheduler
        job = Job(
            replicas=alloc_mode.train.world_size,
            tasks=[
                self.config.scheduling_spec for _ in range(alloc_mode.train.world_size)
            ],
            schedule_strategy=schedule_strategy,
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
        asyncio.run(
            self._async_create_and_initialize_engines(engine_path, ft_spec, **kwargs)
        )

        # Identify DP head workers
        self._identify_dp_heads()

        self.logger.info("TrainController initialization complete")

    async def _async_create_and_initialize_engines(
        self, engine_path: str, ft_spec: FinetuneSpec, **kwargs
    ):
        """Create and initialize engines on all workers."""
        # Create engines on workers
        self.logger.info("Creating engines on workers...")
        tasks = [
            self.scheduler.create_engine(
                worker_id=worker.id,
                engine=engine_path,
                config=self.config,
            )
            for worker in self.workers
        ]
        await asyncio.gather(*tasks)
        self.logger.info("Engines created on all workers!")

        # Initialize engines
        self.logger.info("Calling engine initialization...")
        tasks = [
            self.scheduler.async_call_engine(
                worker_id=worker.id,
                method="initialize",
                addr=None,
                ft_spec=ft_spec,
                **kwargs,
            )
            for worker in self.workers
        ]
        await asyncio.gather(*tasks)
        self.logger.info("All engines are initialized!")

    def _identify_dp_heads(self):
        """Identify which workers are DP heads by querying their DP rank."""
        self.logger.info("Identifying DP head workers...")

        # Query all workers for their DP rank
        async def _get_dp_ranks():
            tasks = [
                self.scheduler.async_call_engine(
                    worker_id=worker.id, method="data_parallel_rank"
                )
                for worker in self.workers
            ]
            return await asyncio.gather(*tasks)

        dp_ranks = asyncio.run(_get_dp_ranks())

        # Find unique DP ranks and corresponding head workers
        seen_dp_ranks = set()
        self.dp_head_workers = []
        self.engine_dp_ranks = []

        for worker, dp_rank in zip(self.workers, dp_ranks):
            if dp_rank not in seen_dp_ranks:
                self.dp_head_workers.append(worker)
                self.engine_dp_ranks.append(dp_rank)
                seen_dp_ranks.add(dp_rank)

        self.logger.info(
            f"Identified {len(self.dp_head_workers)} DP head workers "
            f"from {len(self.workers)} total workers. "
            f"DP ranks: {self.engine_dp_ranks}"
        )

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
        self.dp_head_workers.clear()
        self.engine_dp_ranks.clear()

        self.logger.info("TrainController destroyed")

    def custom_function_call(self, method: str, *args, **kwargs):
        """Dispatch method call to appropriate workers based on input type.

        If any argument is a DistributedBatch, split data and call only DP heads.
        Otherwise, call all workers with the same arguments.
        """
        # Check if any argument is a DistributedBatch
        has_distributed_batch = any(
            isinstance(arg, DistributedBatch) for arg in args
        ) or any(isinstance(v, DistributedBatch) for v in kwargs.values())

        if has_distributed_batch:
            # Call ONLY DP heads with split data
            return self._call_dp_heads_with_data_split(method, *args, **kwargs)
        else:
            # Call ALL workers (no data splitting needed)
            return self._call_all_workers(method, *args, **kwargs)

    def _call_dp_heads_with_data_split(self, method: str, *args, **kwargs):
        """Call only DP head workers with data split across DP groups."""
        # Find and split DistributedBatch arguments
        split_args = []
        for arg in args:
            if isinstance(arg, DistributedBatch):
                # Split across DP groups
                split_args.append(self._align_batches_with_dp(arg, rebalance=True))
            else:
                # Replicate to all DP heads
                split_args.append([arg] * len(self.dp_head_workers))

        split_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, DistributedBatch):
                split_kwargs[k] = self._align_batches_with_dp(v, rebalance=True)
            else:
                split_kwargs[k] = [v] * len(self.dp_head_workers)

        # Call ONLY DP head workers with their data slice
        async def _call_all():
            tasks = []
            for idx, worker in enumerate(self.dp_head_workers):
                # Get this worker's slice of each argument
                worker_args = [splits[idx] for splits in split_args]
                worker_kwargs = {k: splits[idx] for k, splits in split_kwargs.items()}

                # Convert DistributedBatch to dict for RPC
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
                        worker_id=worker.id,
                        method=method,
                        *worker_args,
                        **worker_kwargs,
                    )
                )
            return await asyncio.gather(*tasks)

        results = asyncio.run(_call_all())
        return self._merge_results(results, method)

    def _call_all_workers(self, method: str, *args, **kwargs):
        """Call all workers with the same arguments (no data splitting)."""

        async def _call_all():
            tasks = [
                self.scheduler.async_call_engine(
                    worker_id=worker.id, method=method, *args, **kwargs
                )
                for worker in self.workers
            ]
            return await asyncio.gather(*tasks)

        results = asyncio.run(_call_all())
        return self._merge_results(results, method)

    def _merge_results(self, results, method: str):
        """Merge results from workers based on result type.

        - For None: return None
        - For dict with scalar values: return first (already synchronized)
        - For dict with tensor/batch values: concat as DistributedBatch
        - For tensors/lists: concat as DistributedBatch
        - For scalars: return first (already synchronized)
        """
        # Filter out None results
        non_none_results = [r for r in results if r is not None]

        if len(non_none_results) == 0:
            return None

        first_result = non_none_results[0]

        # If all results are dicts
        if isinstance(first_result, dict):
            # Check if it's a dict of scalars (like train_batch stats)
            if all(isinstance(v, (int, float)) for v in first_result.values()):
                # Stats are already synchronized within engines - return first
                return first_result
            else:
                # Dict of tensors/batches - concat as DistributedBatch
                return DistributedBatchMemory.concat(
                    [DistributedBatchMemory.from_dict(r) for r in non_none_results]
                )

        # If result is a tensor or torch.Tensor
        elif isinstance(first_result, torch.Tensor):
            # Single tensor, likely already reduced - return first
            return first_result

        # If result is a list/iterable (but not string)
        elif hasattr(first_result, "__iter__") and not isinstance(first_result, str):
            try:
                # Try to concat as DistributedBatch
                return DistributedBatchMemory.concat(
                    [
                        DistributedBatchMemory.from_dict(r)
                        if isinstance(r, dict)
                        else r
                        for r in non_none_results
                    ]
                )
            except Exception:
                # If concat fails, return list of results
                return non_none_results

        # For scalars (int, float, bool, etc.)
        else:
            # Return first (already synchronized)
            return first_result

    def _align_batches_with_dp(
        self, input_: DistributedBatch, rebalance=True
    ) -> list[DistributedBatch]:
        """Split DistributedBatch across DP groups.

        Returns a list of batches, one for each DP head worker.
        """
        if rebalance:
            inputs = input_.chunk_by_ffd(self.group_size, self.alloc_mode.train.dp_size)
        else:
            inputs = input_.chunk(self.alloc_mode.train.dp_size)

        # Return batches corresponding to DP head ranks
        batches = []
        for dp_rank in self.engine_dp_ranks:
            batches.append(inputs[dp_rank])

        return batches

    # ==================== ENGINE RPC WRAPPERS ====================
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
        self.custom_function_call("train", mode)
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

    def update_weights(self, meta: WeightUpdateMeta):
        """Update weights to the inference engine in a blocking manner.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Metadata containing information about the weight update
        """
        self.custom_function_call("update_weights", meta)

    def connect_engine(self, engine: InferenceEngine, meta: WeightUpdateMeta):
        """Connect to an inference engine for online training.

        Parameters
        ----------
        engine : InferenceEngine
            The inference engine to connect to
        meta : WeightUpdateMeta
            Metadata for weight update configuration

        Raises
        ------
        NotImplementedError
            This method is not implemented for TrainController
        """
        raise NotImplementedError(
            "connect_engine is not implemented for TrainController. "
            "Use RolloutController for online training workflows."
        )

    def set_version(self, version: int):
        """Set the current weight version in the training engine.

        Parameters
        ----------
        version : int
            The weight version number to set
        """
        self.custom_function_call("set_version", version)

    def get_version(self) -> int:
        """Get the current weight version in the training engine.

        Returns
        -------
        int
            The current weight version number
        """
        return self.custom_function_call("get_version")

    def save(self, meta: SaveLoadMeta):
        """Save model weights and optimizer states for later use.

        Parameters
        ----------
        meta : SaveLoadMeta
            Metadata containing information about where and how to save
        """
        self.custom_function_call("save", meta)

    def load(self, meta: SaveLoadMeta):
        """Load model weights and optimizer states from a file.

        Parameters
        ----------
        meta : SaveLoadMeta
            Metadata containing information about where and how to load
        """
        self.custom_function_call("load", meta)

    def step_lr_scheduler(self):
        """Step the learning rate scheduler.

        Since PPO uses minibatch updates, this method should be called periodically
        (e.g., once per PPO step). It is separated from train_batch to allow
        for more flexible learning rate scheduling.
        """
        self.custom_function_call("step_lr_scheduler")

    def forward(
        self,
        input_: DistributedBatch,
        output_seqlens: list[int] | None = None,
        post_hook: Callable[[torch.Tensor, dict[str, Any]], Any] | None = None,
        aggregate_fn: Callable[[list[Any]], Any] = torch.cat,
    ) -> Any | None:
        """Run the forward pass or inference on the model.

        Note
        ----
        This operation is gradient-free.

        Parameters
        ----------
        input_ : DistributedBatch
            The distributed input data for model forward pass. Redundant entries are allowed.
        output_seqlens : List[int], optional
            The desired output sequence lengths. If None, assumes that the output
            has the same lengths as inputs, by default None.
        post_hook : Callable[[torch.Tensor, Dict[str, Any]], Any], optional
            The post-processing function for micro-batched outputs. Post-processing
            the output on-the-fly during micro-batched forward can reduce peak
            memory usage, by default None.
        aggregate_fn : Callable[[List[Any]], Any], optional
            A function to aggregate micro-batched outputs, by default torch.cat.

        Returns
        -------
        Any or None
            The result produced by `post_hook` and `aggregate_fn`.
        """
        return self.custom_function_call(
            "forward",
            input_=input_,
            output_seqlens=output_seqlens,
            post_hook=post_hook,
            aggregate_fn=aggregate_fn,
        )

    def train_batch(
        self,
        input_: DistributedBatch,
        loss_fn: Callable[[torch.Tensor, dict[str, Any]], torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
    ) -> dict[str, float]:
        """Update the model with a batch of data and a loss function.

        Note
        ----
        The loss_fn should process packed 1D inputs, instead of 2D inputs.

        Parameters
        ----------
        input_ : DistributedBatch
            The distributed input data for model forward pass and the loss function.
            Redundant entries are allowed.
        loss_fn : Callable[[torch.Tensor, Dict[str, Any]], torch.Tensor]
            The loss function that takes the model's forward output and input_,
            and outputs a scalar normalized loss.
        loss_weight_fn : Callable[[Dict[str, Any]], torch.Tensor]
            A function used to calculate the weight of each micro-batch. Since
            loss_fn normalizes the loss for a micro-batch, we need a corresponding
            weight for each micro-batch to normalize the loss globally. The weight
            is usually the number of response tokens in the batch.

        Returns
        -------
        Dict[str, float]
            Scalar statistics after training, e.g., the current learning rate,
            gradient norm, etc.
        """
        return self.custom_function_call("train_batch", input_, loss_fn, loss_weight_fn)

    def eval_batch(
        self,
        input_: DistributedBatch,
        loss_fn: Callable[[torch.Tensor, dict[str, Any]], torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
    ) -> torch.Tensor | None:
        """Evaluate the model using the forward pass and loss function.

        Note
        ----
        The loss_fn should process packed 1D inputs, instead of 2D inputs.

        Parameters
        ----------
        input_ : DistributedBatch
            The distributed input data for model forward pass and the loss function.
            Redundant entries are allowed.
        loss_fn : Callable[[torch.Tensor, Dict[str, Any]], torch.Tensor]
            The loss function that takes the model's forward output and input_,
            and outputs a scalar normalized loss.
        loss_weight_fn : Callable[[Dict[str, Any]], torch.Tensor]
            A function used to calculate the weight of each micro-batch. Since
            loss_fn normalizes the loss for a micro-batch, we need a corresponding
            weight for each micro-batch to normalize the loss globally. The weight
            is usually the number of response tokens in the batch.

        Returns
        -------
        torch.Tensor or None
            A scalar loss or None. The evaluation statistics should be aggregated
            with `stats_tracker`.
        """
        return self.custom_function_call("eval_batch", input_, loss_fn, loss_weight_fn)

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
        return self.custom_function_call("train_lm", input_, *args, **kwargs)

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
        return self.custom_function_call("evaluate_lm", input_, *args, **kwargs)

    # ==================== PPO RPC WRAPPERS ====================
    def compute_logp(
        self,
        *args,
        **kwargs,
    ):
        """Compute log probabilities across workers.

        Parameters
        ----------
        *args
            Positional arguments passed to the engine
        **kwargs
            Keyword arguments passed to the engine

        Returns
        -------
        Any
            Log probabilities computed by the engine
        """
        return self.custom_function_call("compute_logp", *args, **kwargs)

    def compute_advantages(
        self,
        *args,
        **kwargs,
    ):
        """Compute advantages across workers.

        Parameters
        ----------
        *args
            Positional arguments passed to the engine
        **kwargs
            Keyword arguments passed to the engine

        Returns
        -------
        Any
            Advantages computed by the engine
        """
        return self.custom_function_call("compute_advantages", *args, **kwargs)

    def ppo_update(
        self,
        input_: DistributedBatch,
    ) -> dict[str, float]:
        """Perform PPO update step with the given batch.

        Parameters
        ----------
        input_ : DistributedBatch
            The distributed input data containing trajectories for PPO update

        Returns
        -------
        Dict[str, float]
            Scalar statistics after PPO update
        """
        return self.custom_function_call("ppo_update", input_)
