import asyncio
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime
from typing import Any

import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.alloc_mode import ParallelStrategy
from areal.api.cli_args import TrainEngineConfig
from areal.api.controller_api import DistributedBatch
from areal.api.engine_api import TrainEngine
from areal.api.io_struct import (
    AllocationMode,
    FinetuneSpec,
    SaveLoadMeta,
    WeightUpdateMeta,
)
from areal.api.scheduler_api import Job, Scheduler, Worker
from areal.controller.batch import DistributedBatchMemory
from areal.controller.rollout_controller import RolloutController
from areal.platforms import current_platform
from areal.utils import logging, name_resolve, names
from areal.utils.network import find_free_ports

logger = logging.getLogger("TrainController")


class TrainController:
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
        self.workers_is_dp_head: list[bool] = []  # Only DP head workers
        self.parallel_strategy: ParallelStrategy | None = None

        self.rollout: RolloutController = None
        self.weight_update_group_initialized = False

        self._worker_role: str
        self.logger = None

    def create_process_group(self, parallel_strategy: ParallelStrategy | None = None):
        # A dummy method. Process group will be created during `initialize`
        pass

    def initialize(
        self,
        role: str,
        alloc_mode: AllocationMode,
        ft_spec: FinetuneSpec,
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
        **kwargs
            Additional keyword arguments passed to engine initialization
        """
        self.logger = logging.getLogger("[TrainController]")

        # Store configuration
        self._worker_role = role
        self.alloc_mode = alloc_mode

        if alloc_mode.gen_backend == "sglang":
            self.config.scheduling_spec.env_vars["NCCL_CUMEM_ENABLE"] = "0"
            self.config.scheduling_spec.env_vars["NCCL_NVLS_ENABLE"] = "0"

        self.parallel_strategy = alloc_mode.train

        # Create job for scheduler
        job = Job(
            replicas=alloc_mode.train.world_size,
            tasks=[
                deepcopy(self.config.scheduling_spec)
                for _ in range(alloc_mode.train.world_size)
            ],
            scheduling_strategy=self.config.scheduling_strategy,
            role=self._worker_role,
        )
        # Create environment variables to mimic torchrun
        # FIXME: here master_addr and master_port only work in the local setting
        port = find_free_ports(1)[0]
        for i, task in enumerate(job.tasks):
            task.env_vars["RANK"] = str(i)
            task.env_vars["WORLD_SIZE"] = str(alloc_mode.train.world_size)
            task.env_vars["LOCAL_RANK"] = str(
                0
            )  # because we have only set 1 CUDA_VISIBLE_DEVICES for each process
            # TODO: find a real master addr with scheduler
            task.env_vars["MASTER_ADDR"] = "localhost"
            task.env_vars["MASTER_PORT"] = str(port)

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
        self._run_async_task(self._async_create_engines(engine_path))
        self._run_async_task(self._async_initialize_engines(ft_spec, **kwargs))

        # Identify DP head workers
        self._identify_dp_heads()

        self.logger.info("TrainController initialization complete")

    def _run_async_task(self, task):
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

    async def _async_initialize_engines(self, ft_spec: FinetuneSpec, **kwargs):
        # Initialize engines
        self.logger.info("Calling engine initialization...")
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
        """Identify which workers are DP heads by querying their DP rank."""
        self.logger.info("Identifying DP head workers...")

        # Query all workers for their DP rank
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

        If any argument is a DistributedBatch, split data. Call only DP heads.
        """
        dp_split_args, dp_split_kwargs = self._dispatch_inputs(*args, **kwargs)
        results = self._run_async_task(
            self._call_with_dispatched_inputs(method, dp_split_args, dp_split_kwargs)
        )
        # Only remain data from DP head.
        results = [r for idx, r in enumerate(results) if self.workers_is_dp_head[idx]]
        return self._merge_results(results, method)

    async def _async_custom_function_call(self, method: str, *args, **kwargs):
        dp_split_args, dp_split_kwargs = self._dispatch_inputs(*args, **kwargs)
        results = await self._call_with_dispatched_inputs(
            method, dp_split_args, dp_split_kwargs
        )
        # Only remain data from DP head.
        results = [r for idx, r in enumerate(results) if self.workers_is_dp_head[idx]]
        return self._merge_results(results, method)

    def _dispatch_inputs(self, *args, **kwargs):
        # Find and split DistributedBatch arguments
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
        # Call all workers.
        # ONLY DP head workers get their data slice.
        # Other workers will get data by broadcasting in RPC server.
        tasks = []
        dp_idx = 0
        for idx, worker in enumerate(self.workers):
            if self.workers_is_dp_head[idx]:
                # Get this worker's slice of each argument
                worker_args = [splits[dp_idx] for splits in dp_split_args]
                worker_kwargs = {
                    k: splits[dp_idx] for k, splits in dp_worker_kwargs.items()
                }

                # Convert DistributedBatch to dict for RPC
                # FIXME: pass metadata instead of real tensors
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

        - For torch.Tensor: concat results as DistributedBatch
        - For others: assume they have been synchronized and return the first
        """
        first_result = results[0]

        # FIXME: should use a more general data conversion strategy
        if isinstance(first_result, torch.Tensor):
            # Assume that tensor shapes are [bs, seqlen, *]
            max_length = max(tensor.shape[1] for tensor in results)
            n_dim = first_result.ndim
            padded_tensors = []
            for tensor in results:
                pad_mode = (
                    (0,) * (2 * (n_dim - 2))
                    + (0, max_length - tensor.shape[1])
                    + (0, 0)
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
                # If so, use DistributedBatchMemory.concat which handles padding
                if "attention_mask" in first_result:
                    return DistributedBatchMemory.concat(
                        [DistributedBatchMemory.from_dict(r) for r in results]
                    )
                else:
                    # Simple tensor dict - just concatenate tensors along batch dim
                    merged = {}
                    for key in first_result.keys():
                        if isinstance(first_result[key], torch.Tensor):
                            merged[key] = torch.cat([r[key] for r in results], dim=0)
                        else:
                            merged[key] = first_result[key]
                    return DistributedBatchMemory.from_dict(merged)

        # Return first (already synchronized)
        return first_result

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
            inputs = input_.chunk_by_ffd(1, self.alloc_mode.train.dp_size)
        else:
            inputs = input_.chunk(self.alloc_mode.train.dp_size)
        return inputs

    def connect_engine(self, rollout: RolloutController, meta: WeightUpdateMeta):
        if self.rollout is not None and self.rollout != rollout:
            self.logger.warning(
                f"Connected rollout controller changed from {self.rollout} to {rollout}."
            )
        self.rollout = rollout

        if (
            meta.type == current_platform.communication_backend
            and not self.weight_update_group_initialized
        ):
            self._init_weight_update_from_distributed(meta)
            self.weight_update_group_initialized = True

    def prepare_batch(
        self,
        dataloader: StatefulDataLoader,
        workflow_path: str,
        workflow_kwargs: dict[str, Any],
        should_accept_path: str | None = None,
    ) -> DistributedBatch:
        return self.rollout.prepare_batch(
            dataloader=dataloader,
            workflow_path=workflow_path,
            workflow_kwargs=workflow_kwargs,
            should_accept_path=should_accept_path,
        )

    def rollout_batch(
        self,
        data: list[dict[str, Any]],
        workflow_path: str,
        workflow_kwargs: dict[str, Any],
        should_accept_path: str | None = None,
    ) -> DistributedBatch:
        return self.rollout.rollout_batch(
            data=data,
            workflow_path=workflow_path,
            workflow_kwargs=workflow_kwargs,
            should_accept_path=should_accept_path,
        )

    def _init_weight_update_from_distributed(self, meta: WeightUpdateMeta):
        raise NotImplementedError()

    def _update_weights_from_distributed(self, meta: WeightUpdateMeta):
        raise NotImplementedError()

    def _update_weights_from_disk(self, meta: WeightUpdateMeta):
        # Update all LocalInfEngine's local weight
        save_meta = SaveLoadMeta(
            path=meta.path,
            weight_format="hf",
            with_optim=False,
            tokenizer=None,
            processor=None,
        )

        async def _actor_save():
            await self._async_custom_function_call("save", save_meta)
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

        async def _run():
            rollout_load = self.rollout.update_weights_from_disk(meta)
            actor_save = _actor_save()
            await asyncio.gather(rollout_load, actor_save)

        self._run_async_task(_run())

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

    def export_stats(self):
        async def _call_all():
            tasks = [
                self.scheduler.async_call_engine(worker.id, "export_stats")
                for worker in self.workers
            ]
            return await asyncio.gather(*tasks)

        results = self._run_async_task(_call_all())
        # stats have been aggregated and synchronized.
        return results[0]

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
        return self._custom_function_call(
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
        return self._custom_function_call(
            "train_batch", input_, loss_fn, loss_weight_fn
        )

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
        return self._custom_function_call("eval_batch", input_, loss_fn, loss_weight_fn)

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
        return self._custom_function_call("compute_logp", *args, **kwargs)

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
        return self._custom_function_call("compute_advantages", *args, **kwargs)

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
        return self._custom_function_call("ppo_update", input_)
