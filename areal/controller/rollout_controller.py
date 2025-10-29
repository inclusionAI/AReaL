"""RolloutController implementation using LocalScheduler and RPC workers."""

from __future__ import annotations

import asyncio
import queue
import random
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import InferenceEngineConfig
from areal.api.controller_api import DistributedBatch
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import ModelRequest, ModelResponse, ParamSpec, WeightUpdateMeta
from areal.api.scheduler_api import Job, Scheduler, ScheduleStrategy, Worker
from areal.controller.batch import DistributedBatchMemory
from areal.core.async_task_runner import AsyncTaskRunner, TaskQueueFullError
from areal.core.staleness_manager import StalenessManager
from areal.utils import logging
from areal.utils.data import cycle_dataloader

CREATE_WORKER_TIMEOUT = 60.0
TASK_RUNNER_MAX_QSIZE = 32768


@dataclass
class _RemoteRolloutTaskInput:
    data: dict[str, Any]
    workflow_path: str
    workflow_kwargs: dict[str, Any]
    should_accept_path: str | None = None


class RolloutController:
    """A centralized controller that manages multiple distributed InferenceEngine workers for rollout generation.

    RolloutController orchestrates distributed inference workloads by scheduling and
    dispatching requests across multiple concurrent InferenceEngine instances. It provides
    intelligent load balancing, staleness control, and capacity management to optimize
    rollout generation efficiency.

    Key features:
        - Distributed request scheduling and load balancing across workers
        - Centralized staleness and capacity control for consistent performance
        - Asynchronous rollout generation with configurable acceptance criteria
        - Data aggregation from heterogeneously loaded workers

    The controller handles workload imbalances inherent in rollout generation, where
    different workers may produce varying amounts of data depending on the complexity
    of their assigned tasks. Generated data is stored locally on workers and aggregated
    into `DistributedBatch` objects for seamless integration with TrainController.

    Implementation details:
        - Launches local inference engines on workers via scheduler
        - Schedules requests to specific engines via round-robin
        - Delegates actual execution to AsyncTaskRunner
        - Aggregates results from workers into DistributedBatch

    Parameters
    ----------
    inf_engine : type[InferenceEngine]
        The inference engine class (not instance) to instantiate on each worker
    config : InferenceEngineConfig
        Configuration for inference engines
    scheduler : Scheduler
        Scheduler for worker management
    """

    def __init__(
        self,
        inf_engine: type[InferenceEngine],
        config: InferenceEngineConfig,
        scheduler: Scheduler,
    ):
        """Initialize the RolloutController.

        Parameters
        ----------
        inf_engine : type[InferenceEngine]
            The inference engine class (not instance) to create on workers
        config : InferenceEngineConfig
            Configuration for the inference engines
        scheduler : Scheduler
            Scheduler for managing workers
        """
        self.inf_engine = inf_engine
        self.config = config
        self.scheduler = scheduler

        # Worker management
        self.workers: list[Worker] = []  # List of Worker objects from scheduler
        self._worker_role: str

        # Round-robin scheduling
        self._current_worker_idx = 0

        # Async task execution
        self.runner: AsyncTaskRunner | None = None

        # Thread pool for weight updates
        self.executor: ThreadPoolExecutor | None = None

        # Logging
        self.logger = None

        # State
        self._version = 0

        # Staleness management
        self.staleness_manager: StalenessManager | None = None

        self._pending_results: list[dict[str, Any]] = []
        self._pending_inputs: list[_RemoteRolloutTaskInput] = []

    def initialize(
        self,
        role: str,
        alloc_mode: AllocationMode,
        schedule_strategy: ScheduleStrategy | None = None,
        *args,
        **kwargs,
    ):
        """Initialize environments and launch the background thread for asynchronous distributed inference.

        For remote inference engines, this serves as a client and connects to the inference servers.
        For local inference engines, this creates an LLM engine on the local GPU.

        Parameters
        ----------
        alloc_mode : AllocationMode
            The allocation mode configuration for distributed setup
        *args
            Variable length argument list passed to engine initialization
        **kwargs
            Arbitrary keyword arguments passed to engine initialization
        """
        self.logger = logging.getLogger("[RolloutController]")

        # Get scheduling config from kwargs or use defaults
        self._worker_role = role
        self.config.scheduling_spec.cpu *= alloc_mode.gen_instance_size
        self.config.scheduling_spec.mem *= alloc_mode.gen_instance_size
        self.config.scheduling_spec.gpu = alloc_mode.gen_instance_size
        job = Job(
            replicas=alloc_mode.gen.dp_size,
            tasks=[self.config.scheduling_spec for _ in range(alloc_mode.gen.dp_size)],
            schedule_strategy=schedule_strategy,
            role=self._worker_role,
        )

        # Use asyncio.run to call async scheduler methods synchronously
        asyncio.run(
            self._async_initialize(
                job,
                *args,
                **kwargs,
            )
        )

        # Initialize AsyncTaskRunner for task execution
        self.runner = AsyncTaskRunner(
            max_queue_size=TASK_RUNNER_MAX_QSIZE,
            enable_tracing=self.config.enable_rollout_tracing,
        )
        self.runner.initialize(logger=self.logger)

        # Initialize thread pool for weight updates
        self.executor = ThreadPoolExecutor(max_workers=alloc_mode.gen.dp_size)

        # Initialize staleness manager for global capacity control
        max_concurrent_rollouts = (
            self.config.max_concurrent_rollouts or self.config.consumer_batch_size
        )
        consumer_batch_size = self.config.consumer_batch_size

        self.staleness_manager = StalenessManager(
            max_concurrent_rollouts=max_concurrent_rollouts,
            consumer_batch_size=consumer_batch_size,
            max_staleness=self.config.max_head_offpolicyness,
        )

    async def _async_initialize(self, job: Job, *args, **kwargs):
        # Create workers via scheduler
        self.logger.info("Creating workers via scheduler...")
        worker_ids = self.scheduler.create_workers(job=job)
        self.logger.info(f"Workers created: {worker_ids}")

        # Wait for workers to be ready
        self.logger.info("Waiting for workers to be ready...")
        self.workers = self.scheduler.get_workers(role=job.role)
        self.logger.info(f"Workers ready: {[w.id for w in self.workers]}")

        # Get engine class path for dynamic import on workers
        engine_class = self.inf_engine
        engine_path = f"{engine_class.__module__}.{engine_class.__name__}"

        # Create and initialize engines on workers
        self.logger.info("Creating engines...")
        tasks = [
            self.scheduler.create_engine(
                worker_id=worker.id,
                engine=engine_path,
                config=self.config,
            )
            for worker in self.workers
        ]
        await asyncio.gather(*tasks)
        self.logger.info("Engine created on all workers!")

        self.logger.info("Calling engine initialization...")
        tasks = [
            self.scheduler.async_call_engine(
                worker_id=worker.id, method="initialize", *args, **kwargs
            )
            for worker in self.workers
        ]
        await asyncio.gather(*tasks)
        self.logger.info("All engines are initialized...")

    def destroy(self):
        """Destroy the engine and release GPU memory for the local inference engine.

        This method cleans up all resources including workers, task runner, and thread pool.
        """
        self.logger.info("Destroying RolloutController...")

        # Destroy task runner
        if self.runner is not None:
            self.runner.destroy()
            self.runner = None

        # Delete workers via scheduler
        try:
            self.scheduler.delete_workers(role=self._worker_role)
            self.logger.info("Workers deleted")
        except Exception as e:
            self.logger.error(f"Error deleting workers: {e}")

        self.workers.clear()

        # Shutdown executor
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None

        self.logger.info("RolloutController destroyed")

    def get_capacity(self) -> int:
        """Get current available capacity for new rollouts based on staleness.

        Returns
        -------
        int
            Number of new rollout slots available based on staleness constraints
        """
        version = self.get_version()  # Use controller's global version
        return self.staleness_manager.get_capacity(version)

    def _choose_worker(self) -> Worker:
        """Choose a worker for the next request using round-robin scheduling.

        Returns
        -------
        Worker
            The chosen worker object
        """

        worker = self.workers[self._current_worker_idx]
        self._current_worker_idx = (self._current_worker_idx + 1) % len(self.workers)
        return worker

    async def _run_workflow_on_worker(
        self,
        worker: Worker,
        data: dict[str, Any],
        workflow_path: str,
        workflow_kwargs: dict[str, Any],
        should_accept_path: str | None = None,
    ) -> dict[str, Any] | None:
        # Call run_workflow on worker via scheduler
        # This will hit the /run_workflow endpoint
        result = await self.scheduler.async_call_engine(
            worker_id=worker.id,
            method="run_workflow",
            workflow=workflow_path,
            workflow_kwargs=workflow_kwargs,
            data=data,
            should_accept_path=should_accept_path,
            check_trajectory_format=self.config.check_trajectory_format,
        )

        # The RPCServer will return None if the
        # trajectory is rejected.
        if result is not None:
            self.staleness_manager.on_rollout_accepted()
            if self.config.enable_rollout_tracing:
                stat = self.staleness_manager.get_stats()
                self.logger.info(
                    f"Finish and accept rollout. "
                    f"Submit: {stat.submitted}, "
                    f"running: {stat.running}, "
                    f"accepted: {stat.accepted}."
                )
            return result
        else:
            self.staleness_manager.on_rollout_rejected()
            if self.config.enable_rollout_tracing:
                stat = self.staleness_manager.get_stats()
                self.logger.info(
                    f"Finish but reject rollout. "
                    f"Submit: {stat.submitted}, "
                    f"running: {stat.running}, "
                    f"accepted: {stat.accepted}."
                )
            return None

    def submit(
        self,
        data: dict[str, Any],
        workflow_path: str,
        workflow_kwargs: dict[str, Any],
        should_accept_path: str | None = None,
    ) -> None:
        """Submit a request to the inference engine and return immediately.

        Should be used together with subsequent `wait`.

        Parameters
        ----------
        data : dict[str, Any]
            The input data for rollout. Used by the user's customized workflow implementation.
        workflow_path : str
            The fully qualified path to the workflow class (e.g., "module.submodule.WorkflowClass").
            The workflow will be dynamically imported on the worker.
        workflow_kwargs : dict[str, Any]
            Keyword arguments to pass to the workflow constructor.
        should_accept_path : str | None, optional
            The fully qualified path to a function used to decide whether to accept a specific
            trajectory (dynamic filtering). The function should take a complete trajectory
            output by the workflow and return a bool, by default None.
        """
        # Add to pending queue (will be submitted when capacity allows)
        self._pending_inputs.append(
            _RemoteRolloutTaskInput(
                data=data,
                workflow_kwargs=workflow_kwargs,
                workflow_path=workflow_path,
                should_accept_path=should_accept_path,
            )
        )

    def _commit_one_to_runner(self):
        """Commit one pending input to task runner with staleness tracking."""
        task_input = self._pending_inputs.pop(0)

        # Choose worker via round-robin
        worker = self._choose_worker()

        # Submit to AsyncTaskRunner
        try:
            self.runner.submit(
                self._run_workflow_on_worker,
                worker,
                task_input.data,
                task_input.workflow_path,
                task_input.workflow_kwargs,
                task_input.should_accept_path,
            )
        except TaskQueueFullError:
            raise queue.Full("Input queue full")

        # Notify staleness manager AFTER successful submission
        self.staleness_manager.on_rollout_submitted()
        if self.config.enable_rollout_tracing:
            stat = self.staleness_manager.get_stats()
            self.logger.info(
                f"Submit rollout. "
                f"Submit: {stat.submitted}, "
                f"running: {stat.running}, "
                f"accepted: {stat.accepted}."
            )

    def wait(self, count: int, timeout: float | None = None) -> DistributedBatch:
        """Wait for a specified number of requests to complete, with a timeout.

        Should be used together with preceding `submit`.

        Parameters
        ----------
        count : int
            The number of accepted trajectories to wait for
        timeout : float | None, optional
            Timeout in seconds. Exceeding the timeout will raise a `TimeoutError`, by default None

        Returns
        -------
        DistributedBatch
            A concatenated batch of trajectories

        Raises
        ------
        TimeoutError
            If the timeout is exceeded before enough trajectories are collected
        """
        #######################################################
        # The following logic is copied from WorkflowExecutor #
        #######################################################
        start_time = time.perf_counter()
        timeout = timeout or float(7 * 24 * 3600)

        # Keep requesting results from runner until we have enough accepted
        # (non-None) results. Use short timeout (1 second) for each wait call
        # to allow periodic checking. This matches original behavior where
        # wait() would poll and allow prepare_batch() to continue
        while True:
            # Submit pending inputs
            # Check capacity before submitting
            capacity = self.get_capacity()
            # Submit pending tasks
            for _ in range(capacity):
                if len(self._pending_inputs) == 0:
                    break
                self._commit_one_to_runner()

            if len(self._pending_results) >= count:
                break

            elapsed = time.perf_counter() - start_time
            remaining_timeout = timeout - elapsed

            if remaining_timeout <= 0:
                raise TimeoutError(
                    f"Timed out waiting for {count} rollouts, only received "
                    f"{len(self._pending_results)}."
                )

            # Try to get at least the number we still need, but request at least 1
            # Note: runner.wait() might return fewer due to rejections (None results)
            needed = max(1, count - len(self._pending_results))

            try:
                # Use short timeout to allow periodic returns (matches original
                # polling behavior)
                batch = self.runner.wait(
                    count=needed, timeout=min(0.1, remaining_timeout)
                )

                # Filter out None results (rejected trajectories)
                # runner.wait() returns List[T] where T can be None for
                # rejected rollouts
                accepted = [result for result in batch if result is not None]
                self._pending_results.extend(accepted)
            except TimeoutError:
                pass

        if self.config.enable_rollout_tracing:
            self.logger.info("Rollout results are ready!")

        # Extract requested number of results
        results = self._pending_results[:count]
        self._pending_results = self._pending_results[count:]

        # Shuffle for randomness (helps with data diversity)
        random.shuffle(results)

        # Convert to DistributedBatch
        if len(results) == 0:
            return DistributedBatchMemory.from_dict({})

        return DistributedBatchMemory.concat(
            [DistributedBatchMemory.from_dict(r) for r in results]
        )

    def rollout_batch(
        self,
        data: list[dict[str, Any]],
        workflow_path: str,
        workflow_kwargs: dict[str, Any],
        should_accept_path: str | None = None,
    ) -> DistributedBatch:
        """Submit a batch of requests to the inference engine and wait for the results.

        Parameters
        ----------
        data : list[dict[str, Any]]
            A list of input data dictionaries for rollout
        workflow_path : str
            The fully qualified path to the workflow class (e.g., "module.submodule.WorkflowClass")
        workflow_kwargs : dict[str, Any]
            Keyword arguments to pass to the workflow constructor
        should_accept_path : str | None, optional
            The fully qualified path to a function to decide whether to accept a trajectory, by default None

        Returns
        -------
        DistributedBatch
            A concatenated batch of trajectory results
        """
        # Submit all requests
        for item in data:
            self.submit(
                item,
                workflow_kwargs=workflow_kwargs,
                workflow_path=workflow_path,
                should_accept_path=should_accept_path,
            )

        # Wait for all results
        return self.wait(count=len(data))

    def prepare_batch(
        self,
        dataloader: StatefulDataLoader,
        workflow_path: str,
        workflow_kwargs: dict[str, Any],
        should_accept_path: str | None = None,
    ) -> DistributedBatch:
        """Asynchronously submit and wait until a full batch is ready with controlled staleness.

        Parameters
        ----------
        dataloader : StatefulDataLoader
            The data loader to pull data from for batch preparation
        workflow_path : str
            The fully qualified path to the workflow class (e.g., "module.submodule.WorkflowClass")
        workflow_kwargs : dict[str, Any]
            Keyword arguments to pass to the workflow constructor
        should_accept_path : str | None, optional
            The fully qualified path to a function to decide whether to accept a trajectory, by default None

        Returns
        -------
        DistributedBatch
            A full batch of trajectory results with controlled staleness
        """
        #######################################################
        # The following logic is copied from WorkflowExecutor #
        #######################################################
        if not hasattr(self, "data_generator"):
            self.data_generator = cycle_dataloader(dataloader)
        assert dataloader.batch_size is not None
        while True:
            # Submit at least two batches to allow maximum overlap
            if (
                self.get_capacity() + dataloader.batch_size > 0
                and self.runner.get_input_queue_size() + dataloader.batch_size
                < self.runner.max_queue_size
            ):
                data = next(self.data_generator)
                for item in data:
                    try:
                        self.submit(
                            item,
                            workflow_kwargs=workflow_kwargs,
                            workflow_path=workflow_path,
                            should_accept_path=should_accept_path,
                        )
                    except queue.Full:
                        # Capacity exhausted during batch submission, stop and wait
                        break
            try:
                return self.wait(dataloader.batch_size, timeout=1)
            except TimeoutError:
                pass

    async def agenerate(self, req: ModelRequest) -> ModelResponse:
        """Asynchronously generate a response for the given request.

        This method provides direct access to the inference engine's generation capabilities
        for single requests, bypassing the workflow system.

        Parameters
        ----------
        req : ModelRequest
            The model request containing input data and generation parameters

        Returns
        -------
        ModelResponse
            The generated response from the model
        """
        # Choose worker and delegate
        worker = self._choose_worker()

        # Call agenerate on engine via scheduler
        return await self.scheduler.async_call_engine(
            worker_id=worker.id,
            method="agenerate",
            req=req,
        )

    def init_weights_update_group(self, meta: WeightUpdateMeta) -> Future[None]:
        """Initialize the weight update process group for distributed weight updates.

        This method should be called before performing any weight updates to ensure
        that the necessary communication groups are set up correctly across all workers.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Metadata containing information about the weight update, such as the
            type of communication backend and allocation mode.

        Returns
        -------
        Future[None]
            A future object representing the asynchronous initialization operation.
        """

        async def _init_all_workers():
            tasks = [
                self.scheduler.async_call_engine(
                    worker_id=worker.id,
                    method="init_weights_update_group",
                    meta=meta,
                )
                for worker in self.workers
            ]
            await asyncio.gather(*tasks)

        def init_all_workers():
            asyncio.run(_init_all_workers())

        return self.executor.submit(init_all_workers)

    def update_weights_from_distributed(
        self, meta: WeightUpdateMeta, param_specs: list[ParamSpec]
    ) -> Future[None]:
        """Update weights in the inference engine in a non-blocking manner from distributed memory.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Metadata containing information about the weight update
        param_specs : list[ParamSpec]
            A list of parameter specifications for the weights to be updated

        Returns
        -------
        Future[None]
            A future object representing the asynchronous weight update operation
        """

        async def _update_all_workers():
            tasks = [
                self.scheduler.call_engine(
                    worker_id=worker.id,
                    method="update_weights_from_distributed",
                    meta=meta,
                    param_specs=param_specs,
                )
                for worker in self.workers
            ]
            await asyncio.gather(*tasks)

        def update_all_workers():
            asyncio.run(_update_all_workers())

        return self.executor.submit(update_all_workers)

    def update_weights_from_disk(self, meta: WeightUpdateMeta) -> Future[None]:
        """Update weights in the inference engine from disk in a non-blocking manner.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Metadata containing information about the weight update

        Returns
        -------
        Future[None]
            A future object representing the asynchronous weight update operation
        """

        async def _update_all_workers():
            tasks = [
                self.scheduler.call_engine(
                    worker_id=worker.id,
                    method="update_weights_from_disk",
                    meta=meta,
                )
                for worker in self.workers
            ]
            await asyncio.gather(*tasks)

        def update_all_workers():
            asyncio.run(_update_all_workers)

        return self.executor.submit(update_all_workers)

    def set_version(self, version: int) -> None:
        """Set the current weight version in the inference engine.

        This updates the version number across all workers, which is used for
        staleness tracking in online training scenarios.

        Parameters
        ----------
        version : int
            The weight version number to set
        """
        self._version = version
        for worker in self.workers:
            try:
                self.scheduler.call_engine(
                    worker_id=worker.id,
                    method="set_version",
                    version=version,
                )
            except Exception as e:
                self.logger.error(f"Error setting version for worker {worker.id}: {e}")

    def get_version(self) -> int:
        """Get the current weight version in the inference engine.

        Returns
        -------
        int
            The current weight version number
        """
        return self._version

    def pause(self):
        """Pause request submission for async rollout.

        Used during evaluation to prevent data over-generation across all workers.
        """
        for worker in self.workers:
            try:
                self.scheduler.call_engine(
                    worker_id=worker.id,
                    method="pause",
                )
            except Exception as e:
                self.logger.error(f"Error pausing worker {worker.id}: {e}")

    def resume(self):
        """Resume request submission for async rollout across all workers."""
        for worker in self.workers:
            try:
                self.scheduler.call_engine(
                    worker_id=worker.id,
                    method="resume",
                )
            except Exception as e:
                self.logger.error(f"Error resuming worker {worker.id}: {e}")

    def register_callback_to_all_worker(
        self, method: str, callback: Callable, **kwargs
    ):
        """Register a callback function for the specified method across all workers.

        Partial rollout API. After successful registration, the controller will poll
        and call the specified method in a background thread. When the return value
        is obtained, it will be used as a parameter to call the `callback` function.

        Parameters
        ----------
        method : str
            The name of the method to register the callback for
        callback : Callable
            The callback function to be called with the method's return value
        **kwargs
            Additional keyword arguments for the callback registration

        Raises
        ------
        NotImplementedError
            This method is not yet implemented
        """
        raise NotImplementedError()

    def abort_all_requests(self) -> None:
        """Abort all ongoing requests in the inference engine.

        Partial rollout API for canceling all queued and in-progress requests across
        all workers.

        Raises
        ------
        NotImplementedError
            This method is not yet implemented
        """
        raise NotImplementedError()
