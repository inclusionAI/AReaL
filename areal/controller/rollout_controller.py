from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock
from typing import Any

from dataclass import asdict
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import InferenceEngineConfig, SchedulingSpec
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import ModelRequest, ModelResponse, ParamSpec, WeightUpdateMeta
from areal.api.scheduler_api import Job, Scheduler, Worker
from areal.api.workflow_api import RolloutWorkflow
from areal.controller.batch import DistributedBatchMemory
from areal.core.staleness_manager import StalenessManager
from areal.core.workflow_executor import BatchTaskDispatcher
from areal.utils import logging, perf_tracer
from areal.utils.data import concat_padded_tensors, cycle_dataloader
from areal.utils.dynamic_import import import_from_string
from areal.utils.perf_tracer import trace_perf

logger = logging.getLogger(__name__)


# NOTE: remote task input has a slightly different
# type annotation, which disallows workflow object or types
@dataclass
class _RemoteRolloutTaskInput:
    data: dict[str, Any]
    workflow: str
    workflow_kwargs: dict[str, Any]
    should_accept_fn: str
    task_id: int | None = None


@dataclass
class _RemoteRolloutResult:
    trajectory: dict[str, Any]
    task_id: int | None = None


class RolloutController:
    def __init__(
        self,
        inf_engine: type[InferenceEngine],
        config: InferenceEngineConfig,
        scheduler: Scheduler,
    ):
        self.inf_engine = inf_engine
        self.config = config
        self.scheduler = scheduler

        # Worker management
        self.workers: list[Worker] = []  # List of Worker objects from scheduler
        self._worker_role: str

        # Round-robin scheduling
        self._current_worker_idx = 0

        # State
        self._version_lock = Lock()
        self._version = 0

        # Use provided staleness manager or create a default one
        # The manager will be properly initialized in initialize()
        self._staleness_manager: StalenessManager | None = None

        # Dispatcher will be initialized in initialize() after staleness_manager is ready
        self._dispatcher: (
            BatchTaskDispatcher[_RemoteRolloutTaskInput, _RemoteRolloutResult] | None
        ) = None

    def initialize(
        self,
        role: str,
        alloc_mode: AllocationMode,
        server_args: dict[str, Any],
        *args,
        **kwargs,
    ):
        # Get scheduling config from kwargs or use defaults
        # Schedule inference engines in the granularity of instance sizes,
        # usually TP x PP.
        self._worker_role = role

        # The first element of `self.config.scheduling_spec` is the resource spec
        # of workers, aka the RPC server process. Since a worker exactly matches
        # to a single engine instance in the local environment, we can dirrectly
        # use the spec of engines  as the spec of workers here. Engine scheduling
        # specs are ignored.
        sch_spec = SchedulingSpec(**asdict(self.config.scheduling_spec[0]))
        sch_spec.cpu *= alloc_mode.gen_instance_size
        sch_spec.mem *= alloc_mode.gen_instance_size
        sch_spec.gpu = alloc_mode.gen_instance_size
        job = Job(
            replicas=alloc_mode.gen.dp_size,
            tasks=[sch_spec for _ in range(alloc_mode.gen.dp_size)],
            scheduling_strategy=self.config.scheduling_strategy,
            role=self._worker_role,
        )

        # Use asyncio.run to call async scheduler methods synchronously
        asyncio.run(
            self._async_initialize(
                job,
                server_args,
                *args,
                **kwargs,
            )
        )

        # Initialize staleness manager for global capacity control
        max_concurrent_rollouts = (
            self.config.max_concurrent_rollouts or self.config.consumer_batch_size
        )
        consumer_batch_size = self.config.consumer_batch_size
        self._staleness_manager = StalenessManager(
            version_provider=self,
            max_concurrent_rollouts=max_concurrent_rollouts,
            consumer_batch_size=consumer_batch_size,
            max_staleness=self.config.max_head_offpolicyness,
        )

        # Create and initialize the dispatcher
        qsize = self.config.queue_size or max_concurrent_rollouts * 16
        self._dispatcher = BatchTaskDispatcher[
            _RemoteRolloutTaskInput, _RemoteRolloutResult
        ](
            max_queue_size=qsize,
            task_factory=self._create_submit_callback,
            staleness_manager=self._staleness_manager,
            enable_tracing=self.config.enable_rollout_tracing,
        )
        # Initialize the dispatcher's async task runner
        self._dispatcher.initialize(logger=logger)

    async def _async_initialize(
        self, job: Job, server_args: dict[str, Any], *args, **kwargs
    ):
        # Create workers via scheduler
        logger.info("Creating workers via scheduler...")
        worker_ids = self.scheduler.create_workers(job=job)
        logger.info(f"Workers created: {worker_ids}")

        # Wait for workers to be ready
        logger.info("Waiting for workers to be ready...")
        self.workers = self.scheduler.get_workers(role=job.role)
        logger.info(f"Workers ready: {[w.id for w in self.workers]}")

        # Get engine class path for dynamic import on workers
        engine_class = self.inf_engine
        engine_path = f"{engine_class.__module__}.{engine_class.__name__}"

        # Create and initialize engines on workers
        logger.info("Creating engines...")
        tasks = [
            self.scheduler.create_engine(
                worker_id=worker.id,
                engine=engine_path,
                config=self.config,
            )
            for worker in self.workers
        ]
        await asyncio.gather(*tasks)
        logger.info("Engine created on all workers!")

        logger.info("Calling engine initialization...")
        await self._collective_rpc_async("launch_server", server_args=server_args)
        await self._collective_rpc_async("initialize", *args, **kwargs)
        logger.info("All engines are initialized...")

    def destroy(self):
        # Stop background threads and shutdown the async task runner
        if self._dispatcher is not None:
            self._dispatcher.destroy()

        self._collective_rpc("destroy", http_timeout=60.0)

        # Delete workers via scheduler
        try:
            self.scheduler.delete_workers(role=self._worker_role)
            logger.info("Workers deleted")
        except Exception as e:
            logger.error(f"Error deleting workers: {e}")

        self.workers.clear()

    def _collective_rpc(self, method: str, *args, **kwargs) -> list[Any]:
        return asyncio.run(self._collective_rpc_async(method, *args, **kwargs))

    async def _collective_rpc_async(self, method: str, *args, **kwargs) -> list[Any]:
        tasks = [
            self.scheduler.async_call_engine(
                worker_id=worker.id,
                method=method,
                *args,
                **kwargs,
            )
            for worker in self.workers
        ]
        return await asyncio.gather(*tasks)

    def _choose_worker(self) -> Worker:
        """Choose a worker for the next request using round-robin scheduling.

        Returns
        -------
        Worker
            The chosen worker object
        """
        if not self.workers:
            raise RuntimeError("No workers available to choose from.")
        worker = self.workers[self._current_worker_idx]
        self._current_worker_idx = (self._current_worker_idx + 1) % len(self.workers)
        return worker

    def _resolve_workflow_str(
        self, workflow: RolloutWorkflow | type[RolloutWorkflow] | str
    ) -> str:
        if isinstance(workflow, str):
            return workflow
        elif isinstance(workflow, type) and issubclass(workflow, RolloutWorkflow):
            return f"{workflow.__module__}.{workflow.__name__}"
        elif isinstance(workflow, RolloutWorkflow):
            return f"{workflow.__module__}.{workflow.__class__.__name__}"
        else:
            raise ValueError(f"Invalid workflow type: {type(workflow)}")

    def _resolve_should_accept_fn(
        self, should_accept_fn: Callable[[dict[str, Any]], bool] | str | None
    ):
        if callable(should_accept_fn):
            raise RuntimeError(
                "If given, `should_accept_fn` must be an importable string path, e.g., 'my_module.filter_func'."
            )
        if should_accept_fn is not None:
            try:
                import_from_string(should_accept_fn)
            except Exception:
                raise RuntimeError(
                    f"Failed to import `should_accept_fn` from string path: {should_accept_fn}"
                )
        return should_accept_fn

    def _rollout_stats(self) -> str:
        stats = self._staleness_manager.get_stats()
        return (
            f"enqueued: {stats.enqueued}, "
            f"running: {stats.running}, "
            f"accepted: {stats.accepted}, "
            f"rejected: {stats.rejected}."
        )

    def _create_submit_callback(self, pending_task: _RemoteRolloutTaskInput):
        async def _submit_then_wait() -> _RemoteRolloutResult | None:
            # Choose worker via round-robin
            worker = self._choose_worker()

            # NOTE: No need to call `on_rollout_submitted` here.
            # This function will be passed to `BatchTaskDispather` where
            # `on_rollout_submitted` will be called upon dispatching
            await self.scheduler.async_call_engine(
                worker.id,
                "submit",
                data=pending_task.data,
                workflow=pending_task.workflow,
                workflow_kwargs=pending_task.workflow_kwargs,
                should_accept_fn=pending_task.should_accept_fn,
                http_timeout=self.config.request_timeout,
            )

            task_id = pending_task.task_id
            manager = self.staleness_manager
            traj: dict[str, Any] | None = None

            try:
                # Wait for a generation to return
                # NOTE: the returned result may not be the one that has
                # been submitted before this callback
                result = []
                tik = time.time()
                while (
                    len(result) == 0 and time.time() - tik < self.config.request_timeout
                ):
                    result = await self.scheduler.async_call_engine(
                        worker.id,
                        "wait",
                        count=1,
                        timeout=0.1,  # A short time to prevent blocking other requests
                        raise_timeout=False,
                        http_timeout=self.config.request_timeout,
                    )

                # TimeourError will be catched below
                if len(result) == 0:
                    raise TimeoutError(
                        f"Rollout request timed out after {self.config.request_timeout}"
                    )

                assert len(result) == 1
                traj = result[0]

                should_accept_traj = traj is not None
                if should_accept_traj:
                    manager.on_rollout_accepted()
                    if self.config.enable_rollout_tracing:
                        logger.info(
                            f"Finish and accept rollout. {self._rollout_stats()}",
                        )
                    assert traj is not None
                    return _RemoteRolloutResult(task_id=task_id, trajectory=traj)

                manager.on_rollout_rejected()
                if self.config.enable_rollout_tracing:
                    logger.info(
                        f"Finish but reject rollout. {self._rollout_stats()}",
                    )
                return None

            except Exception as exc:  # pragma: no cover - workflow execution errors
                manager.on_rollout_rejected()
                logger.error("Workflow execution failed: %s", exc, exc_info=True)
                return None

        return _submit_then_wait

    def get_capacity(self):
        return self.staleness_manager.get_capacity()

    def submit(
        self,
        data: dict[str, Any],
        workflow: RolloutWorkflow | type[RolloutWorkflow] | str,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: str | None = None,
    ) -> None:
        workflow_str = self._resolve_workflow_str(workflow)
        should_accept_fn = self._resolve_should_accept_fn(should_accept_fn)
        if workflow_kwargs is None:
            workflow_kwargs = {}

        # NOTE: RolloutController does not support `should_accept_fn`
        # If the workflow's result should be aborted,
        # `arun_episode` should return None instead.
        task_input = _RemoteRolloutTaskInput(
            data=data,
            workflow=workflow_str,
            workflow_kwargs=workflow_kwargs,
            should_accept_fn=should_accept_fn,
            # NOTE: For now we don't trace tasks at the controller level
            task_id=None,
        )

        # Delegate to dispatcher
        self.dispatcher.submit_task_input(task_input)

    def wait(
        self, count: int, timeout: float | None = None, raise_timeout: bool = True
    ) -> list[dict[str, Any] | None]:
        # Delegate to dispatcher and extract trajectories
        results = self.dispatcher.wait_results(count, timeout, raise_timeout)
        # Log and trace
        if self.config.enable_rollout_tracing:
            logger.info("Rollout results are ready!")
        return [r.trajectory if r is not None else None for r in results]

    @trace_perf("rollout_controller.rollout_batch", category="scheduler")
    def rollout_batch(
        self,
        data: list[dict[str, Any]],
        workflow: RolloutWorkflow | type[RolloutWorkflow] | str,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: str | None = None,
    ) -> dict[str, Any]:
        perf_tracer.instant(
            "rollout_controller.rollout_batch",
            category="scheduler",
            args={"data": len(data)},
        )
        for item in data:
            self.submit(
                data=item,
                workflow=workflow,
                workflow_kwargs=workflow_kwargs,
                should_accept_fn=should_accept_fn,
            )
        results = self.wait(count=len(data))
        # Concatenate into batch tensor format
        batch = concat_padded_tensors([r for r in results if r is not None])

        # NOTE: DistributedBatchMemory.from_dict does nothing for now
        # Just for sync with internal code
        return DistributedBatchMemory.from_dict(batch)

    @trace_perf("rollout_controller.prepare_batch", category="scheduler")
    def prepare_batch(
        self,
        dataloader: StatefulDataLoader,
        workflow: RolloutWorkflow | type[RolloutWorkflow] | str,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: str | None = None,
    ):
        """Prepare a batch with controlled staleness.

        Continuously submits from dataloader and waits for results, ensuring at least
        two batches are pending to maximize overlap.

        See :meth:`~areal.api.engine_api.InferenceEngine.prepare_batch` for parameters.
        """

        workflow_str = self._resolve_workflow_str(workflow)
        if workflow_kwargs is None:
            workflow_kwargs = {}

        def task_input_generator():
            for data in cycle_dataloader(dataloader):
                for item in data:
                    yield _RemoteRolloutTaskInput(
                        data=item,
                        workflow=workflow_str,
                        workflow_kwargs=workflow_kwargs,
                        should_accept_fn=should_accept_fn,
                        task_id=None,
                    )

        if not hasattr(self, "data_generator"):
            self.data_generator = task_input_generator()

        # Delegate to dispatcher
        assert dataloader.batch_size is not None
        results = self.dispatcher.active_submit_and_wait(
            self.data_generator, batch_size=dataloader.batch_size
        )

        # Extract trajectories and concatenate
        trajectories = [r.trajectory if r is not None else None for r in results]
        batch = concat_padded_tensors([t for t in trajectories if t is not None])

        # NOTE: DistributedBatchMemory.from_dict does nothing for now
        # Just for sync with internal code
        return DistributedBatchMemory.from_dict(batch)

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

    async def init_weights_update_group(self, meta: WeightUpdateMeta) -> None:
        await self._collective_rpc_async("init_weights_update_group", meta=meta)

    async def update_weights_from_distributed(
        self, meta: WeightUpdateMeta, param_specs: list[ParamSpec]
    ):
        await self._collective_rpc_async(
            "update_weights_from_distributed", meta=meta, param_specs=param_specs
        )

    async def update_weights_from_disk(self, meta: WeightUpdateMeta):
        await self._collective_rpc_async("update_weights_from_disk", meta=meta)

    def set_version(self, version: int) -> None:
        with self._version_lock:
            self._version = version
            self._collective_rpc("set_version", version=version, http_timeout=60.0)

    def get_version(self) -> int:
        with self._version_lock:
            return self._version

    def pause(self):
        self.dispatcher.pause()
        self._collective_rpc("pause", http_timeout=60.0)

    def resume(self):
        self._collective_rpc("resume", http_timeout=60.0)
        self.dispatcher.resume()

    def export_stats(self) -> dict[str, float]:
        all_raw_stats = self._collective_rpc(method="export_stats", http_timeout=60.0)
        stats = defaultdict(float)
        counts = defaultdict(int)

        for raw_stats in all_raw_stats:
            for k, v in raw_stats.items():
                if k.endswith("__count"):
                    counts[k] += v
                else:
                    stats[k] += v * raw_stats.get(k + "__count", 0)

        # Average non-count stats
        final_stats = {}
        for k, v in stats.items():
            count_key = k + "__count"
            if count_key in counts and counts[count_key] > 0:
                final_stats[k] = v / counts[count_key]
        return final_stats

    @property
    def staleness_manager(self):
        return self._staleness_manager

    @property
    def dispatcher(
        self,
    ) -> BatchTaskDispatcher[_RemoteRolloutTaskInput, _RemoteRolloutResult]:
        """Get the task dispatcher, ensuring initialization has been called."""
        if self._dispatcher is None:
            raise RuntimeError(
                "RolloutController.initialize() must be called before scheduling rollouts."
            )
        return self._dispatcher

    @property
    def runner(self):
        """For backward compatibility. The runner is now owned by the dispatcher."""
        return self.dispatcher.runner
