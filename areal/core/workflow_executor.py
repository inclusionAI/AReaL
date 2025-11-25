from __future__ import annotations  # noqa

import random
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar, Generic
from collections.abc import Generator
from collections import deque

import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.cli_args import InferenceEngineConfig
from areal.api.workflow_api import RolloutWorkflow
from areal.core.async_task_runner import (
    AsyncTaskRunner,
    TaskQueueFullError,
    TimedResult,
)
from areal.core.staleness_manager import StalenessManager
from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.utils import logging, perf_tracer
from areal.utils.data import concat_padded_tensors, cycle_dataloader
from areal.utils.dynamic_import import import_from_string
from areal.utils.perf_tracer import trace_perf, trace_session_event
from logging import Logger

if TYPE_CHECKING:
    from .remote_inf_engine import RemoteInfEngine


def check_trajectory_format(
    data: dict[str, Any] | None | dict[str, InteractionWithTokenLogpReward],
    batch_size: int | None = None,
    expected_keys: set | None = None,
    logger: Any = None,
) -> bool:
    """Check the format of trajectory data returned by workflow.arun_episode.

    This function validates trajectory data to ensure it conforms to one of three
    expected formats:

    1. **None**: Indicates a rejected trajectory that will not be used for
       training
    2. **Dict[str, InteractionWithTokenLogpReward]**: Completion/Response results from
       the workflow
    3. **Dict[str, torch.Tensor]**: Tensor format with specific shape and
       key requirements

    For tensor format validation, the function ensures:

    - Required keys ``input_ids`` and ``attention_mask`` are present
    - All tensors have consistent batch size and sequence length dimensions
    - Tensor shapes follow the pattern ``[batch_size, max_seqlen]``
    - Keys are consistent across different episodes when ``expected_keys`` is
      provided

    Special handling is provided for:

    - **multi_modal_input**: Expected to be a non-empty list of dictionaries
      containing ``pixel_values``
    - **Non-tensor data**: Logged for informational purposes

    Parameters
    ----------
    data : Dict[str, Any] | None | Dict[str, InteractionWithTokenLogpReward]
        The trajectory data to validate. Can be:

        - ``None`` for rejected trajectories
        - Dictionary mapping strings to ``InteractionWithTokenLogpReward`` objects
        - Dictionary mapping strings to PyTorch tensors or other data types

    batch_size : int | None, optional
        Expected batch size for tensor validation. If ``None``, batch size is inferred
        from the first dimension of ``input_ids``. Default is ``None``.

    expected_keys : set | None, optional
        Set of expected keys for consistency checking across multiple episodes.
        If provided, validates that the current trajectory contains all expected keys.
        Default is ``None``.

    logger : Any, optional
        Logger instance for warning and info messages. If ``None``, creates a default
        logger named "Workflow API". Default is ``None``.

    Returns
    -------
    bool
        ``True`` if the trajectory format is valid, ``False`` otherwise.

    Raises
    ------
    ValueError
        If the trajectory format is invalid. Error messages provide detailed information
        about the specific validation failure, including:

        - Missing required keys
        - Incorrect tensor dimensions
        - Inconsistent batch sizes or sequence lengths
        - Invalid multi-modal input format
        - Key inconsistencies across episodes

    Examples
    --------
    Basic usage with tensor data:

    >>> import torch
    >>> data = {
    ...     'input_ids': torch.randint(0, 1000, (2, 10)),
    ...     'attention_mask': torch.ones(2, 10)
    ... }
    >>> check_trajectory_format(data, batch_size=2)
    True

    Validation with expected keys:

    >>> expected = {'input_ids', 'attention_mask', 'labels'}
    >>> data_with_labels = {
    ...     'input_ids': torch.randint(0, 1000, (2, 10)),
    ...     'attention_mask': torch.ones(2, 10),
    ...     'labels': torch.randint(0, 1000, (2, 10))
    ... }
    >>> check_trajectory_format(data_with_labels, expected_keys=expected)
    True

    Rejected trajectory:

    >>> check_trajectory_format(None)
    True

    See Also
    --------
    RolloutWorkflow.arun_episode : Method that returns trajectory data
    WorkflowExecutor : Class that uses this function when
        ``check_trajectory_format`` is enabled
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    if data is None:
        return True

    if not isinstance(data, dict):
        raise ValueError(f"Expected data to be None or dict, got {type(data)}")

    if len(data) == 0:
        raise ValueError("Data dict cannot be empty")

    # Check if all values are InteractionWithTokenLogpReward
    if all(isinstance(v, InteractionWithTokenLogpReward) for v in data.values()):
        return True

    # Check required keys
    # At least require `input_ids` and `attention_mask`
    required_keys = {"input_ids", "attention_mask"}
    missing_keys = required_keys - set(data.keys())
    if missing_keys:
        raise ValueError(f"Missing required keys in tensor data: {missing_keys}")

    # Check tensor shapes
    input_ids = data["input_ids"]
    if input_ids.dim() != 2:
        raise ValueError(
            f"Expected 2D tensors with shape [batch_size, max_seqlen], "
            f"got {input_ids.dim()}D"
        )

    inferred_batch_size, max_seqlen = input_ids.shape

    if batch_size is not None and inferred_batch_size != batch_size:
        raise ValueError(f"Expected batch size {batch_size}, got {inferred_batch_size}")

    # Check all tensors have consistent shape
    for key, value in data.items():
        if torch.is_tensor(value):
            if value.shape[0] != inferred_batch_size:
                logger.warning(
                    f"The first dim of tensor `{key}` is {value.shape[0]}, "
                    f"rather than the batch size of input_ids ({inferred_batch_size})."
                )
            if value.ndim >= 2 and value.shape[1] != max_seqlen:
                logger.warning(
                    f"The second dim of tensor `{key}` is {value.shape[1]}, "
                    f"rather than the max seqlen of input_ids ({max_seqlen})."
                )
        elif key == "multi_modal_input":
            if (
                not isinstance(value, list)
                or len(value) == 0
                or any(not isinstance(v, dict) for v in value)
            ):
                raise ValueError(
                    "multi_modal_input should be a non-empty list of dicts"
                )
            if not all("pixel_values" in v for v in value):
                raise ValueError(
                    "multi_modal_input should at least contain the "
                    "`pixel_values` field."
                )
        else:
            logger.info(f"Encounter non-tensor data with key `{key}`: {value}")

    # Check key consistency if expected_keys is provided
    if expected_keys is not None:
        missing_keys = expected_keys - set(data.keys())
        if missing_keys:
            raise ValueError(
                f"Inconsistent keys compared to expected: "
                f"expected {expected_keys}, but {missing_keys} are missing."
            )

    return True


@dataclass
class _RolloutTaskInput:
    """Internal wrapper for rollout-specific task input."""

    data: dict[str, Any]
    workflow: RolloutWorkflow
    should_accept_fn: Callable[[dict[str, Any]], bool] | None = None
    task_id: int | None = None


@dataclass
class _RolloutResult:
    trajectory: dict[str, Any]
    task_id: int | None = None


# Batch size for fetching from the async task runner
_MAX_FETCH_BATCH_SIZE = 100
# Timeout for shutting down threads
_SHUTDOWN_TIMEOUT_SECONDS = 2.0


TInput = TypeVar("TInput")
TResult = TypeVar("TResult")


class BatchTaskDispatcher(Generic[TInput, TResult]):
    """Generic dispatcher for asynchronous task execution with staleness control.

    Manages background threads for task submission and result collection.
    Uses producer-consumer pattern with AsyncTaskRunner for async execution.

    Architecture:
    - Producer thread: Submits tasks from _pending_inputs to AsyncTaskRunner
      based on staleness capacity
    - Consumer thread: Collects results from AsyncTaskRunner to _pending_results
    - Main thread: submit_task_input() enqueues, wait_results() polls results
    """

    def __init__(
        self,
        max_queue_size: int,
        task_factory: Callable[[TInput], Callable[[], Awaitable[TResult | None]]],
        staleness_manager: StalenessManager,
        enable_tracing: bool = False,
    ):
        self.runner = AsyncTaskRunner(
            max_queue_size=max_queue_size,
            enable_tracing=enable_tracing,
        )
        self.task_factory = task_factory
        self.staleness_manager = staleness_manager
        self.enable_tracing = enable_tracing
        self.logger: Logger

        # Unbounded deques for producer/consumer pattern
        self._pending_inputs: deque[TInput] = deque()
        self._pending_results: deque[TimedResult[TResult]] = deque()

        # Condition variables for coordination
        self._input_lock = threading.Lock()
        self._input_cv = threading.Condition(self._input_lock)
        self._result_lock = threading.Lock()
        self._result_cv = threading.Condition(self._result_lock)

        # Background thread infrastructure
        self._shutdown_event = threading.Event()
        self._commit_thread: threading.Thread | None = None
        self._fetch_thread: threading.Thread | None = None

        # Exception propagation for fail-fast behavior
        self._thread_exception: Exception | None = None
        self._thread_exception_lock = threading.Lock()

    def _set_thread_exception(self, exc: Exception):
        """Store exception from background thread for fail-fast behavior."""
        with self._thread_exception_lock:
            if self._thread_exception is None:
                self._thread_exception = exc

    def _check_thread_exception(self):
        """Check if any background thread has failed and raise if so."""
        with self._thread_exception_lock:
            if self._thread_exception is not None:
                raise RuntimeError(
                    f"Background thread failed: {self._thread_exception}"
                ) from self._thread_exception

    def _has_runner_capacity(self) -> bool:
        return (
            not self.runner.paused.is_set()
            and self.staleness_manager.get_capacity() > 0
            and self.runner.get_input_queue_size() < self.runner.max_queue_size
        )

    def _commit_loop(self) -> None:
        """Producer thread - continuously submits tasks based on capacity."""
        while not self._shutdown_event.is_set():
            try:
                # Check for errors from other threads (fail-fast)
                self._check_thread_exception()

                task_input = self._get_next_task_for_submission()
                if task_input is None:
                    continue

                task_fn = self.task_factory(task_input)
                try:
                    self.runner.submit(task_fn)
                    self.staleness_manager.on_rollout_submitted()
                    if self.enable_tracing:
                        self.logger.info(f"Submit rollout. {self._rollout_stats()}")
                except TaskQueueFullError:
                    with self._input_cv:
                        self._pending_inputs.appendleft(task_input)
                        self._input_cv.wait_for(
                            lambda: self._shutdown_event.is_set()
                            or self._has_runner_capacity()
                        )
                    # Allow other threads to make progress before retrying
                    continue

            except Exception as e:
                self.logger.error("Producer thread failed", exc_info=True)
                self._set_thread_exception(e)
                break

    def _fetch_loop(self) -> None:
        """Consumer thread - continuously collects results from runner."""
        while not self._shutdown_event.is_set():
            try:
                # Check for errors from other threads (fail-fast)
                self._check_thread_exception()

                # Poll runner for available results (non-blocking)
                output_queue_size = self.runner.get_output_queue_size()
                count = max(1, min(output_queue_size, _MAX_FETCH_BATCH_SIZE))

                try:
                    # Use short timeout for responsiveness (latency-optimized)
                    results = self.runner.wait(
                        count=count, timeout=0.05, with_timing=True
                    )
                except TimeoutError:
                    continue

                with self._result_cv:
                    for result in results:
                        self._pending_results.append(result)
                    self._result_cv.notify_all()

                # Newly available capacity after result processing should wake producers
                with self._input_cv:
                    self._input_cv.notify()

            except Exception as e:
                self.logger.error("Consumer thread failed", exc_info=True)
                self._set_thread_exception(e)
                break

    def _get_next_task_for_submission(self) -> TInput | None:
        with self._input_cv:
            while not self._shutdown_event.is_set():
                # There is capacity and pending inputs
                if (
                    not self.runner.paused.is_set()
                    and self.staleness_manager.get_capacity() > 0
                    and self._pending_inputs
                ):
                    return self._pending_inputs.popleft()
                self._input_cv.wait()

        return None

    def initialize(self, logger: Logger):
        self.logger = logger
        self.runner.initialize(logger=logger)

        self._shutdown_event.clear()

        self._commit_thread = threading.Thread(target=self._commit_loop)
        self._commit_thread.start()

        self._fetch_thread = threading.Thread(target=self._fetch_loop)
        self._fetch_thread.start()

    def destroy(self):
        self._shutdown_event.set()
        with self._input_cv:
            self._input_cv.notify()
        with self._result_cv:
            self._result_cv.notify_all()

        if self._commit_thread and self._commit_thread.is_alive():
            self._commit_thread.join(timeout=_SHUTDOWN_TIMEOUT_SECONDS)
            if self._commit_thread.is_alive() and self.logger:
                self.logger.warning(
                    "Producer thread did not exit cleanly within timeout"
                )

        if self._fetch_thread and self._fetch_thread.is_alive():
            self._fetch_thread.join(timeout=_SHUTDOWN_TIMEOUT_SECONDS)
            if self._fetch_thread.is_alive() and self.logger:
                self.logger.warning(
                    "Consumer thread did not exit cleanly within timeout"
                )

        # Shutdown the async task runner
        self.runner.destroy()

    def pause(self):
        """Pause request submission for async tasks.

        After calling pause(), no new tasks will be started from the
        input queue, but existing running tasks will continue to completion.
        """
        self.runner.pause()
        with self._input_cv:
            self._input_cv.notify()

    def resume(self):
        """Resume request submission for async tasks.

        Allows new tasks to be pulled from the input queue and started.
        """
        self.runner.resume()
        with self._input_cv:
            self._input_cv.notify()

    def is_paused(self) -> bool:
        """Check if the dispatcher is currently paused.

        Returns
        -------
        bool
            True if paused, False otherwise.
        """
        return self.runner.paused.is_set()

    def _rollout_stats(self) -> str:
        stats = self.staleness_manager.get_stats()
        return (
            f"enqueued: {stats.enqueued}, "
            f"running: {stats.running}, "
            f"accepted: {stats.accepted}, "
            f"rejected: {stats.rejected}."
        )

    def submit_task_input(self, task_input: TInput) -> None:
        """Submit a task input for processing.

        Parameters
        ----------
        task_input : TInput
            Task input to be processed.
        """
        self._check_thread_exception()
        with self._input_cv:
            self._pending_inputs.append(task_input)
            self.staleness_manager.on_rollout_enqueued()
            if self.enable_tracing:
                self.logger.info(f"Enqueue rollout. {self._rollout_stats()}")
            self._input_cv.notify()

    def wait_results(
        self, count: int, timeout: float | None = None, raise_timeout: bool = True
    ) -> list[TResult | None]:
        """Wait for the completion of `count` tasks.

        Parameters
        ----------
        count : int
            Number of results to wait for.
        timeout : float | None
            Maximum time to wait in seconds.
        raise_timeout : bool
            Whether to raise TimeoutError on timeout.

        Returns
        -------
        list[TResult | None]
            List of task results, None for rejected tasks.
        """
        if count <= 0:
            raise ValueError(f"count must be positive, got {count}")

        start_time = time.perf_counter()
        if timeout is None:
            timeout = float(7 * 24 * 3600)

        with self._result_cv:
            while len(self._pending_results) < count:
                self._check_thread_exception()

                elapsed = time.perf_counter() - start_time
                remaining = timeout - elapsed
                if remaining <= 0:
                    if raise_timeout:
                        raise TimeoutError(
                            f"Timed out waiting for {count} results, "
                            f"only received {len(self._pending_results)}"
                        )
                    return []

                self._result_cv.wait(timeout=remaining)

            drained: list[TimedResult[TResult]] = list(self._pending_results)
            self._pending_results.clear()

        drained.sort(key=lambda x: x.create_time)
        selected, pending = drained[:count], drained[count:]
        if pending:
            with self._result_cv:
                self._pending_results.extendleft(reversed(pending))
                self._result_cv.notify_all()

        random.shuffle(selected)

        return [r.data for r in selected]

    def active_submit_and_wait(
        self, input_generator: Generator[TInput, None, None], batch_size: int
    ) -> list[TResult]:
        """Continuously submit tasks and wait until a full batch of results is ready.

        This method maintains overlap between submission and result collection
        to maximize throughput.

        Parameters
        ----------
        input_generator : Generator[TInput, None, None]
            An **infinite** generator that yields task inputs. The generator
            must not be exhausted before the batch is complete. Use
            :func:`~areal.utils.data.cycle_dataloader` to wrap finite data sources.
        batch_size : int
            Number of results to collect before returning.

        Returns
        -------
        list[TResult]
            A list of ``batch_size`` task results.

        Raises
        ------
        RuntimeError
            If the input generator is exhausted before the batch is complete.
        """
        cnt = 0
        results = []

        while True:
            # Submit tasks to maintain overlap
            with self._input_cv:
                pending_inputs = len(self._pending_inputs)
            cap_staleness = self.staleness_manager.get_pending_limit() - pending_inputs
            cap_queue = self.runner.max_queue_size - (
                self.runner.get_input_queue_size() + batch_size
            )
            capacity = min(cap_staleness, cap_queue)
            if capacity > 0:
                if self.enable_tracing:
                    perf_tracer.instant(
                        "batch_task_dispatcher.continously_submit",
                        category="scheduler",
                        args={"data": capacity},
                    )
                for _ in range(min(batch_size, capacity)):
                    try:
                        self.submit_task_input(next(input_generator))
                    except StopIteration:
                        raise RuntimeError(
                            "Input generator exhausted before batch completion. "
                            "Use cycle_dataloader() or provide an infinite generator."
                        ) from None
            try:
                res = self.wait_results(count=1, timeout=1)
                if not res or res[0] is None:
                    continue
                assert len(res) == 1
                cnt += 1
                results.append(res[0])
                if cnt >= batch_size:
                    break
            except TimeoutError:
                pass

        return results


class WorkflowExecutor:
    """Executor for asynchronous workflow-based rollout generation.

    Orchestrates workflow execution with AReaL-specific features including
    staleness management, trajectory validation, and result filtering.
    Delegates task dispatching to BatchTaskDispatcher.
    """

    def __init__(
        self,
        config: InferenceEngineConfig,
        inference_engine: RemoteInfEngine,
        staleness_manager: StalenessManager | None = None,
    ):
        self.max_concurrent_rollouts = (
            config.max_concurrent_rollouts or config.consumer_batch_size
        )
        self.consumer_batch_size = config.consumer_batch_size
        self.max_staleness = config.max_head_offpolicyness

        self.config = config
        self.inference_engine = inference_engine

        # Use provided staleness manager or create a default one
        # The manager will be properly initialized in initialize()
        self._staleness_manager = staleness_manager

        # For trajectory format checking
        self._expected_trajectory_keys: set | None = None

        # Dispatcher will be initialized in initialize() after staleness_manager is ready
        self._dispatcher: (
            BatchTaskDispatcher[_RolloutTaskInput, _RolloutResult] | None
        ) = None

    def initialize(self, logger=None, train_data_parallel_size: int | None = None):
        """Initialize the workflow executor and start background threads.

        Creates and initializes BatchTaskDispatcher with StalenessManager.
        The dispatcher starts producer and consumer threads for async execution.

        Parameters
        ----------
        logger : logging.Logger, optional
            Logger instance for debugging and tracing. If None, creates a
            default logger.
        train_data_parallel_size : int | None, optional
            Data parallel world size for capacity scaling. If None, will be inferred
            from distributed state.
        """
        if logger is None:
            dist_ready = dist.is_initialized()
            name = (
                f"WorkflowExecutor Rank {dist.get_rank()}"
                if dist_ready
                else "WorkflowExecutor"
            )
            logger = logging.getLogger(name)
        self.logger = logger

        # Initialize staleness manager if not provided
        if self._staleness_manager is None:
            if train_data_parallel_size is not None:
                dp_world_size = train_data_parallel_size
            else:
                if dist.is_initialized():
                    if not mpu.is_initialized():
                        dp_world_size = dist.get_world_size()
                    else:
                        dp_world_size = mpu.get_data_parallel_world_size()
                else:
                    dp_world_size = 1

            # Apply data parallel scaling
            max_concurrent_rollouts = max(
                1, self.max_concurrent_rollouts // dp_world_size
            )
            consumer_batch_size = max(1, self.consumer_batch_size // dp_world_size)

            self._staleness_manager = StalenessManager(
                version_provider=self.inference_engine,
                max_concurrent_rollouts=max_concurrent_rollouts,
                consumer_batch_size=consumer_batch_size,
                max_staleness=self.config.max_head_offpolicyness,
            )

        # Create and initialize the dispatcher
        qsize = self.config.queue_size or self.max_concurrent_rollouts * 16
        self._dispatcher = BatchTaskDispatcher[_RolloutTaskInput, _RolloutResult](
            max_queue_size=qsize,
            task_factory=self._create_workflow_task,
            staleness_manager=self._staleness_manager,
            enable_tracing=self.config.enable_rollout_tracing,
        )

        # Initialize the dispatcher's async task runner
        self._dispatcher.initialize(logger=logger)

    def destroy(self):
        """Shutdown the workflow executor and clean up resources.

        Destroys the dispatcher (which stops background threads and AsyncTaskRunner),
        then flushes the performance tracer.
        """
        # Stop background threads and shutdown the async task runner
        if self._dispatcher is not None:
            self._dispatcher.destroy()

        # Flush performance tracer
        tracer = perf_tracer.get_session_tracer()
        if tracer is not None:
            tracer.flush(force=True)

    def get_capacity(self):
        """Get current available capacity for new rollouts.

        Returns
        -------
        int
            Number of new rollout slots available based on staleness constraints.
        """
        return self.staleness_manager.get_capacity()

    def _resolve_workflow(
        self,
        workflow: RolloutWorkflow | type[RolloutWorkflow] | str,
        workflow_kwargs: dict[str, Any] | None,
    ) -> RolloutWorkflow:
        """Resolve workflow parameter to a RolloutWorkflow instance.

        Parameters
        ----------
        workflow : RolloutWorkflow | type[RolloutWorkflow] | str
            The workflow specification
        workflow_kwargs : Dict[str, Any] | None
            Keyword arguments for workflow initialization

        Returns
        -------
        RolloutWorkflow
            A workflow instance ready to use

        Raises
        ------
        ValueError
            If workflow_kwargs is required but not provided
        TypeError
            If workflow type is invalid
        """

        # Case 1: Already a workflow instance
        if isinstance(workflow, RolloutWorkflow):
            if workflow_kwargs is not None:
                self.logger.warning(
                    "workflow_kwargs is ignored when workflow is already an instance"
                )
            return workflow

        workflow_class: type[RolloutWorkflow]

        # Resolve to a class type
        if isinstance(workflow, str):
            try:
                imported_obj = import_from_string(workflow)
            except (ValueError, ImportError, AttributeError) as e:
                raise ValueError(
                    f"Failed to import workflow from string {workflow!r}: {e}"
                ) from e

            if not isinstance(imported_obj, type) or not issubclass(
                imported_obj, RolloutWorkflow
            ):
                raise TypeError(
                    f"Imported object from {workflow!r} is not a valid RolloutWorkflow class."
                )
            workflow_class = imported_obj
        elif isinstance(workflow, type) and issubclass(workflow, RolloutWorkflow):
            workflow_class = workflow
        else:
            raise TypeError(
                f"Invalid workflow type: {type(workflow)}. "
                f"Expected RolloutWorkflow instance, RolloutWorkflow class, or string module path."
            )

        # Instantiate the class
        if workflow_kwargs is None:
            raise ValueError(
                f"workflow_kwargs is required when workflow is a class or string. "
                f"Got workflow={workflow}, but workflow_kwargs=None."
            )

        try:
            return workflow_class(**workflow_kwargs)
        except Exception as e:
            raise TypeError(
                f"Failed to instantiate workflow class {workflow_class} "
                f"with kwargs {workflow_kwargs}: {e}"
            ) from e

    def _resolve_should_accept_fn(
        self, should_accept_fn: Callable[[dict[str, Any]], bool] | str | None
    ) -> Callable[[dict[str, Any]], bool] | None:
        """Resolve should_accept_fn parameter to a callable or None.

        Parameters
        ----------
        should_accept_fn : Callable[[Dict[str, Any]], bool] | str | None
            The should_accept_fn specification

        Returns
        -------
        Callable[[Dict[str, Any]], bool] | None
            A callable for trajectory filtering, or None

        Raises
        ------
        ValueError
            If string import fails
        TypeError
            If imported object is not callable
        """
        if should_accept_fn is None or callable(should_accept_fn):
            return should_accept_fn

        if isinstance(should_accept_fn, str):
            try:
                func = import_from_string(should_accept_fn)
            except (ValueError, ImportError, AttributeError) as e:
                raise ValueError(
                    f"Failed to import should_accept_fn from string {should_accept_fn!r}: {e}"
                ) from e
            if not callable(func):
                raise TypeError(
                    f"Imported object {func} from {should_accept_fn!r} is not callable"
                )
            return func

        raise TypeError(
            f"Invalid should_accept_fn type: {type(should_accept_fn)}. "
            f"Expected callable or string module path."
        )

    def _rollout_stats(self) -> str:
        stats = self.staleness_manager.get_stats()
        return (
            f"enqueued: {stats.enqueued}, "
            f"running: {stats.running}, "
            f"accepted: {stats.accepted}, "
            f"rejected: {stats.rejected}."
        )

    def _create_workflow_task(
        self, pending_task: _RolloutTaskInput
    ) -> Callable[[], Awaitable[_RolloutResult | None]]:
        """Wrapper to create an async function that will be executed by AsyncTaskRunner.

        This is a synchronous function that returns an async function, which allows
        us to capture the pending_task context.

        Parameters
        ----------
        pending_task : _RolloutTaskInput
            The rollout task input containing workflow, data, and filter callback.

        Returns
        -------
        Callable
            An async function that executes the workflow and applies
            filtering/validation.
        """

        async def _execute_workflow() -> _RolloutResult | None:
            """Execute workflow.arun_episode and apply AReaL-specific logic."""
            task_id = pending_task.task_id

            # Set task_id in ContextVar before entering arun_episode
            perf_tracer.set_task_id(task_id)

            manager = self.staleness_manager
            traj: dict[str, Any] | None = None
            should_accept_fn = pending_task.should_accept_fn
            should_accept: bool | None = None
            reason: str | None = None

            try:
                traj = await pending_task.workflow.arun_episode(
                    self.inference_engine, pending_task.data
                )

                # Trajectory format checking
                if self.config.check_trajectory_format and traj is not None:
                    check_trajectory_format(
                        traj,
                        expected_keys=self._expected_trajectory_keys,
                        logger=self.logger,
                    )
                    # Track expected keys for consistency checking
                    if isinstance(traj, dict) and "input_ids" in traj:
                        if self._expected_trajectory_keys is None:
                            self._expected_trajectory_keys = set(traj.keys())
                            self.logger.info(
                                "Trajectory format check: tracking keys %s",
                                self._expected_trajectory_keys,
                            )

                # Convert InteractionWithTokenLogpReward to tensor dict if needed
                if isinstance(traj, dict) and all(
                    isinstance(v, InteractionWithTokenLogpReward) for v in traj.values()
                ):
                    traj = concat_padded_tensors(
                        [v.to_tensor_dict() for v in traj.values()]
                    )

                assert traj is None or isinstance(traj, dict), traj

                if traj is None:
                    should_accept_traj = False
                    reason = "returned_none"
                else:
                    if should_accept_fn is None:
                        should_accept = True
                    else:
                        should_accept = bool(should_accept_fn(traj))
                    should_accept_traj = bool(should_accept)
                    if not should_accept_traj and should_accept_fn is not None:
                        reason = "rejected"

                if should_accept_traj:
                    manager.on_rollout_accepted()
                    trace_session_event(
                        "mark_finalized",
                        task_id=task_id,
                        status="accepted",
                    )
                    if self.config.enable_rollout_tracing:
                        self.logger.info(
                            f"Finish and accept rollout. {self._rollout_stats()}",
                        )
                    assert traj is not None
                    return _RolloutResult(task_id=task_id, trajectory=traj)

                manager.on_rollout_rejected()
                trace_session_event(
                    "mark_finalized",
                    task_id=task_id,
                    status="rejected",
                    reason=reason,
                )
                if self.config.enable_rollout_tracing:
                    self.logger.info(
                        f"Finish but reject rollout. {self._rollout_stats()}",
                    )
                return None

            except Exception as exc:  # pragma: no cover - workflow execution errors
                manager.on_rollout_rejected()
                trace_session_event(
                    "mark_finalized",
                    task_id=task_id,
                    status="failed",
                    reason="workflow_exception",
                )
                if self.logger is not None:
                    self.logger.error(
                        "Workflow execution failed: %s", exc, exc_info=True
                    )
                return None

        async def _managed_workflow() -> _RolloutResult | None:
            async with self.inference_engine.managed_session():
                return await _execute_workflow()

        return _managed_workflow

    def submit(
        self,
        data: dict[str, Any],
        workflow: RolloutWorkflow | type[RolloutWorkflow] | str,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: Callable[[dict[str, Any]], bool] | str | None = None,
    ) -> None:
        """Submit a rollout request to the workflow executor.

        Enqueues the request to _pending_inputs. The background producer thread
        will submit it to AsyncTaskRunner when staleness capacity allows. Non-blocking.

        See :meth:`~areal.api.engine_api.InferenceEngine.submit` for parameters.
        """
        # Resolve workflow and should_accept to their concrete forms
        resolved_workflow = self._resolve_workflow(workflow, workflow_kwargs)
        resolved_should_accept_fn = self._resolve_should_accept_fn(should_accept_fn)

        task_id = perf_tracer.register_task()
        task_input = _RolloutTaskInput(
            data=data,
            workflow=resolved_workflow,
            should_accept_fn=resolved_should_accept_fn,
            task_id=task_id,
        )

        # Delegate to dispatcher
        self.dispatcher.submit_task_input(task_input)

    def wait(
        self, count: int, timeout: float | None = None, raise_timeout: bool = True
    ) -> list[dict[str, Any] | None]:
        """Wait for the completion of `count` workflows.

        Returns a list of trajectory dictionaries (or None for rejected rollouts).
        Results are sorted by creation time and shuffled for diversity.

        See :meth:`~areal.api.engine_api.InferenceEngine.wait` for parameters.
        """
        # Delegate to dispatcher and extract trajectories
        results = self.dispatcher.wait_results(count, timeout, raise_timeout)
        # Log and trace
        if self.config.enable_rollout_tracing:
            self.logger.info("Rollout results are ready!")
        return [r.trajectory if r is not None else None for r in results]

    @trace_perf("workflow_executor.rollout_batch", category="scheduler")
    def rollout_batch(
        self,
        data: list[dict[str, Any]],
        workflow: RolloutWorkflow | type[RolloutWorkflow] | str,
        workflow_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Submit a batch of requests and wait for results.

        This method does not support asynchronous rollout and should be used for offline
        data collection or debugging, not in production experiments.

        See :meth:`~areal.api.engine_api.InferenceEngine.rollout_batch` for
        detailed documentation.
        """
        perf_tracer.instant(
            "workflow_executor.rollout_batch",
            category="scheduler",
            args={"data": len(data)},
        )
        for item in data:
            self.submit(
                data=item,
                workflow=workflow,
                workflow_kwargs=workflow_kwargs,
            )
        results = self.wait(count=len(data))
        # Concatenate into batch tensor format
        return concat_padded_tensors([r for r in results if r is not None])

    @trace_perf("workflow_executor.prepare_batch", category="scheduler")
    def prepare_batch(
        self,
        dataloader: StatefulDataLoader,
        workflow: RolloutWorkflow | type[RolloutWorkflow] | str,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: Callable[[dict[str, Any]], bool] | str | None = None,
    ):
        """Prepare a batch with controlled staleness.

        Continuously submits from dataloader and waits for results, ensuring at least
        two batches are pending to maximize overlap.

        .. warning::

            This method caches an internal data generator on the first call.
            The ``dataloader``, ``workflow``, ``workflow_kwargs``, and
            ``should_accept_fn`` parameters are captured at the first invocation
            and reused in all subsequent calls. Passing different arguments in
            later calls will **not** take effect.

            If you need to switch configurations mid-training, consider:

            - Using a separate :class:`WorkflowExecutor` (or engine) instance
            - Using the :meth:`submit` / :meth:`wait` pattern for finer control

        See :meth:`~areal.api.engine_api.InferenceEngine.prepare_batch` for parameters.
        """

        def task_input_generator():
            resolved_should_accept_fn = self._resolve_should_accept_fn(should_accept_fn)
            for data in cycle_dataloader(dataloader):
                for item in data:
                    # Resolve workflow for each submission to allow
                    # resource separation if a type or string is provided.
                    resolved_workflow = self._resolve_workflow(
                        workflow, workflow_kwargs
                    )
                    yield _RolloutTaskInput(
                        data=item,
                        workflow=resolved_workflow,
                        should_accept_fn=resolved_should_accept_fn,
                        task_id=perf_tracer.register_task(),
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
        return concat_padded_tensors([t for t in trajectories if t is not None])

    def pause(self):
        """Pause request submission for async rollout.

        See :meth:`~areal.api.engine_api.InferenceEngine.pause` for detailed
        documentation.
        """
        self.dispatcher.pause()

    def resume(self):
        """Resume request submission for async rollout.

        See :meth:`~areal.api.engine_api.InferenceEngine.resume` for detailed
        documentation.
        """
        self.dispatcher.resume()

    def is_paused(self):
        return self.dispatcher.is_paused()

    @property
    def staleness_manager(self) -> StalenessManager:
        manager = self._staleness_manager
        if manager is None:
            raise RuntimeError(
                "WorkflowExecutor.initialize() must be called before scheduling rollouts."
            )
        return manager

    @property
    def dispatcher(self) -> BatchTaskDispatcher[_RolloutTaskInput, _RolloutResult]:
        """Get the task dispatcher, ensuring initialization has been called."""
        if self._dispatcher is None:
            raise RuntimeError(
                "WorkflowExecutor.initialize() must be called before scheduling rollouts."
            )
        return self._dispatcher

    @property
    def runner(self):
        """For backward compatibility. The runner is now owned by the dispatcher."""
        return self.dispatcher.runner
