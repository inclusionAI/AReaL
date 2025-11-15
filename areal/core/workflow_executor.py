from __future__ import annotations  # noqa

import queue
import random
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from collections import deque

import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.cli_args import InferenceEngineConfig
from areal.api.engine_api import NO_RESULT, NoResult
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

if TYPE_CHECKING:
    from areal.api.engine_api import InferenceEngine


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


# Polling interval for background threads
_POLL_INTERVAL_SECONDS = 0.5
# Batch size for fetching from the async task runner
_MAX_FETCH_BATCH_SIZE = 100
# Timeout for shutting down threads
_SHUTDOWN_TIMEOUT_SECONDS = 2.0


class WorkflowExecutor:
    """Executor for asynchronous workflow-based rollout generation.

    This class orchestrates the execution of rollout workflows with
    AReaL-specific features including staleness management, trajectory
    validation, and result filtering.

    Architecture:
    - Main thread: submit() enqueues to _pending_inputs, wait() polls _pending_results
    - Producer thread (_commit_loop): transfers tasks from _pending_inputs to
      AsyncTaskRunner based on staleness capacity
    - Consumer thread (_fetch_loop): collects results from AsyncTaskRunner and
      places them in _pending_results
    - AsyncTaskRunner: generic async executor running workflows in background event loop

    The executor manages:
    - Integration with InferenceEngine for model generation
    - Staleness-aware capacity control via StalenessManager
    - Trajectory format validation
    - Result filtering via should_accept_fn callbacks
    - InteractionWithTokenLogpReward processing
    - Fail-fast error propagation from background threads

    Parameters
    ----------
    config : InferenceEngineConfig
        Configuration for the inference engine including queue sizes,
        concurrency limits, and validation settings.
    inference_engine : InferenceEngine
        The inference engine to use for generating completions/responses.
    staleness_manager : StalenessManager | None, optional
        Manager for staleness-aware capacity control. If None, a default manager
        will be created during initialization. Default is None.

    See Also
    --------
    AsyncTaskRunner : Generic async task executor used internally
    StalenessManager : Manages capacity based on staleness constraints
    RolloutWorkflow : Interface for rollout episode execution
    """

    def __init__(
        self,
        config: InferenceEngineConfig,
        inference_engine: InferenceEngine,
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

        # Create the generic async task runner
        qsize = config.queue_size or self.max_concurrent_rollouts * 16
        self.runner = AsyncTaskRunner[_RolloutResult | None](
            max_queue_size=qsize,
            enable_tracing=config.enable_rollout_tracing,
        )

        # For trajectory format checking
        self._expected_trajectory_keys: set | None = None

        # Unbounded deques for producer/consumer pattern
        self._pending_inputs: deque[_RolloutTaskInput] = deque()
        self._pending_results: deque[TimedResult[_RolloutResult]] = deque()

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
            if self._thread_exception is None:  # First exception wins
                self._thread_exception = exc

    def _check_thread_exception(self):
        """Check if any background thread has failed and raise if so (fail-fast)."""
        with self._thread_exception_lock:
            if self._thread_exception is not None:
                raise RuntimeError(
                    f"Background thread failed: {self._thread_exception}"
                ) from self._thread_exception

    def _commit_loop(self) -> None:
        """Producer thread main loop - continuously submits tasks to runner based on capacity.

        This method runs in a background thread and continuously:
        1. Checks for errors from other threads (fail-fast)
        2. Waits if runner is paused
        3. Gets capacity from staleness manager based on current model version
        4. Pulls up to 'capacity' tasks from _pending_inputs deque
        5. Submits them to AsyncTaskRunner (handles TaskQueueFullError by re-inserting)
        6. Updates staleness manager metrics on successful submission

        The loop exits when _shutdown_event is set. Polling interval: 0.5s.
        """

        while not self._shutdown_event.is_set():
            try:
                # Check for errors from other threads (fail-fast)
                self._check_thread_exception()

                # Wait for resume if paused
                if self.runner.paused.is_set():
                    time.sleep(_POLL_INTERVAL_SECONDS)
                    continue

                # Get capacity from staleness manager
                version = self.inference_engine.get_version()
                capacity = self.staleness_manager.get_capacity(version)

                if capacity <= 0:
                    time.sleep(_POLL_INTERVAL_SECONDS)
                    continue

                # Try to submit up to 'capacity' tasks
                for _ in range(capacity):
                    try:
                        task = self._pending_inputs.popleft()
                    except IndexError:
                        break

                    # Submit to runner (may raise TaskQueueFullError)
                    workflow_fn = self._create_workflow_task(task)
                    try:
                        self.runner.submit(workflow_fn)

                        self.staleness_manager.on_rollout_submitted()
                        if self.config.enable_rollout_tracing:
                            self.logger.info(f"Submit rollout. {self._rollout_stats()}")
                    except TaskQueueFullError:
                        # Put back and retry later
                        self._pending_inputs.appendleft(task)
                        break

                # Small sleep to avoid busy-waiting (latency-optimized)
                time.sleep(_POLL_INTERVAL_SECONDS)

            except Exception as e:
                self.logger.error("Producer thread failed", exc_info=True)
                self._set_thread_exception(e)
                break

    def _fetch_loop(self) -> None:
        """Consumer thread main loop - continuously collects results from runner.

        This method runs in a background thread and continuously:
        1. Checks for errors from other threads (fail-fast)
        2. Polls AsyncTaskRunner for available results (non-blocking)
        3. Collects results in batches up to 100 with short timeout (0.05s)
        4. Filters out None (rejected) results
        5. Appends accepted TimedResult objects to _pending_results deque

        The loop exits when _shutdown_event is set. Polling interval: 0.5s.
        """
        while not self._shutdown_event.is_set():
            try:
                # Check for errors from other threads (fail-fast)
                self._check_thread_exception()

                # Poll runner for available results (non-blocking)
                output_queue_size = self.runner.get_output_queue_size()

                if output_queue_size == 0:
                    time.sleep(_POLL_INTERVAL_SECONDS)
                    continue

                # Collect all available results at once (batch for efficiency)
                # Limit batch size to avoid blocking too long
                count = min(output_queue_size, _MAX_FETCH_BATCH_SIZE)

                try:
                    # Use short timeout for responsiveness (latency-optimized)
                    results = self.runner.wait(
                        count=count, timeout=0.05, with_timing=True
                    )

                    # Filter out None (rejected rollouts) and enqueue accepted
                    for result in results:
                        if result.data is not None:
                            self._pending_results.append(result)

                except TimeoutError:
                    # No results ready yet
                    pass

                # Small sleep to avoid busy-waiting (latency-optimized)
                time.sleep(_POLL_INTERVAL_SECONDS)

            except Exception as e:
                self.logger.error("Consumer thread failed", exc_info=True)
                self._set_thread_exception(e)
                break

    def initialize(self, logger=None, train_data_parallel_size: int | None = None):
        """Initialize the workflow executor and start background threads.

        Initializes StalenessManager (if needed), AsyncTaskRunner, and starts
        producer (_commit_loop) and consumer (_fetch_loop) threads.

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
                max_concurrent_rollouts=max_concurrent_rollouts,
                consumer_batch_size=consumer_batch_size,
                max_staleness=self.config.max_head_offpolicyness,
            )

        # Initialize the generic async task runner
        self.runner.initialize(logger=logger)

        # Start background threads for producer and consumer
        self._shutdown_event.clear()

        self._commit_thread = threading.Thread(target=self._commit_loop)
        self._commit_thread.start()

        self._fetch_thread = threading.Thread(target=self._fetch_loop)
        self._fetch_thread.start()

    def destroy(self):
        """Shutdown the workflow executor and clean up resources.

        Signals shutdown, waits for threads to exit (5s timeout each),
        flushes perf tracer, and destroys AsyncTaskRunner.
        """
        # Signal shutdown to background threads
        self._shutdown_event.set()

        # Wait for producer thread to finish (with timeout)
        if self._commit_thread and self._commit_thread.is_alive():
            self._commit_thread.join(timeout=_SHUTDOWN_TIMEOUT_SECONDS)
            if self._commit_thread.is_alive():
                self.logger.warning(
                    "Producer thread did not exit cleanly within timeout"
                )

        # Wait for consumer thread to finish (with timeout)
        if self._fetch_thread and self._fetch_thread.is_alive():
            self._fetch_thread.join(timeout=_SHUTDOWN_TIMEOUT_SECONDS)
            if self._fetch_thread.is_alive():
                self.logger.warning(
                    "Consumer thread did not exit cleanly within timeout"
                )

        # Flush performance tracer
        tracer = perf_tracer.get_session_tracer()
        if tracer is not None:
            tracer.flush(force=True)

        # Shutdown the async task runner
        self.runner.destroy()

    def get_capacity(self):
        """Get current available capacity for new rollouts.

        Returns
        -------
        int
            Number of new rollout slots available based on staleness constraints.
        """
        version = self.inference_engine.get_version()
        capacity = self.staleness_manager.get_capacity(version)
        return capacity

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

        return _execute_workflow

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
        # Check for thread errors (fail-fast)
        self._check_thread_exception()

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

        # Enqueue to thread-safe queue (may block if queue is full)
        self._pending_inputs.append(task_input)

        # Notify staleness manager of enqueued rollout tasks
        self.staleness_manager.on_rollout_enqueued()
        if self.config.enable_rollout_tracing:
            self.logger.info(f"Enqueue rollout. {self._rollout_stats()}")

    def wait(
        self, count: int, timeout: float | None = None, raise_timeout: bool = True
    ) -> dict[str, Any] | NoResult:
        """Wait for workflow results from _pending_results deque.

        Polls _pending_results (populated by consumer thread), sorts by create_time,
        shuffles, and returns concatenated batch tensors.

        See :meth:`~areal.api.engine_api.InferenceEngine.wait` for parameters.
        """
        if count <= 0:
            raise ValueError(f"count must be positive, got {count}")

        start_time = time.perf_counter()
        timeout = timeout or float(7 * 24 * 3600)

        while time.perf_counter() - start_time < timeout:
            self._check_thread_exception()

            if len(self._pending_results) >= count:
                break

            elapsed = time.perf_counter() - start_time
            remaining = timeout - elapsed
            time.sleep(min(_POLL_INTERVAL_SECONDS, remaining))

        if len(self._pending_results) < count:
            if raise_timeout:
                raise TimeoutError(
                    f"Timed out waiting for {count} rollouts, "
                    f"only received {len(self._pending_results)}"
                )
            else:
                return NO_RESULT

        # Log and trace
        if self.config.enable_rollout_tracing:
            self.logger.info("Rollout results are ready!")

        # Drain all available requests and sort them by time of creation
        # This prioritizes data submitted earlier.
        results: list[TimedResult[_RolloutResult]] = []
        while True:
            try:
                results.append(self._pending_results.popleft())
            except IndexError:
                break
        # Sort results be create time
        results.sort(key=lambda x: x.create_time)
        results, pending = results[:count], results[count:]
        self._pending_results.extendleft(reversed(pending))

        perf_tracer.instant(
            "workflow_executor.wait",
            category="scheduler",
            args={
                "queue_size": self.runner.get_output_queue_size(),
                "pending_results": len(self._pending_results),
            },
        )

        # Shuffle for randomness (helps with data diversity)
        random.shuffle(results)

        # Concatenate into batch tensor format
        trajectories = [r.data.trajectory for r in results]
        return concat_padded_tensors(trajectories)

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
        return self.wait(count=len(data))

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

        See :meth:`~areal.api.engine_api.InferenceEngine.prepare_batch` for parameters.
        """
        manager = self.staleness_manager
        if not hasattr(self, "data_generator"):
            self.data_generator = cycle_dataloader(dataloader)
        assert dataloader.batch_size is not None
        while True:
            # Submit at least two batches to allow maximum overlap
            if (
                len(self._pending_inputs) < manager.get_pending_limit()
                and self.runner.get_input_queue_size() + dataloader.batch_size
                < self.runner.max_queue_size
            ):
                data = next(self.data_generator)
                perf_tracer.instant(
                    "workflow_executor.prepare_batch",
                    category="scheduler",
                    args={"data": len(data)},
                )
                for item in data:
                    self.submit(
                        item,
                        workflow=workflow,
                        should_accept_fn=should_accept_fn,
                        workflow_kwargs=workflow_kwargs,
                    )
            try:
                return self.wait(dataloader.batch_size, timeout=1)
            except (TimeoutError, queue.Full):
                pass

    def pause(self):
        """Pause request submission for async rollout.

        See :meth:`~areal.api.engine_api.InferenceEngine.pause` for detailed
        documentation.
        """
        self.runner.pause()

    def resume(self):
        """Resume request submission for async rollout.

        See :meth:`~areal.api.engine_api.InferenceEngine.resume` for detailed
        documentation.
        """
        self.runner.resume()

    def is_paused(self):
        return self.runner.paused.is_set()

    @property
    def staleness_manager(self) -> StalenessManager:
        manager = self._staleness_manager
        if manager is None:
            raise RuntimeError(
                "WorkflowExecutor.initialize() must be called before scheduling rollouts."
            )
        return manager
