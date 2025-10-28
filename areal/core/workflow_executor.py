from __future__ import annotations  # noqa

import queue
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.cli_args import InferenceEngineConfig
from areal.api.workflow_api import RolloutWorkflow
from areal.core.async_task_runner import AsyncTaskRunner, TaskQueueFullError
from areal.core.staleness_manager import StalenessManager
from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.utils import logging
from areal.utils.data import concat_padded_tensors, cycle_dataloader

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
    should_accept: Callable | None = None


class WorkflowExecutor:
    """Executor for asynchronous workflow-based rollout generation.

    This class orchestrates the execution of rollout workflows with
    AReaL-specific features including staleness management, trajectory
    validation, and result filtering. It uses a generic AsyncTaskRunner
    internally for task execution while adding domain-specific logic for
    RL training.

    The executor manages:
    - Integration with InferenceEngine for model generation
    - Staleness-aware capacity control via StalenessManager
    - Trajectory format validation
    - Result filtering via should_accept callbacks
    - InteractionWithTokenLogpReward processing

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

        self.config = config
        self.inference_engine = inference_engine

        # Use provided staleness manager or create a default one
        # The manager will be properly initialized in initialize()
        self.staleness_manager = staleness_manager

        # Create the generic async task runner
        qsize = config.queue_size or self.max_concurrent_rollouts * 16
        self.runner = AsyncTaskRunner[dict[str, Any] | None](
            max_queue_size=qsize,
            enable_tracing=config.enable_rollout_tracing,
        )

        # For trajectory format checking
        self._expected_trajectory_keys: set | None = None

        # Cache for tracking inputs and accepted/rejected results
        self._pending_results: list[dict[str, Any]] = []
        self._pending_inputs: list[_RolloutTaskInput] = []

    def initialize(self, logger=None, train_data_parallel_size: int | None = None):
        """Initialize the workflow executor and start the async task runner.

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
            logger = logging.getLogger(__name__)
        self.logger = logger

        # Initialize staleness manager if not provided
        if self.staleness_manager is None:
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

            self.staleness_manager = StalenessManager(
                max_concurrent_rollouts=max_concurrent_rollouts,
                consumer_batch_size=consumer_batch_size,
                max_staleness=self.config.max_head_offpolicyness,
            )

        # Initialize the generic async task runner
        self.runner.initialize(logger=logger)

    def destroy(self):
        """Shutdown the workflow executor and clean up resources."""
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

    def _create_workflow_task(
        self, task_input: _RolloutTaskInput
    ) -> Callable[[], Awaitable[dict[str, Any] | None]]:
        """Wrapper to create an async function that will be executed by AsyncTaskRunner.

        This is a synchronous function that returns an async function, which allows
        us to capture the task_input context.

        Parameters
        ----------
        task_input : _RolloutTaskInput
            The rollout task input containing workflow, data, and filter callback.

        Returns
        -------
        Callable
            An async function that executes the workflow and applies
            filtering/validation.
        """

        async def _execute_workflow():
            """Execute workflow.arun_episode and apply AReaL-specific logic."""
            # Execute the workflow
            traj = await task_input.workflow.arun_episode(
                self.inference_engine, task_input.data
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
                            f"Trajectory format check: tracking keys "
                            f"{self._expected_trajectory_keys}"
                        )

            # Convert InteractionWithTokenLogpReward to tensor dict if needed
            if isinstance(traj, dict) and all(
                isinstance(v, InteractionWithTokenLogpReward) for v in traj.values()
            ):
                traj = concat_padded_tensors(
                    [v.to_tensor_dict() for v in traj.values()]
                )

            assert traj is None or isinstance(traj, dict), traj

            # Apply should_accept filtering
            should_accept_traj = traj is not None and (
                task_input.should_accept is None or task_input.should_accept(traj)
            )

            # Notify staleness manager
            if should_accept_traj:
                self.staleness_manager.on_rollout_accepted()
                if self.config.enable_rollout_tracing:
                    stat = self.staleness_manager.get_stats()
                    self.logger.info(
                        f"Finish and accept rollout. "
                        f"Submit: {stat.submitted}, "
                        f"running: {stat.running}, "
                        f"accepted: {stat.accepted}."
                    )
                return traj
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

        return _execute_workflow

    def submit(
        self,
        data: dict[str, Any],
        workflow: RolloutWorkflow | None = None,
        workflow_builder: Callable | None = None,
        should_accept: Callable | None = None,
    ) -> None:
        """Submit a request to the workflow executor.

        See :meth:`~areal.api.engine_api.InferenceEngine.submit` for detailed
        documentation.
        """
        # Create workflow if builder provided
        if workflow is None:
            workflow = workflow_builder()
        self._pending_inputs.append(
            _RolloutTaskInput(
                data=data,
                workflow=workflow,
                should_accept=should_accept,
            )
        )

    def _commit_one_to_runner(self):
        task_input = self._pending_inputs.pop(0)
        # Create the async workflow execution function and submit to runner
        workflow_fn = self._create_workflow_task(task_input)
        try:
            self.runner.submit(workflow_fn)
        except TaskQueueFullError:
            # Convert RuntimeError from AsyncTaskRunner to queue.Full for
            # backward compatibility
            raise queue.Full("Input queue full. Please increase queue_size.")

        # Notify staleness manager of submission only after successful submission
        self.staleness_manager.on_rollout_submitted()
        if self.config.enable_rollout_tracing:
            stat = self.staleness_manager.get_stats()
            self.logger.info(
                f"Submit rollout. "
                f"Submit: {stat.submitted}, "
                f"running: {stat.running}, "
                f"accepted: {stat.accepted}."
            )

    def wait(self, count: int, timeout: float | None = None) -> dict[str, Any]:
        """Wait for workflow results.

        See :meth:`~areal.api.engine_api.InferenceEngine.wait` for detailed
        documentation.
        """
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

        # Concatenate into batch tensor format
        return concat_padded_tensors(results)

    def rollout_batch(
        self,
        data: list[dict[str, Any]],
        workflow: RolloutWorkflow | None = None,
        workflow_builder: Callable | None = None,
        should_accept: Callable | None = None,
    ) -> dict[str, Any]:
        """Submit a batch of requests and wait for results.

        See :meth:`~areal.api.engine_api.InferenceEngine.rollout_batch` for
        detailed documentation.
        """
        for item in data:
            self.submit(
                data=item,
                workflow=workflow,
                workflow_builder=workflow_builder,
                should_accept=should_accept,
            )
        return self.wait(count=len(data))

    def prepare_batch(
        self,
        dataloader: StatefulDataLoader,
        workflow: RolloutWorkflow | None = None,
        workflow_builder: Callable | None = None,
        should_accept: Callable | None = None,
    ):
        """Prepare a batch with controlled staleness.

        See :meth:`~areal.api.engine_api.InferenceEngine.prepare_batch` for
        detailed documentation.
        """
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
                            workflow=workflow,
                            workflow_builder=workflow_builder,
                            should_accept=should_accept,
                        )
                    except queue.Full:
                        # Capacity exhausted during batch submission, stop and wait
                        break
            try:
                return self.wait(dataloader.batch_size, timeout=1)
            except TimeoutError:
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
