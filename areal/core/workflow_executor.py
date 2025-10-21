from __future__ import annotations  # noqa

import asyncio
import queue
import random
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List

import torch
import torch.distributed as dist
import uvloop
from megatron.core import parallel_state as mpu
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.cli_args import InferenceEngineConfig
from areal.api.workflow_api import RolloutWorkflow
from areal.core.staleness_manager import StalenessManager
from areal.experimental.openai.types import (
    CompletionWithTokenLogpReward,
    ResponseWithTokenLogpReward,
)
from areal.utils import logging
from areal.utils.data import concat_padded_tensors, cycle_dataloader

if TYPE_CHECKING:
    from areal.api.engine_api import InferenceEngine


ROLLOUT_POLL_WAIT_TIME = 0.05
ROLLOUT_POLL_SLEEP_TIME = 1


def check_trajectory_format(
    data: (
        Dict[str, Any]
        | None
        | Dict[str, CompletionWithTokenLogpReward | ResponseWithTokenLogpReward]
    ),
    batch_size: int | None = None,
    expected_keys: set | None = None,
    logger: Any = None,
) -> bool:
    """Check the format of trajectory data returned by workflow.arun_episode.

    This function validates trajectory data to ensure it conforms to one of three expected formats:

    1. **None**: Indicates a rejected trajectory that will not be used for training
    2. **Dict[str, CompletionWithTokenLogpReward | ResponseWithTokenLogpReward]**: Completion/Response results from the workflow
    3. **Dict[str, torch.Tensor]**: Tensor format with specific shape and key requirements

    For tensor format validation, the function ensures:

    - Required keys ``input_ids`` and ``attention_mask`` are present
    - All tensors have consistent batch size and sequence length dimensions
    - Tensor shapes follow the pattern ``[batch_size, max_seqlen]``
    - Keys are consistent across different episodes when ``expected_keys`` is provided

    Special handling is provided for:

    - **multi_modal_input**: Expected to be a non-empty list of dictionaries containing ``pixel_values``
    - **Non-tensor data**: Logged for informational purposes

    Parameters
    ----------
    data : Dict[str, Any] | None | Dict[str, CompletionWithTokenLogpReward | ResponseWithTokenLogpReward]
        The trajectory data to validate. Can be:

        - ``None`` for rejected trajectories
        - Dictionary mapping strings to ``CompletionWithTokenLogpReward`` or ``ResponseWithTokenLogpReward`` objects
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
    RolloutWorkflow.arun_episode : Method that returns trajectory data to be validated
    WorkflowExecutor : Class that uses this function when ``check_trajectory_format`` is enabled
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    if data is None:
        return True

    if not isinstance(data, dict):
        raise ValueError(f"Expected data to be None or dict, got {type(data)}")

    if len(data) == 0:
        raise ValueError("Data dict cannot be empty")

    # Check if all values are CompletionWithTokenLogpReward or ResponseWithTokenLogpReward
    if all(
        isinstance(v, (CompletionWithTokenLogpReward, ResponseWithTokenLogpReward))
        for v in data.values()
    ):
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
            f"Expected 2D tensors with shape [batch_size, max_seqlen], got {input_ids.dim()}D"
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
                    "multi_modal_input should at least contain the `pixel_values` field."
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
class _TimedResult:
    t: int
    data: Dict[str, Any]


@dataclass
class _RolloutTaskInput:
    data: Dict[str, Any]
    workflow: RolloutWorkflow
    should_accept: Callable | None = None


@dataclass
class _RolloutTask:
    create_time: int
    task: asyncio.Task
    task_input: _RolloutTaskInput


class WorkflowExecutor:

    def __init__(
        self,
        config: InferenceEngineConfig,
        inference_engine: "InferenceEngine",
        staleness_manager: StalenessManager | None = None,
    ):
        self.max_concurrent_rollouts = (
            config.max_concurrent_rollouts or config.consumer_batch_size
        )
        self.consumer_batch_size = config.consumer_batch_size

        self.config = config
        self.exiting = threading.Event()
        self.paused = threading.Event()

        self.inference_engine = inference_engine

        # Use provided staleness manager or create a default one
        # The manager will be properly initialized in initialize()
        self.staleness_manager = staleness_manager

        qsize = config.queue_size or self.max_concurrent_rollouts * 16
        self.input_queue = queue.Queue(maxsize=qsize)
        self.output_queue = queue.Queue(maxsize=qsize)
        self.result_cache: List[_TimedResult] = []

        # For trajectory format checking
        self._expected_trajectory_keys: set | None = None

        # For thread exception handling
        self._thread_exception_lock = threading.Lock()
        self._thread_exception: Exception | None = None

    def initialize(self, logger=None, train_data_parallel_size: int | None = None):
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

        self.rollout_thread = threading.Thread(
            target=self._rollout_thread, daemon=True
        )  # set daemon=True to automatically exit when error occurs
        self.rollout_thread.start()

    def destroy(self):
        self.exiting.set()
        self.rollout_thread.join()

    def get_capacity(self):
        version = self.inference_engine.get_version()
        capacity = self.staleness_manager.get_capacity(version)
        return capacity

    def _check_thread_health(self):
        """Check if the rollout thread has encountered a fatal error.

        Raises
        ------
        RuntimeError
            If the rollout thread has died due to an exception.
        """
        with self._thread_exception_lock:
            if self._thread_exception is not None:
                raise RuntimeError(
                    "Rollout thread has died due to an exception. "
                    "No further rollouts can be processed."
                ) from self._thread_exception

    def _rollout_thread(self):
        """Thread that runs the rollout loop."""
        try:
            uvloop.run(self._rollout_thread_async())
        except Exception as e:
            # Store exception with lock for thread-safe access
            with self._thread_exception_lock:
                self._thread_exception = e
            self.logger.error(
                f"Rollout thread failed with exception: {e}", exc_info=True
            )
            # Signal that we're exiting due to error
            self.exiting.set()

    async def _rollout_thread_async(self):
        rollout_tasks: Dict[str, _RolloutTask] = {}
        rid = 0
        try:
            while not self.exiting.is_set():
                # Check capacity
                capacity = self.get_capacity()
                # Create new rollout task
                while (
                    capacity > 0
                    and not self.paused.is_set()
                    and self.input_queue.qsize() > 0
                ):
                    x = self.input_queue.get_nowait()
                    x: _RolloutTaskInput
                    self.logger.debug(f"Get data from puller: {x.data}")
                    task = asyncio.create_task(
                        x.workflow.arun_episode(self.inference_engine, x.data),
                        name=str(rid),
                    )
                    rollout_tasks[str(rid)] = _RolloutTask(
                        create_time=time.monotonic_ns(), task=task, task_input=x
                    )
                    # Notify staleness manager
                    self.staleness_manager.on_rollout_submitted()
                    if self.config.enable_rollout_tracing:
                        stat = self.staleness_manager.get_stats()
                        self.logger.info(
                            f"Submit rollout rid {rid}. "
                            f"Submit: {stat.submitted}, "
                            f"running: {stat.running}, "
                            f"accepted: {stat.accepted}."
                        )
                    capacity -= 1
                    rid += 1
                tasks = [x.task for x in rollout_tasks.values()]

                # Wait for rollout completion
                done = []
                if tasks:
                    done, _ = await asyncio.wait(
                        tasks,
                        timeout=ROLLOUT_POLL_WAIT_TIME,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                # Collect done results
                for task in done:
                    traj = await task

                    # Trajectory format checking, directly raise an error when the format is wrong.
                    if self.config.check_trajectory_format and traj is not None:
                        check_trajectory_format(
                            traj, expected_keys=self._expected_trajectory_keys
                        )
                        # Track expected keys for consistency checking
                        if isinstance(traj, dict) and "input_ids" in traj:
                            if self._expected_trajectory_keys is None:
                                self._expected_trajectory_keys = set(traj.keys())
                                self.logger.info(
                                    f"Trajectory format check: tracking keys {self._expected_trajectory_keys}"
                                )

                    if isinstance(traj, dict) and all(
                        isinstance(
                            v,
                            (
                                CompletionWithTokenLogpReward,
                                ResponseWithTokenLogpReward,
                            ),
                        )
                        for v in traj.values()
                    ):
                        traj = concat_padded_tensors(
                            [v.to_tensor_dict() for v in traj.values()]
                        )
                    assert traj is None or isinstance(traj, dict), traj
                    task_rid = task.get_name()
                    task_obj = rollout_tasks.pop(task_rid)

                    task_input = task_obj.task_input
                    # Check if trajectory should be accepted
                    should_accept_traj = traj is not None and (
                        task_input.should_accept is None
                        or task_input.should_accept(traj)
                    )

                    if should_accept_traj:
                        # Notify staleness manager of accepted rollout
                        self.staleness_manager.on_rollout_accepted()
                        if self.config.enable_rollout_tracing:
                            stat = self.staleness_manager.get_stats()
                            self.logger.info(
                                f"Finish and accept rollout {task_rid}. "
                                f"Submit: {stat.submitted}, "
                                f"running: {stat.running}, "
                                f"accepted: {stat.accepted}."
                            )
                        try:
                            self.output_queue.put_nowait(
                                _TimedResult(task_obj.create_time, traj)
                            )
                        except queue.Full:
                            raise RuntimeError(
                                "Output queue full. Please increase queue_size."
                            )
                    else:
                        # Rollout completed but was rejected
                        # Only decrement running count since it was never accepted
                        self.staleness_manager.on_rollout_rejected()
                        if self.config.enable_rollout_tracing:
                            stat = self.staleness_manager.get_stats()
                            self.logger.info(
                                f"Finish but reject rollout {task_rid}. "
                                f"Submit: {stat.submitted}, "
                                f"running: {stat.running}, "
                                f"accepted: {stat.accepted}."
                            )

                await asyncio.sleep(ROLLOUT_POLL_SLEEP_TIME)
        finally:
            # Cancel remaining tasks
            pending_tasks = [
                task_obj.task
                for task_obj in rollout_tasks.values()
                if not task_obj.task.done()
            ]
            if pending_tasks:
                for task in pending_tasks:
                    task.cancel()
                await asyncio.gather(*pending_tasks, return_exceptions=True)

    def submit(
        self,
        data: Dict[str, Any],
        workflow: "RolloutWorkflow" | None = None,
        workflow_builder: Callable | None = None,
        should_accept: Callable | None = None,
    ) -> None:
        """Submit a request to the workflow executor.

        See :meth:`~areal.api.engine_api.InferenceEngine.submit` for detailed documentation.
        """
        # Check if rollout thread has died before accepting new work
        self._check_thread_health()

        try:
            if workflow is None:
                workflow = workflow_builder()
            x = _RolloutTaskInput(
                data=data, workflow=workflow, should_accept=should_accept
            )
            self.input_queue.put_nowait(x)
        except queue.Full:
            raise RuntimeError("Input queue full. Please increase queue_size.")

    def wait(self, count: int, timeout: float | None = None) -> Dict[str, Any]:
        """Wait for workflow results.

        See :meth:`~areal.api.engine_api.InferenceEngine.wait` for detailed documentation.
        """
        tik = time.perf_counter()
        timeout = timeout or float(7 * 24 * 3600)
        while not self.exiting.is_set() and time.perf_counter() - tik < timeout:
            # Check thread health to detect failures early
            self._check_thread_health()

            while True:
                # Drain all outputs.
                try:
                    timed_result = self.output_queue.get_nowait()
                    self.result_cache.append(timed_result)
                except queue.Empty:
                    break
            if len(self.result_cache) >= count:
                break
            else:
                time.sleep(ROLLOUT_POLL_WAIT_TIME)
        accepted = len(self.result_cache)
        if self.exiting.is_set():
            # Check if exiting due to an exception
            self._check_thread_health()
            raise RuntimeError("Rollout engine is exiting, cannot wait for results.")
        if accepted < count:
            raise TimeoutError(
                f"Timed out waiting for {count} rollouts, only received {accepted}."
            )
        if self.config.enable_rollout_tracing:
            self.logger.info(f"Rollout results are ready!")
        self.result_cache.sort(key=lambda x: x.t)
        results, self.result_cache = (
            self.result_cache[:count],
            self.result_cache[count:],
        )
        random.shuffle(results)
        return concat_padded_tensors([r.data for r in results])

    def rollout_batch(
        self,
        data: List[Dict[str, Any]],
        workflow: "RolloutWorkflow" | None = None,
        workflow_builder: Callable | None = None,
        should_accept: Callable | None = None,
    ) -> Dict[str, Any]:
        """Submit a batch of requests and wait for results.

        See :meth:`~areal.api.engine_api.InferenceEngine.rollout_batch` for detailed documentation.
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
        workflow: "RolloutWorkflow" | None = None,
        workflow_builder: Callable | None = None,
        should_accept: Callable | None = None,
    ):
        """Prepare a batch with controlled staleness.

        See :meth:`~areal.api.engine_api.InferenceEngine.prepare_batch` for detailed documentation.
        """
        if not hasattr(self, "data_generator"):
            self.data_generator = cycle_dataloader(dataloader)
        assert dataloader.batch_size is not None
        while True:
            # Submit at least two batches to allow maximum overlap
            if (
                self.get_capacity() + dataloader.batch_size > 0
                and self.input_queue.qsize() + dataloader.batch_size
                < self.input_queue.maxsize
            ):
                data = next(self.data_generator)
                for item in data:
                    self.submit(
                        item,
                        workflow=workflow,
                        workflow_builder=workflow_builder,
                        should_accept=should_accept,
                    )
            try:
                return self.wait(dataloader.batch_size, timeout=1)
            except TimeoutError:
                pass

    def pause(self):
        """Pause request submission for async rollout.

        See :meth:`~areal.api.engine_api.InferenceEngine.pause` for detailed documentation.
        """
        self.paused.set()

    def resume(self):
        """Resume request submission for async rollout.

        See :meth:`~areal.api.engine_api.InferenceEngine.resume` for detailed documentation.
        """
        self.paused.clear()
