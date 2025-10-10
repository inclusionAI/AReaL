from __future__ import annotations  # noqa

import asyncio
import queue
import random
import threading
import time
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List

import torch
import torch.distributed as dist
import uvloop
from megatron.core import parallel_state as mpu
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.cli_args import InferenceEngineConfig
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import RolloutStat
from areal.experimental.openai.types import CompletionWithTokenLogpReward
from areal.utils import logging
from areal.utils.data import concat_padded_tensors, cycle_dataloader

if TYPE_CHECKING:
    from areal.api.engine_api import InferenceEngine


ROLLOUT_POLL_WAIT_TIME = 0.05


class RolloutWorkflow:

    async def arun_episode(
        self, engine: "InferenceEngine", data: Dict[str, Any]
    ) -> Dict[str, Any] | None | Dict[str, CompletionWithTokenLogpReward]:
        """Run a single episode of the workflow.

        Note
        ----
        Returning `None` implies that this trajectory is rejected and will not be used for training.

        See concrete example implementations under the `areal/workflow` directory.

        Parameters
        ----------
        engine : InferenceEngine
            The inference engine to use for generating responses
        data : Dict[str, Any]
            Input data for the workflow episode

        Returns
        -------
        Dict[str, Any] | None | Dict[str, CompletionWithTokenLogpReward]
            The trajectory result, None if rejected, or a dictionary of completion results
        """
        raise NotImplementedError()


def check_trajectory_format(
    data: Dict[str, Any] | None | Dict[str, CompletionWithTokenLogpReward],
    batch_size: int | None = None,
    expected_keys: set | None = None,
    logger: Any = None,
) -> bool:
    """Check the format of trajectory data returned by workflow.arun_episode.

    This function validates trajectory data to ensure it conforms to one of three expected formats:

    1. **None**: Indicates a rejected trajectory that will not be used for training
    2. **Dict[str, CompletionWithTokenLogpReward]**: Completion results from the workflow
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
    data : Dict[str, Any] | None | Dict[str, CompletionWithTokenLogpReward]
        The trajectory data to validate. Can be:

        - ``None`` for rejected trajectories
        - Dictionary mapping strings to ``CompletionWithTokenLogpReward`` objects
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
        logger = logging.getLogger("Workflow API")
    if data is None:
        return True

    if not isinstance(data, dict):
        raise ValueError(f"Expected data to be None or dict, got {type(data)}")

    if len(data) == 0:
        raise ValueError("Data dict cannot be empty")

    # Check if all values are CompletionWithTokenLogpReward
    if all(isinstance(v, CompletionWithTokenLogpReward) for v in data.values()):
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
    ):
        self.max_concurrent_rollouts = (
            config.max_concurrent_rollouts or config.consumer_batch_size
        )
        self.config = config
        self.exiting = threading.Event()
        self.paused = threading.Event()
        self.lock = threading.Lock()

        self.inference_engine = inference_engine

        qsize = config.queue_size or self.max_concurrent_rollouts * 16
        self.input_queue = queue.Queue(maxsize=qsize)
        self.output_queue = queue.Queue(maxsize=qsize)
        self.result_cache: List[_TimedResult] = []

        self.rollout_stat = RolloutStat()

        # For trajectory format checking
        self._expected_trajectory_keys: set | None = None

    def initialize(self, logger=None, train_data_parallel_size: int | None = None):
        if logger is None:
            logger = logging.getLogger("WorkflowExecutor")
        self.logger = logger

        if train_data_parallel_size is not None:
            self.dp_world_size = train_data_parallel_size
        else:
            if dist.is_initialized():
                if not mpu.is_initialized():
                    self.dp_world_size = dist.get_world_size()
                else:
                    self.dp_world_size = mpu.get_data_parallel_world_size()
            else:
                self.dp_world_size = 1

        self.rollout_tasks: Dict[str, _RolloutTask] = {}
        self.rollout_thread = threading.Thread(
            target=self._rollout_thread, daemon=True
        )  # set daemon=True to automatically exit when error occurs
        self.rollout_thread.start()

    def destroy(self):
        self.exiting.set()
        self.rollout_thread.join()

    def get_capacity(self):
        with self.lock:
            max_concurrent_rollouts = max(
                1, self.max_concurrent_rollouts // self.dp_world_size
            )
            capacity = max_concurrent_rollouts - len(self.rollout_tasks)
            # Staleness control
            version = self.inference_engine.get_version()
            ofp = self.config.max_head_offpolicyness
            sample_cnt = self.rollout_stat.accepted + self.rollout_stat.running
            consumer_bs = max(1, self.config.consumer_batch_size // self.dp_world_size)
            capacity = min(capacity, (ofp + version + 1) * consumer_bs - sample_cnt)
        return capacity

    def _rollout_thread(self):
        """Thread that runs the rollout loop."""
        try:
            uvloop.run(self._rollout_thread_async())
        except Exception:
            traceback.print_exc()

    async def _rollout_thread_async(self):
        rollout_tasks = self.rollout_tasks
        rid = 0
        try:
            while not self.exiting.is_set():
                # Check capacity
                capacity = self.get_capacity()
                # Create new rollout task
                self.lock.acquire()
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
                    self.rollout_stat.submitted += 1
                    self.rollout_stat.running += 1
                    if self.config.enable_rollout_tracing:
                        self.logger.info(
                            f"Submit rollout rid {rid}. "
                            f"Submit: {self.rollout_stat.submitted}, "
                            f"running: {self.rollout_stat.running}, "
                            f"accepted: {self.rollout_stat.accepted}."
                        )
                    capacity -= 1
                    rid += 1
                tasks = [x.task for x in rollout_tasks.values()]
                self.lock.release()

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
                        isinstance(v, CompletionWithTokenLogpReward)
                        for v in traj.values()
                    ):
                        traj = concat_padded_tensors(
                            [v.to_tensor_dict() for v in traj.values()]
                        )
                    assert traj is None or isinstance(traj, dict), traj
                    task_rid = task.get_name()
                    with self.lock:
                        task_obj = rollout_tasks.pop(task_rid)
                        self.rollout_stat.accepted += 1
                        self.rollout_stat.running -= 1
                        if self.config.enable_rollout_tracing:
                            self.logger.info(
                                f"Finish rollout {task_rid}. "
                                f"Submit: {self.rollout_stat.submitted}, "
                                f"running: {self.rollout_stat.running}, "
                                f"accepted: {self.rollout_stat.accepted}."
                            )

                    task_input = task_obj.task_input
                    if traj is not None and (
                        task_input.should_accept is None
                        or task_input.should_accept(traj)
                    ):
                        if self.config.enable_rollout_tracing:
                            self.logger.info(
                                f"Accept rollout result of task {task_rid}."
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
                        if self.config.enable_rollout_tracing:
                            self.logger.info(f"Rollout is rejected.")
                        with self.lock:
                            self.rollout_stat.accepted -= 1

                await asyncio.sleep(1)
        except Exception:
            traceback.print_exc()
        finally:
            # Cancel remaining tasks
            with self.lock:
                for task_obj in rollout_tasks.values():
                    if not task_obj.task.done():
                        task_obj.task.cancel()
                        try:
                            await task_obj.task
                        except asyncio.CancelledError:
                            pass

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
        dataloader: StatefulDataLoader | List[Dict[str, Any]],
        workflow: "RolloutWorkflow" | None = None,
        workflow_builder: Callable | None = None,
        should_accept: Callable | None = None,
    ):
        """Prepare a batch with controlled staleness.

        See :meth:`~areal.api.engine_api.InferenceEngine.prepare_batch` for detailed documentation.
        """
        if isinstance(dataloader, StatefulDataLoader):
            # 处理StatefulDataLoader类型 - 保持原有逻辑不变
            if not hasattr(self, "data_generator"):
                self.data_generator = cycle_dataloader(dataloader)
            assert dataloader.batch_size is not None
            batch_size = dataloader.batch_size
            
            while True:
                # Submit at least two batches to allow maximum overlap
                if (
                    self.get_capacity() + batch_size > 0
                    and self.input_queue.qsize() + batch_size
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
                    return self.wait(batch_size, timeout=1)
                except TimeoutError:
                    pass
        else:
            self.data_list_index = 0
            
            # 对于List类型，使用固定的batch_size=1
            batch_size = 1
            
            while True:
                # Submit at least two batches to allow maximum overlap
                if (
                    self.get_capacity() + batch_size > 0
                    and self.input_queue.qsize() + batch_size
                    < self.input_queue.maxsize
                ):
                    # 从List中获取数据，支持循环访问
                    if self.data_list_index >= len(dataloader):
                        self.data_list_index = 0  # 循环访问
                    
                    item = dataloader[self.data_list_index]
                    self.data_list_index += 1
                    
                    self.submit(
                        item,
                        workflow=workflow,
                        workflow_builder=workflow_builder,
                        should_accept=should_accept,
                    )
                try:
                    return self.wait(batch_size, timeout=1)
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
