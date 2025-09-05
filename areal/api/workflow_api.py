import asyncio
import itertools
import queue
import random
import threading
import time
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import torch.distributed as dist
import uvloop
from megatron.core import parallel_state as mpu
from tensordict import TensorDict
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.cli_args import InferenceEngineConfig
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import RolloutStat
from areal.experimental.openai.types import CompletionWithTokenLogpReward
from areal.utils import logging
from areal.utils.data import concat_padded_tensors

if TYPE_CHECKING:
    from areal.api.engine_api import InferenceEngine

logger = logging.getLogger("areal.workflow_api")


ROLLOUT_POLL_WAIT_TIME = 0.05


class RolloutWorkflow:

    async def arun_episode(
        self, engine: "InferenceEngine", data: Dict[str, Any]
    ) -> Union[TensorDict, None, Dict[str, CompletionWithTokenLogpReward]]:
        """Run a single episode of the workflow.

        `None` implies that this trajectory is rejected and will not be used for training.

        See concrete example implementations under the `areal/workflow` directory.
        """
        raise NotImplementedError()


@dataclass
class _TimedResult:
    t: int
    data: TensorDict


class WorkflowExecutor:

    def __init__(
        self,
        config: InferenceEngineConfig,
        inference_engine: "InferenceEngine",
    ):
        config.max_concurrent_rollouts = (
            config.max_concurrent_rollouts or config.consumer_batch_size
        )
        self.config = config
        self.exiting = threading.Event()
        self.paused = threading.Event()
        self.lock = threading.Lock()

        self.inference_engine = inference_engine

        qsize = config.queue_size or config.max_concurrent_rollouts * 16
        self.input_queue = queue.Queue(maxsize=qsize)
        self.output_queue = queue.Queue(maxsize=qsize)
        self.result_cache: List[_TimedResult] = []

        self.rollout_stat = RolloutStat()

    def initialize(self, train_data_parallel_size: int | None = None):
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

        self.rollout_tasks: Dict[str, asyncio.Task] = {}
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
                1, self.config.max_concurrent_rollouts // self.dp_world_size
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
        task_create_time = {}
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
                    data, workflow = self.input_queue.get_nowait()
                    logger.debug(f"Get data from puller: {data}")
                    task = asyncio.create_task(
                        workflow.arun_episode(self.inference_engine, data),
                        name=str(rid),
                    )
                    rollout_tasks[str(rid)] = task
                    task_create_time[str(rid)] = time.monotonic_ns()
                    self.rollout_stat.submitted += 1
                    self.rollout_stat.running += 1
                    if self.config.enable_rollout_tracing:
                        logger.info(
                            f"Submit rollout rid {rid}. "
                            f"Submit: {self.rollout_stat.submitted}, "
                            f"running: {self.rollout_stat.running}, "
                            f"accepted: {self.rollout_stat.accepted}."
                        )
                    capacity -= 1
                    rid += 1
                tasks = list(rollout_tasks.values())
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
                    if isinstance(traj, dict) and all(
                        isinstance(v, CompletionWithTokenLogpReward)
                        for v in traj.values()
                    ):
                        traj = concat_padded_tensors(
                            [v.to_tensor_dict() for v in traj.values()]
                        )
                    assert traj is None or isinstance(traj, TensorDict), traj
                    task_rid = task.get_name()
                    with self.lock:
                        rollout_tasks.pop(task_rid)
                        create_time = task_create_time.pop(task_rid)
                        self.rollout_stat.accepted += 1
                        self.rollout_stat.running -= 1
                        if self.config.enable_rollout_tracing:
                            logger.info(
                                f"Finish rollout {task_rid}. "
                                f"Submit: {self.rollout_stat.submitted}, "
                                f"running: {self.rollout_stat.running}, "
                                f"accepted: {self.rollout_stat.accepted}."
                            )
                    try:
                        self.output_queue.put_nowait(_TimedResult(create_time, traj))
                    except queue.Full:
                        raise RuntimeError(
                            "Output queue full. Please increase queue_size."
                        )

                await asyncio.sleep(1)
        except Exception:
            traceback.print_exc()
        finally:
            # Cancel remaining tasks
            with self.lock:
                for task in rollout_tasks.values():
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass

    def submit(
        self,
        data: Dict[str, Any],
        workflow: Optional["RolloutWorkflow"] = None,
        workflow_builder: Optional[Callable] = None,
    ) -> None:
        try:
            if workflow is None:
                workflow = workflow_builder()
            self.input_queue.put_nowait((data, workflow))
        except queue.Full:
            raise RuntimeError("Input queue full. Please increase queue_size.")

    def wait(
        self,
        count: int,
        timeout: float | None = None,
        should_accept: Callable | None = None,
    ) -> TensorDict:
        tik = time.perf_counter()
        timeout = timeout or float(7 * 24 * 3600)
        while not self.exiting.is_set() and time.perf_counter() - tik < timeout:
            while True:
                # Drain all outputs.
                try:
                    timed_result = self.output_queue.get_nowait()
                    if timed_result.data is not None and (
                        should_accept is None or should_accept(timed_result.data)
                    ):
                        if self.config.enable_rollout_tracing:
                            logger.info(
                                f"Accept rollout result. accepted/count = {len(self.result_cache)}/{count}"
                            )
                        self.result_cache.append(timed_result)
                    else:
                        if self.config.enable_rollout_tracing:
                            logger.info(f"Rollout is rejected.")
                        with self.lock:
                            self.rollout_stat.accepted -= 1
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
                f"Timed out waiting for {count} rollouts, " f"only received {accepted}."
            )
        if self.config.enable_rollout_tracing:
            logger.info(
                f"Rollout results are ready! accepted/count = {accepted}/{count}"
            )
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
        workflow: Optional["RolloutWorkflow"] = None,
        workflow_builder: Optional[Callable] = None,
    ) -> TensorDict:
        """Submit a batch of requests to the inference engine and wait for the results."""
        for item in data:
            self.submit(item, workflow, workflow_builder)
        return self.wait(count=len(data))

    def prepare_batch(
        self,
        dataloader: StatefulDataLoader,
        workflow: Optional["RolloutWorkflow"] = None,
        workflow_builder: Optional[Callable] = None,
        should_accept: Callable | None = None,
    ):
        if not hasattr(self, "data_generator"):
            self.data_generator = itertools.cycle(dataloader)
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
                    )
            try:
                return self.wait(
                    dataloader.batch_size, timeout=1, should_accept=should_accept
                )
            except TimeoutError:
                pass

    def pause(self):
        self.paused.set()

    def resume(self):
        self.paused.clear()
