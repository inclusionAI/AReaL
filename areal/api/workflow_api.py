import asyncio
import itertools
import os
import queue
import random
import threading
import time
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import requests
import torch.distributed as dist
import uvicorn
import uvloop
from fastapi import FastAPI
from tensordict import TensorDict
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.cli_args import InferenceEngineConfig
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import RolloutStat
from areal.experimental.openai.types import CompletionWithTokenLogpReward
from areal.utils import logging, network
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


def create_capacity_app(staleness_manager: "StalenessManager") -> FastAPI:
    """Create a centralized FastAPI app for staleness management.

    Each experiment has a single instance of this app.
    """
    app = FastAPI(title="Staleness Management Server", docs_url=None, redoc_url=None)

    @app.post("/request_capacity")
    async def request_capacity(request: dict):
        requested = request.get("requested", 1)
        response = staleness_manager.request_capacity(requested)
        return response

    @app.post("/release_capacity")
    async def release_capacity(request: dict):
        accepted = request.get("accepted", 1)
        completed = request.get("completed", 1)
        staleness_manager.release_capacity(completed=completed, accepted=accepted)
        return {"status": "ok"}

    @app.post("/update_version")
    async def update_version(request: dict):
        staleness_manager.update_version(request["version"])
        return {"status": "ok"}

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    return app


class StalenessManager:

    def __init__(
        self,
        max_capacity: int,
        max_head_offpolicyness: int,
        consumer_batch_size: int,
        enable_rollout_tracing: bool,
    ):
        self.max_capacity = max_capacity
        self.max_head_offpolicyness = max_head_offpolicyness
        self.consumer_batch_size = consumer_batch_size

        self.lock = threading.Lock()
        self.rollout_stat = RolloutStat()
        self.current_version = 0
        self.enable_rollout_tracing = enable_rollout_tracing

    def update_version(self, version: int):
        """Update the current model version for staleness control."""
        with self.lock:
            self.current_version = version

    def request_capacity(self, requested: int) -> Dict[str, Any]:
        """Request capacity for rollouts."""
        with self.lock:
            # Staleness control
            max_allowed_samples = (
                self.max_head_offpolicyness + self.current_version + 1
            ) * self.consumer_batch_size
            sample_cnt = self.rollout_stat.accepted + self.rollout_stat.running
            staleness_capacity = max_allowed_samples - sample_cnt

            # Available capacity considering both concurrent limit and staleness
            available = min(
                self.max_capacity - self.rollout_stat.running, staleness_capacity
            )
            granted = min(requested, max(0, available))

            if granted > 0:
                self.rollout_stat.running += granted
                self.rollout_stat.submitted += granted

            return {
                "granted": granted,
                "requested": requested,
                "submitted": self.rollout_stat.submitted,
                "max_capacity": self.max_capacity,
                "running": self.rollout_stat.running,
                "accepted": self.rollout_stat.accepted,
            }

    def release_capacity(self, completed: int, accepted: int):
        """Release capacity when rollouts complete."""
        with self.lock:
            assert completed >= accepted
            self.rollout_stat.accepted += accepted
            assert completed <= self.rollout_stat.running
            self.rollout_stat.running -= completed
            if self.enable_rollout_tracing:
                logger.info(
                    f"Finish {completed} rollouts. "
                    f"Accepted: {accepted}/{completed} ({accepted/completed:.2%}). "
                    f"Submit: {self.rollout_stat.submitted}, "
                    f"running: {self.rollout_stat.running}, "
                    f"accepted: {self.rollout_stat.accepted}."
                )


class StalenessManagerServer:
    """Uvicorn-based HTTP server for staleness management."""

    def __init__(self, host: str, port: int, staleness_manager: StalenessManager):
        self.host = host
        self.port = port
        self.staleness_manager = staleness_manager
        self.app = create_capacity_app(staleness_manager)
        self.server = None
        self.server_thread = None
        self._shutdown_event = threading.Event()

    def start(self):
        """Start the staleness manager."""
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="error",  # Suppress uvicorn logging
            access_log=False,
        )
        self.server = uvicorn.Server(config)

        def run_server():
            try:
                self.server.run()
            except Exception as e:
                if not self._shutdown_event.is_set():
                    logger.error(f"Staleness server error: {e}")

        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()

        # Wait for server to start
        time.sleep(0.5)
        logger.info(f"Staleness server started on {self.host}:{self.port}")

    def stop(self):
        """Stop the staleness manager gracefully."""
        self._shutdown_event.set()
        if self.server:
            self.server.should_exit = True
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=2.0)


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

        self.staleness_manager = None
        self.staleness_manager_host = None
        self.staleness_manager_port = None
        self.staleness_manager_url = None

        qsize = config.queue_size or config.max_concurrent_rollouts * 16
        self.input_queue = queue.Queue(maxsize=qsize)
        self.output_queue = queue.Queue(maxsize=qsize)
        self.result_cache: List[_TimedResult] = []

        self.local_rollout_stat = RolloutStat()

    def initialize(self, data_parallel_group: Optional[dist.ProcessGroup] = None):
        if dist.is_initialized():
            self.dp_world_size = dist.get_world_size(group=data_parallel_group)
            self.dp_group = data_parallel_group
        else:
            self.dp_world_size = 1
            self.dp_group = None

        # Initialize staleness manager on rank 0
        # and share the address via torch distributed
        is_rank_zero = (not dist.is_initialized()) or (dist.get_rank() == 0)

        # Check environment variable for server address
        env_server_addr = os.environ.get("AREAL_STALENESS_SERVER_ADDR")
        if is_rank_zero:
            if env_server_addr:
                self.staleness_manager_host, port_str = env_server_addr.split(":")
                self.staleness_manager_port = int(port_str)
                logger.info(f"Using staleness server from env var: {env_server_addr}")
            else:
                if dist.is_initialized():
                    # Find free port and get host IP
                    # Will broadcast the address via pytorch
                    self.staleness_manager_host = network.gethostip()
                    self.staleness_manager_port = network.find_free_ports(1)[0]
                else:
                    # Unknown how to sync server addresses. Assume single-node single-process.
                    self.staleness_manager_host = "127.0.0.1"
                    self.staleness_manager_port = 11379
                server_address = (
                    f"{self.staleness_manager_host}:{self.staleness_manager_port}"
                )
                logger.info(
                    f"AREAL_STALENESS_SERVER_ADDR not set. Using staleness server: {server_address}"
                )

            staleness_manager = StalenessManager(
                max_capacity=self.config.max_concurrent_rollouts,
                max_head_offpolicyness=self.config.max_head_offpolicyness,
                consumer_batch_size=self.config.consumer_batch_size,
                enable_rollout_tracing=self.config.enable_rollout_tracing,
            )
            self.staleness_manager = StalenessManagerServer(
                self.staleness_manager_host,
                self.staleness_manager_port,
                staleness_manager,
            )
            self.staleness_manager.start()

        if env_server_addr:
            # Use environment variable
            self.staleness_manager_host, port_str = env_server_addr.split(":")
            self.staleness_manager_port = int(port_str)
            self.staleness_manager_url = f"http://{env_server_addr}"
        else:
            # Broadcast server address via PyTorch distributed
            if dist.is_initialized():
                if is_rank_zero:
                    obj_lis = [
                        f"{self.staleness_manager_host}:{self.staleness_manager_port}"
                    ]
                    # Broadcast the server address
                    dist.broadcast_object_list(obj_lis, src=0, group=self.dp_group)
                else:
                    # Receive the server address
                    obj_lis = [None]
                    dist.broadcast_object_list(obj_lis, src=0, group=self.dp_group)
                    server_address = obj_lis[0]
                    self.staleness_manager_host, port_str = server_address.split(":")
                    self.staleness_manager_port = int(port_str)
                logger.info(f"Retrieved staleness server address: {server_address}")

            self.staleness_manager_url = (
                f"http://{self.staleness_manager_host}:{self.staleness_manager_port}"
            )

        self.rollout_tasks: Dict[str, asyncio.Task] = {}
        self.rollout_thread = threading.Thread(target=self._rollout_thread, daemon=True)
        self.rollout_thread.start()

    def destroy(self):
        self.exiting.set()
        self.rollout_thread.join()
        if self.staleness_manager:
            self.staleness_manager.stop()
            logger.info("Staleness server stopped")

    def get_local_capacity(self):
        with self.lock:
            max_concurrent_rollouts = max(
                1, self.config.max_concurrent_rollouts // self.dp_world_size
            )
            capacity = max_concurrent_rollouts - len(self.rollout_tasks)
            # Staleness control
            version = self.inference_engine.get_version()
            ofp = self.config.max_head_offpolicyness
            sample_cnt = (
                self.local_rollout_stat.accepted + self.local_rollout_stat.running
            )
            consumer_bs = max(1, self.config.consumer_batch_size // self.dp_world_size)
            capacity = min(capacity, (ofp + version + 1) * consumer_bs - sample_cnt)
        return capacity

    def get_capacity(self):
        """Request capacity from the server."""
        local_capacity = self.get_local_capacity()
        with self.lock:
            response = requests.post(
                f"{self.staleness_manager_url}/update_version",
                json={"version": self.inference_engine.get_version()},
                timeout=2.0,
            )
            response.raise_for_status()
            # Local constraint: at least fill up the local capacity
            # If exceeding the local capacity (data imbalance), we will over-subscribe one-by-one
            requested = min(self.input_queue.qsize(), max(local_capacity, 1))

        # Request from remote server (no lock needed for HTTP calls)
        try:
            response = requests.post(
                f"{self.staleness_manager_url}/request_capacity",
                json={"requested": requested},
                timeout=2.0,
            )
            if response.status_code == 200:
                result = response.json()
                return result.get("granted", 0)
            else:
                logger.warning(
                    f"Capacity request failed with status {response.status_code}: {response.text}"
                )
                return 0
        except Exception as e:
            logger.warning(f"Capacity request failed with exception: {e}")
            return 0

    def release_capacity(self, completed: int, accepted: int):
        """Release capacity back to the server."""
        requests.post(
            f"{self.staleness_manager_url}/release_capacity",
            json={"completed": completed, "accepted": accepted},
            timeout=2.0,
        )

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

                # Create new rollout tasks
                with self.lock:
                    assert capacity <= self.input_queue.qsize()
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
                        self.local_rollout_stat.submitted += 1
                        self.local_rollout_stat.running += 1
                        if self.config.enable_rollout_tracing:
                            logger.info(
                                f"Submit rollout rid {rid}. "
                                f"Submit: {self.local_rollout_stat.submitted}, "
                                f"running: {self.local_rollout_stat.running}, "
                                f"accepted: {self.local_rollout_stat.accepted}."
                            )
                        capacity -= 1
                        rid += 1
                    tasks = list(rollout_tasks.values())

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
                        self.local_rollout_stat.accepted += 1
                        self.local_rollout_stat.running -= 1
                        if self.config.enable_rollout_tracing:
                            logger.info(
                                f"Finish rollout {task_rid}. "
                                f"Submit: {self.local_rollout_stat.submitted}, "
                                f"running: {self.local_rollout_stat.running}, "
                                f"accepted: {self.local_rollout_stat.accepted}."
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
                _accepted = _completed = 0
                try:
                    timed_result = self.output_queue.get_nowait()
                    _completed += 1
                    if timed_result.data is not None and (
                        should_accept is None or should_accept(timed_result.data)
                    ):
                        if self.config.enable_rollout_tracing:
                            logger.info(
                                f"Accept rollout result. accepted/count = {len(self.result_cache)}/{count}"
                            )
                        _accepted += 1
                        self.result_cache.append(timed_result)
                    else:
                        if self.config.enable_rollout_tracing:
                            logger.info(f"Rollout is rejected.")
                        with self.lock:
                            self.local_rollout_stat.accepted -= 1
                    self.release_capacity(completed=_completed, accepted=_accepted)
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
