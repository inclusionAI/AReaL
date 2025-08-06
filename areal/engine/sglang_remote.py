import asyncio
import os
import random
import shutil
import threading
import time
import traceback
from concurrent.futures import Future, ProcessPoolExecutor
from datetime import datetime
from queue import Empty, Full, Queue
from typing import TYPE_CHECKING, Any, Callable, Dict, List

import aiohttp
import requests
import torch.distributed as dist
import uvloop
from tensordict import TensorDict
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.cli_args import InferenceEngineConfig
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import (
    FinetuneSpec,
    LLMRequest,
    LLMResponse,
    RolloutStat,
    VLMRequest,
    VLMResponse,
    WeightUpdateMeta,
)
from areal.utils.data import concat_padded_tensors
from areal.utils.http import arequest_with_retry, get_default_connector
from realhf.base import logging, name_resolve, names

if TYPE_CHECKING:
    from areal.api.workflow_api import RolloutWorkflow
logger = logging.getLogger(__name__)


ROLLOUT_POLL_WAIT_TIME = 0.05
RID_CACHE_SIZE = 128


class RemoteSGLangEngine(InferenceEngine):

    def __init__(self, config: InferenceEngineConfig):
        config.max_concurrent_rollouts = (
            config.max_concurrent_rollouts or config.consumer_batch_size
        )
        self.config = config

        self.rid_to_address = {}
        # Maintain the addresses for the recent 128 requests
        self.rid_queue = []

        self.addresses = os.getenv("AREAL_LLM_SERVER_ADDRS").split(",")
        if not self.addresses:
            raise RuntimeError("No configured SGLang servers.")
        logger.info("Waiting for server ready...")
        for addr in self.addresses:
            self._wait_for_server(addr)
        logger.info("Servers are all ready!")

        self.server_idx = random.randint(0, len(self.addresses) - 1)

        qsize = config.queue_size or config.max_concurrent_rollouts * 16
        self.input_queue = Queue(maxsize=qsize)
        self.output_queue = Queue(maxsize=qsize)
        self.result_cache = []

        self.exiting = threading.Event()
        self.paused = threading.Event()
        self.lock = threading.Lock()

        self.rollout_stat = RolloutStat()
        self.distributed_weight_update_initialized = False

        self._version = 0

    def _wait_for_server(self, address):
        base_url = f"http://{address}"
        tik = time.time()
        while time.time() - tik < self.config.setup_timeout:
            if self.check_health(base_url):
                return
            time.sleep(1)
        raise RuntimeError("server launch failed")

    def check_health(self, base_url):
        # Check server endpoint
        try:
            response = requests.get(f"{base_url}/health", timeout=30)
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            return False

    def initialize(self, addr: str | None, ft_spec: FinetuneSpec = None):
        self.rollout_tasks: Dict[str, asyncio.Task] = {}

        self.executor = ProcessPoolExecutor(max_workers=1)
        self.rollout_thread = threading.Thread(target=self._rollout_thread)
        self.rollout_thread.start()

    def destroy(self):
        self.executor.shutdown()
        self.exiting.set()
        self.rollout_thread.join()

    def set_version(self, version):
        with self.lock:
            self._version = version

    def get_version(self):
        with self.lock:
            return self._version

    def _rollout_thread(self):
        """Thread that runs the rollout loop."""
        try:
            uvloop.run(self._rollout_thread_async())
        except Exception:
            traceback.print_exc()

    async def _rollout_thread_async(self):
        rollout_tasks = self.rollout_tasks
        rid = 0

        # NOTE: session is not thread-safe, but we only submit requests in the sub-thread.
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=self.config.request_timeout,
                sock_connect=self.config.request_timeout,
                connect=self.config.request_timeout,
            ),
            read_bufsize=1024 * 1024 * 10,
            connector=get_default_connector(),
        )

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
                    data, workflow = self.input_queue.get_nowait()
                    logger.debug(f"Get data from puller: {data}")
                    task = asyncio.create_task(
                        workflow.arun_episode(self, data), name=str(rid)
                    )
                    with self.lock:
                        rollout_tasks[str(rid)] = task
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
                # Wait for rollout completion
                with self.lock:
                    tasks = list(rollout_tasks.values())
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
                    traj: TensorDict
                    task_rid = task.get_name()
                    with self.lock:
                        rollout_tasks.pop(task_rid)
                        self.rollout_stat.accepted += 1

                    try:
                        self.output_queue.put_nowait(traj)
                    except Full:
                        raise RuntimeError(
                            "Output queue full. Please increase queue_size."
                        )

                    with self.lock:
                        self.rollout_stat.running -= 1
                        if self.config.enable_rollout_tracing:
                            logger.info(
                                f"Finish rollout {task_rid}. "
                                f"Submit: {self.rollout_stat.submitted}, "
                                f"running: {self.rollout_stat.running}, "
                                f"accepted: {self.rollout_stat.accepted}."
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

    def choose_server(self) -> str:
        with self.lock:
            if self.config.schedule_policy == "round_robin":
                server = self.addresses[self.server_idx]
                self.server_idx = (self.server_idx + 1) % len(self.addresses)
                return server
        raise NotImplementedError("Only round-robin scheduling is implemented.")

    async def agenerate(
        self, req: LLMRequest | VLMRequest
    ) -> LLMResponse | VLMResponse:
        """Async version of generate using aiohttp."""
        # Prepare request payload
        gconfig = req.gconfig
        stop_token_ids = gconfig.stop_token_ids
        stop = gconfig.stop

        if gconfig.n_samples != 1:
            raise ValueError(
                "RemoteSGLangEngine does not support n_samples > 1. "
                "Please call generate for multiple times with n_samples = 1."
            )
        sample_params = {
            "top_p": gconfig.top_p,
            "top_k": gconfig.top_k,
            "max_new_tokens": gconfig.max_new_tokens,
            "temperature": 0.0 if gconfig.greedy else gconfig.temperature,
            "stop_token_ids": stop_token_ids,
        }
        if stop is not None:
            sample_params["stop"] = stop

        if isinstance(req, VLMRequest):
            # VLMRequest has image_data
            payload = {
                "input_ids": req.input_ids.copy(),
                "image_data": req.image_data,  # ImageObject or str
                "sampling_params": sample_params,
                "return_logprob": True,
                "stream": False,
            }
        else:
            # NOTE: rid should NOT be passed in payload
            payload = {
                "input_ids": req.input_ids.copy(),
                "sampling_params": sample_params,
                "return_logprob": True,
                "stream": False,
            }

        # Make request
        start_time = time.perf_counter()
        accumulated_output_tokens = []
        accumulated_output_logprobs = []
        accumulated_versions = []

        # A single "rid" shares the same sever to allow KV cache reuse
        if req.rid in self.rid_to_address:
            server_addr = self.rid_to_address[req.rid]
        else:
            server_addr = self.choose_server()
            if len(self.rid_queue) >= RID_CACHE_SIZE:
                # Remove the oldest entry if cache is full
                oldest_rid = self.rid_queue.pop(0)
                self.rid_to_address.pop(oldest_rid, None)
            self.rid_to_address[req.rid] = server_addr
            self.rid_queue.append(req.rid)

        # Deal with rollout interruption
        # "abort" is the stop reason for later v0.4.9.post2 after
        # we call the pause_generation endpoint
        stop_reason = None
        while (
            stop_reason != "stop"
            and len(accumulated_output_tokens) < gconfig.max_new_tokens
        ):
            # Request is interrupted, wait for some time to avoid interfering
            # with update weights requests
            if stop_reason is not None:
                await asyncio.sleep(0.5)

            # loop until the generation is complete
            result = await arequest_with_retry(
                session=self.session,
                addr=server_addr,
                endpoint="/generate",
                payload=payload,
                method="POST",
                max_retries=self.config.request_retries,
                timeout=self.config.request_timeout,
            )

            meta_info = result["meta_info"]
            # Check if generation is complete
            finish_reason = meta_info["finish_reason"]
            stop_reason = finish_reason["type"]
            if (
                stop_reason == "abort"
                and finish_reason.get("message") == "Abort before prefill"
            ):
                continue

            # Parse response
            output_tokens = [x[1] for x in meta_info["output_token_logprobs"]]
            output_logprobs = [x[0] for x in meta_info["output_token_logprobs"]]

            # Update accumulated outputs
            accumulated_output_tokens.extend(output_tokens)
            accumulated_output_logprobs.extend(output_logprobs)
            # FIXME: Update with actual server versions
            accumulated_versions.extend([-1] * len(output_tokens))

            payload["input_ids"] += output_tokens
            sample_params["max_new_tokens"] -= len(output_tokens)

        latency = time.perf_counter() - start_time

        if isinstance(req, VLMRequest):
            response = VLMResponse(
                input_tokens=req.input_ids,
                input_images=req.image_data,
                output_tokens=accumulated_output_tokens,
                output_logprobs=accumulated_output_logprobs,
                output_versions=accumulated_versions,
                stop_reason=stop_reason,
                latency=latency,
                ttft=latency,  # Simplified for non-streaming
            )
        else:
            response = LLMResponse(
                input_tokens=req.input_ids,
                output_tokens=accumulated_output_tokens,
                output_logprobs=accumulated_output_logprobs,
                output_versions=accumulated_versions,
                stop_reason=stop_reason,
                latency=latency,
                ttft=latency,  # Simplified for non-streaming
            )
        return response

    def update_weights(self, meta: WeightUpdateMeta):
        for addr in self.addresses:
            res = requests.post(f"http://{addr}/pause_generation")
            res.raise_for_status()
        fut = Future()
        if meta.type == "nccl":
            fut = self.executor.submit(
                update_weights_from_distributed,
                meta,
                self.addresses,
                self.config.request_timeout,
                not self.distributed_weight_update_initialized,
            )

            def callback(fut):
                self.distributed_weight_update_initialized = True

            fut.add_done_callback(callback)
        elif meta.type == "disk":
            # Update weights from disk
            # Use ProcessPool to bypass python GIL for running async coroutines
            fut = self.executor.submit(
                update_weights_from_disk,
                self.config.experiment_name,
                self.config.trial_name,
                self.get_version(),
                self.addresses,
                meta.path,
                self.config.request_retries,
                self.config.request_timeout,
            )

            def callback(fut):
                shutil.rmtree(meta.path, ignore_errors=True)

            fut.add_done_callback(callback)
        else:
            raise NotImplementedError(f"Unsupported weight update type: {meta.type}")

        def callback(fut):
            for addr in self.addresses:
                res = requests.post(f"http://{addr}/continue_generation")
                res.raise_for_status()

        fut.add_done_callback(callback)
        return fut

    def get_capacity(self):
        if dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = 1

        max_concurrent_rollouts = max(
            1, self.config.max_concurrent_rollouts // world_size
        )
        capacity = max_concurrent_rollouts - len(self.rollout_tasks)
        # Staleness control
        version = self.get_version()
        ofp = self.config.max_head_offpolicyness
        with self.lock:
            sample_cnt = self.rollout_stat.accepted + self.rollout_stat.running
        consumer_bs = max(1, self.config.consumer_batch_size // world_size)
        capacity = min(capacity, (ofp + version + 1) * consumer_bs - sample_cnt)
        return capacity

    def submit(self, data: Dict[str, Any], workflow: "RolloutWorkflow") -> None:
        try:
            self.input_queue.put_nowait((data, workflow))
        except Full:
            raise RuntimeError("Input queue full. Please increase queue_size.")

    def wait(
        self,
        count: int,
        timeout: float | None = None,
        should_accept: Callable | None = None,
    ) -> TensorDict:
        tik = time.perf_counter()
        accepted = len(self.result_cache)
        timeout = timeout or float(7 * 24 * 3600)
        while (
            accepted < count
            and not self.exiting.is_set()
            and time.perf_counter() - tik < timeout
        ):
            try:
                result = self.output_queue.get(timeout=ROLLOUT_POLL_WAIT_TIME)
                if should_accept is None or should_accept(result):
                    self.result_cache.append(result)
                    accepted += 1
                else:
                    with self.lock:
                        self.rollout_stat.accepted -= 1
            except Empty:
                pass
        if self.exiting.is_set():
            raise RuntimeError("Rollout engine is exiting, cannot wait for results.")
        if accepted < count:
            raise TimeoutError(
                f"Timed out waiting for {count} rollouts, " f"only received {accepted}."
            )
        results, self.result_cache = (
            self.result_cache[:count],
            self.result_cache[count:],
        )
        return concat_padded_tensors(results)

    def rollout_batch(
        self, data: List[Dict[str, Any]], workflow: "RolloutWorkflow"
    ) -> TensorDict:
        """Submit a batch of requests to the inference engine and wait for the results."""
        for item in data:
            self.submit(item, workflow)
        return self.wait(count=len(data))

    def prepare_batch(
        self,
        dataloader: StatefulDataLoader,
        workflow: RolloutWorkflow,
        should_accept: Callable | None = None,
    ):
        return self.workflow_executor.prepare_batch(dataloader, workflow, should_accept)

    def pause(self):
        self.paused.set()

    def resume(self):
        self.paused.clear()


def update_weights_from_disk(
    experiment_name,
    trial_name,
    model_version,
    addresses,
    path,
    request_retries,
    request_timeout,
):
    async def _fn():
        # Wait for model checkpoints of meta.version
        update_name = names.update_weights_from_disk(
            experiment_name, trial_name, model_version
        )
        save_timestamp = float(name_resolve.wait(update_name, timeout=120))
        load_timestamp = datetime.now().timestamp()
        logger.info(
            f"Begin update weights from {path}, responded in {(load_timestamp - save_timestamp):.2f}s"
        )
        session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=request_timeout,
                sock_connect=request_timeout,
                connect=request_timeout,
            ),
            read_bufsize=1024 * 1024 * 10,
            connector=get_default_connector(),
        )
        jobs = [
            arequest_with_retry(
                addr=addr,
                session=session,
                endpoint="/update_weights_from_disk",
                payload=dict(model_path=str(path)),
                method="POST",
                max_retries=request_retries,
                timeout=request_timeout,
            )
            for addr in addresses
        ]
        await asyncio.gather(*jobs)
        await session.close()
        logger.info(
            f"Loading weights done in {(datetime.now().timestamp() - load_timestamp):.2f}s"
        )

    return uvloop.run(_fn())


def update_weights_from_distributed(
    meta: WeightUpdateMeta,
    addresses: List[str],
    request_timeout,
    init_group: bool,
):
    async def _fn():
        tik = time.perf_counter()
        if init_group:
            await asyncio.gather(
                *[
                    ainit_weights_update_group(addr, i, meta, request_timeout)
                    for i, addr in enumerate(addresses)
                ]
            )
        await asyncio.gather(
            *[
                arequest_with_retry(
                    addr=addr,
                    endpoint="/update_weights_from_distributed",
                    payload={
                        "names": [pspec.name for pspec in meta.nccl_param_specs],
                        "dtypes": [pspec.dtype for pspec in meta.nccl_param_specs],
                        "shapes": [pspec.shape for pspec in meta.nccl_param_specs],
                        "group_name": meta.nccl_group_name,
                    },
                    method="POST",
                    max_retries=1,
                    timeout=request_timeout,
                )
                for addr in addresses
            ]
        )

        logger.info(f"Distributed update weights done in {time.perf_counter() - tik}s")

    return uvloop.run(_fn())


async def ainit_weights_update_group(
    addr: str,
    server_idx: int,
    meta: WeightUpdateMeta,
    request_timeout: float,
):
    assert meta.alloc_mode is not None
    if meta.alloc_mode.gen_pp_size != 1:
        raise NotImplementedError(
            "NCCL weight update with PP size > 1 is not implemented yet."
        )
    rank_offset = 1 + server_idx * meta.alloc_mode.gen_tp_size
    payload = {
        "master_address": meta.nccl_master_address,
        "master_port": str(meta.nccl_master_port),
        "rank_offset": rank_offset,
        "world_size": meta.alloc_mode.gen_world_size + 1,
        "backend": "nccl",
        "group_name": meta.nccl_group_name,
    }
    res = await arequest_with_retry(
        addr=addr,
        endpoint="/init_weights_update_group",
        payload=payload,
        method="POST",
        max_retries=1,
        timeout=request_timeout,
    )
    assert res["success"]
