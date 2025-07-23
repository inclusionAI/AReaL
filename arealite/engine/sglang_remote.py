import asyncio
import os
import random
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime
from queue import Empty, Full, Queue
from typing import TYPE_CHECKING, Any, Callable, Dict, List

import aiohttp
import requests
import torch.distributed as dist
import uvloop
from tensordict import TensorDict
from torchdata.stateful_dataloader import StatefulDataLoader

from arealite.api.cli_args import InferenceEngineConfig
from arealite.api.engine_api import InferenceEngine
from arealite.api.io_struct import (
    FinetuneSpec,
    LLMRequest,
    LLMResponse,
    RolloutStat,
    VLMRequest,
    VLMResponse,
    WeightUpdateMeta,
)
from arealite.utils.data import concat_padded_tensors
from arealite.utils.http import arequest_with_retry, get_default_connector
from realhf.base import logging, name_resolve, names, pkg_version

if TYPE_CHECKING:
    from arealite.api.workflow_api import RolloutWorkflow
logger = logging.getLogger(__name__)

if pkg_version.is_available("sglang"):
    if pkg_version.is_version_greater_or_equal("sglang", "0.4.4"):
        SGLANG_TOKEN_OUTPUT_IDENTIFIER = "output_ids"
    else:
        SGLANG_TOKEN_OUTPUT_IDENTIFIER = "token_ids"

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
            response = requests.get(
                f"{base_url}/metrics",
                timeout=30,
            )
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
        except Exception as e:
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

    async def agenerate(self, req: LLMRequest|VLMRequest) -> LLMResponse|VLMResponse:
        """Async version of generate using aiohttp."""
        # Prepare request payload
        gconfig = req.gconfig
        stop_token_ids = gconfig.stop_token_ids

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

        # Deal with rollout interruption
        stop_reason = "length"

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

        while (
            stop_reason != "stop"
            and len(accumulated_output_tokens) < gconfig.max_new_tokens
        ):
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

            # Parse response
            meta_info = result["meta_info"]
            output_tokens = [x[1] for x in meta_info["output_token_logprobs"]]
            output_logprobs = [x[0] for x in meta_info["output_token_logprobs"]]

            # Update accumulated outputs
            accumulated_output_tokens.extend(output_tokens)
            accumulated_output_logprobs.extend(output_logprobs)
            # FIXME: Update with actual server versions
            accumulated_versions.extend([-1] * len(output_tokens))

            # Check if generation is complete
            finish_reason = meta_info["finish_reason"]
            stop_reason = finish_reason["type"]

            payload["input_ids"] += result[SGLANG_TOKEN_OUTPUT_IDENTIFIER]
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
            response=LLMResponse(
            input_tokens=req.input_ids,
            output_tokens=accumulated_output_tokens,
            output_logprobs=accumulated_output_logprobs,
            output_versions=accumulated_versions,
            stop_reason=stop_reason,
            latency=latency,
            ttft=latency,  # Simplified for non-streaming
        )
        return response

    def update_weights(self, meta):
        executor = ThreadPoolExecutor(max_workers=1)
        return executor.submit(self._update_weights, meta)

    def _update_weights(self, meta: WeightUpdateMeta):
        if meta.type == "nccl":
            if not self.distributed_weight_update_initialized:
                self._init_distributed_weight_update(meta)
            tik = time.perf_counter()
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                for param in meta.parameter_names:
                    jobs = [
                        self.aupdate_weights_from_distributed(addr, meta, param)
                        for addr in self.addresses
                    ]
                    loop.run_until_complete(asyncio.gather(*jobs))
            finally:
                loop.close()
            logger.info(
                f"Distributed update weights done in {time.perf_counter() - tik}s"
            )
            self.set_version(meta.model_version)
        elif meta.type == "disk":
            # Update weights from disk
            # Use ProcessPool to bypass python GIL for running async coroutines
            fut = self.executor.submit(
                update_weights_from_disk,
                self.config.experiment_name,
                self.config.trial_name,
                meta.model_version,
                self.addresses,
                meta.path,
                self.config.request_retries,
                self.config.request_timeout,
            )

            def callback(fut):
                self.set_version(meta.model_version)

            fut.add_done_callback(callback)
            return fut
        else:
            raise NotImplementedError(f"Unsupported weight update type: {meta.type}")

    def _init_distributed_weight_update(self, meta: WeightUpdateMeta):
        try:
            # Initialize weights update group
            jobs = [
                self.ainit_weights_update_group(addr, meta) for addr in self.addresses
            ]
            loop = asyncio.new_event_loop()
            # asyncio event loop should be manually set when running asyncio stuff in another thread
            asyncio.set_event_loop(loop)
            loop.run_until_complete(asyncio.gather(*jobs))
            self.distributed_weight_update_initialized = True
            logger.info(f"Distributed update weights initialized")
        finally:
            loop.close()

    async def ainit_weights_update_group(self, addr: str, meta: WeightUpdateMeta):
        rank_offset = 1 + self.addresses.index(addr) * meta.tp_size
        payload = {
            "master_address": meta.master_address,
            "master_port": str(meta.master_port),
            "rank_offset": rank_offset,
            "world_size": meta.world_size,
            "group_name": meta.group_name,
            "backend": "nccl",
        }
        res = await arequest_with_retry(
            addr=addr,
            endpoint="/init_weights_update_group",
            payload=payload,
            method="POST",
            max_retries=1,
            timeout=self.config.request_timeout,
        )
        assert res["success"]

    async def aupdate_weights_from_distributed(
        self, addr: str, meta: WeightUpdateMeta, parameter_name: str
    ):
        res = await arequest_with_retry(
            addr=addr,
            endpoint="/update_weights_from_distributed",
            payload={
                "name": parameter_name,
                "dtype": "bfloat16",
                "shape": meta.state_dict_key_to_shape[parameter_name],
            },
            method="POST",
            max_retries=1,
            timeout=self.config.request_timeout,
        )
        assert res["success"]

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
        workflow: "RolloutWorkflow",
    ):
        if not hasattr(self, "data_generator"):
            self.data_generator = iter(dataloader)
        assert dataloader.batch_size is not None
        while True:
            # Submit at least two batches to allow maximum overlap
            if (
                self.get_capacity() + dataloader.batch_size > 0
                and self.input_queue.qsize() + dataloader.batch_size
                < self.input_queue.maxsize
            ):
                try:
                    data = next(self.data_generator)
                    
                except StopIteration:
                    self.data_generator = iter(dataloader)
                    data = next(self.data_generator)
                for item in data:
                    self.submit(item, workflow=workflow)
            try:
                return self.wait(dataloader.batch_size, timeout=1)
            except TimeoutError:
                pass

    def pause(self):
        self.paused.set()

    def resume(self):
        self.paused.clear()


async def aupdate_weights_from_disk(
    session, addr, path: str, request_retries: int, request_timeout: float
):
    tik = time.time()
    res = await arequest_with_retry(
        addr=addr,
        session=session,
        endpoint="/update_weights_from_disk",
        payload=dict(model_path=str(path), allow_interrupt=True),
        method="POST",
        max_retries=request_retries,
        timeout=request_timeout,
    )
    assert res["success"]
    if "num_paused_requests" in res:
        logger.info(
            f"{res['num_paused_requests']} requests are interrupted "
            f"during updating weights for server {addr}"
        )


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
            aupdate_weights_from_disk(
                session=session,
                addr=addr,
                path=path,
                request_retries=request_retries,
                request_timeout=request_timeout,
            )
            for addr in addresses
        ]
        await asyncio.gather(*jobs)
        await session.close()
        logger.info(
            f"Loading weights done in {(datetime.now().timestamp() - load_timestamp):.2f}s"
        )

    return uvloop.run(_fn())
