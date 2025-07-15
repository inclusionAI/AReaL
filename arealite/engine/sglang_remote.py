import asyncio
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Full, Queue
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import aiohttp
import torch.distributed as dist
from tensordict import TensorDict

from arealite.api.cli_args import InferenceEngineConfig
from arealite.api.engine_api import InferenceEngine
from arealite.api.io_struct import (
    LLMRequest,
    LLMResponse,
    RolloutStat,
    WeightUpdateMeta,
)
from realhf.base import logging, name_resolve, names, pkg_version

if TYPE_CHECKING:
    from arealite.api.workflow_api import RolloutWorkflow
logger = logging.getLogger(__name__)

if pkg_version.is_available("sglang"):
    if pkg_version.is_version_greater_or_equal("sglang", "0.4.4"):
        SGLANG_TOKEN_OUTPUT_IDENTIFIER = "output_ids"
    else:
        SGLANG_TOKEN_OUTPUT_IDENTIFIER = "token_ids"

ROLLOUT_POLL_WAIT_TIME = 0.4
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

        self.addresses = config.server_addrs
        self.server_idx = 0

        qsize = config.queue_size or config.max_concurrent_rollouts * 10
        self.input_queue = Queue(maxsize=qsize)
        self.output_queue = Queue(maxsize=qsize)
        self.result_cache = []

        self.exiting = threading.Event()
        self.lock = threading.Lock()

        self.rollout_stat = RolloutStat()
        self.distributed_weight_update_initialized = False

        self._version = 0

    def initialize(self, addr: str | None, ft_spec: Optional[Dict[str, Any]] = None):
        self.rollout_thread = threading.Thread(target=self._rollout_thread)
        self.rollout_thread.start()

    def destroy(self):
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
            asyncio.run(self._rollout_thread_async())
        except Exception as e:
            traceback.print_exc()

    async def _rollout_thread_async(self):
        data = None

        rollout_tasks: Dict[str, asyncio.Task] = {}
        rid = 0

        try:
            while not self.exiting.is_set():
                # Load next data from controller
                if data is None:
                    try:
                        data, workflow = self.input_queue.get_nowait()
                        logger.info(f"Get data from puller: {data}")
                    except Empty:
                        logger.debug(f"No data from puller stream.")

                # Check capacity
                if dist.is_initialized():
                    world_size = dist.get_world_size()
                else:
                    world_size = 1

                cannot_rollout_reason = []
                capacity = max(1, self.config.max_concurrent_rollouts // world_size)
                can_rollout = len(rollout_tasks) < capacity
                if not can_rollout:
                    cannot_rollout_reason.append(
                        f"Exceeding capacity: # running tasks {len(rollout_tasks)} >= capacity {capacity}"
                    )

                # Staleness control
                version = self.get_version()
                ofp = self.config.max_head_offpolicyness
                with self.lock:
                    sample_cnt = self.rollout_stat.accepted + self.rollout_stat.running
                expected_version = sample_cnt // self.config.consumer_batch_size
                not_staled = expected_version <= ofp + version
                can_rollout &= not_staled
                if not not_staled:
                    cannot_rollout_reason.append(
                        f"Staled: expected version ({expected_version}) = "
                        f"global sample cnt ({sample_cnt}) // batch size ({self.config.consumer_batch_size}), "
                        f"current latest version {version}, "
                        f"offpolicyness {self.config.max_head_offpolicyness}."
                    )

                if not can_rollout:
                    logger.debug(
                        f"Cannot submit new rollouts. "
                        + "\n".join(cannot_rollout_reason)
                    )

                # Create new rollout task
                if can_rollout and data is not None:
                    task = asyncio.create_task(
                        workflow.arun_episode(self, data), name=str(rid)
                    )
                    rollout_tasks[str(rid)] = task

                    with self.lock:
                        self.rollout_stat.submitted += 1
                        self.rollout_stat.running += 1
                        logger.info(
                            f"Submit rollout rid {rid}. "
                            f"Submit: {self.rollout_stat.submitted}, "
                            f"running: {self.rollout_stat.running}, "
                            f"accepted: {self.rollout_stat.accepted}."
                        )

                    rid += 1
                    data = None

                # Wait for rollout completion
                tasks = list(rollout_tasks.values())
                done = []
                if tasks:
                    done, _ = await asyncio.wait(
                        tasks,
                        timeout=ROLLOUT_POLL_WAIT_TIME,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                else:
                    await asyncio.sleep(ROLLOUT_POLL_WAIT_TIME)

                # Collect done results
                for task in done:
                    traj = await task
                    traj: TensorDict
                    task_rid = task.get_name()
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
                        logger.info(
                            f"Finish rollout {task_rid}. "
                            f"Submit: {self.rollout_stat.submitted}, "
                            f"running: {self.rollout_stat.running}, "
                            f"accepted: {self.rollout_stat.accepted}."
                        )
        finally:
            # Cancel remaining tasks
            for task in rollout_tasks.values():
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

    def choose_server(self) -> str:
        if self.config.schedule_policy == "round_robin":
            server = self.addresses[self.server_idx]
            self.server_idx = (self.server_idx + 1) % len(self.addresses)
            return server
        raise NotImplementedError("Only round-robin scheduling is implemented.")

    async def arequest_with_retry(
        self,
        endpoint: str,
        payload: Optional[Dict[str, Any]] = None,
        method: str = "POST",
        max_retries: Optional[int] = None,
        timeout: Optional[float] = None,
        retry_delay: float = 1.0,
        target_addr: Optional[str] = None,
    ) -> aiohttp.ClientResponse:
        timeout = timeout or self.config.request_timeout
        last_exception = None
        max_retries = max_retries or self.config.request_retries

        # Try with retries
        for _ in range(max_retries):
            if target_addr:
                addr = target_addr
            else:
                addr = self.choose_server()
            base_url = f"http://{addr}"
            url = f"{base_url}{endpoint}"

            for attempt in range(max_retries):
                try:
                    async with aiohttp.ClientSession(
                        timeout=aiohttp.ClientTimeout(
                            total=timeout,
                            sock_connect=30,
                            sock_read=timeout,
                        )
                    ) as session:
                        if method.upper() == "GET":
                            response = await session.get(url)
                        elif method.upper() == "POST":
                            response = await session.post(url, json=payload)
                        elif method.upper() == "PUT":
                            response = await session.put(url, json=payload)
                        elif method.upper() == "DELETE":
                            response = await session.delete(url)
                        else:
                            raise ValueError(f"Unsupported HTTP method: {method}")

                        response.raise_for_status()
                        return response

                except (
                    aiohttp.ClientError,
                    aiohttp.ClientResponseError,
                    asyncio.TimeoutError,
                ) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                    continue
        raise RuntimeError(
            f"Failed after {max_retries} retries each. " f"Last error: {last_exception}"
        )

    async def agenerate(self, req: LLMRequest) -> LLMResponse:
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

        # NOTE: rid should NOT be passed in payload
        payload = {
            "text": req.text,
            "sampling_params": sample_params,
            "return_logprob": True,
            "stream": False,
        }
        if req.text:
            payload["text"] = req.text
        else:
            payload["input_ids"] = req.input_ids

        # Make request
        start_time = time.perf_counter()
        accumulated_output_tokens = []
        accumulated_output_logprobs = []
        accumulated_versions = []

        # Deal with rollout interruption
        completions = ""
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
            response = await self.arequest_with_retry(
                endpoint="/generate",
                payload=payload,
                method="POST",
                max_retries=3,
                timeout=self.config.request_timeout,
                target_addr=server_addr,
            )
            result = await response.json()

            # Parse response
            completions += result["text"]
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

            payload["text"] += result["text"]

        latency = time.perf_counter() - start_time

        return LLMResponse(
            completions=completions,
            input_tokens=req.input_ids,
            output_tokens=accumulated_output_tokens,
            output_logprobs=accumulated_output_logprobs,
            output_versions=accumulated_versions,
            stop_reason=stop_reason,
            latency=latency,
            ttft=latency,  # Simplified for non-streaming
        )

    def update_weights(self, meta):
        executor = ThreadPoolExecutor(max_workers=1)
        return executor.submit(self._update_weights, meta)

    def _update_weights(self, meta: WeightUpdateMeta):
        print("_update_weights in sglang remote called with meta")
        if meta.type == "nccl":
            if not self.distributed_weight_update_initialized:
                self._init_distributed_weight_update(meta)
            print(f"Begin update weights from {meta}")
            try:
                # Update weights from distributed
                for parameter_name in meta.parameter_names:
                    jobs = [
                        self.aupdate_weights_from_distributed(
                            addr, meta, parameter_name
                        )
                        for addr in self.addresses
                    ]
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(asyncio.gather(*jobs))
            finally:
                loop.close()
            logger.info(f"Distributed update weights done")
            self.set_version(meta.model_version)

        elif meta.type == "disk":
            # Update weights from disk
            # Wait for model checkpoints of meta.version
            update_name = names.update_weights_from_disk(
                self.config.experiment_name, self.config.trial_name, meta.model_version
            )
            save_timestamp = int(name_resolve.wait(update_name, timeout=120))
            load_timestamp = time.time_ns()
            logger.info(
                f"Begin update weights from {meta.path}, responded in {(load_timestamp - save_timestamp)/1e6:.2f} ms"
            )
            try:
                jobs = [
                    self.aupdate_weights_from_disk(addr, meta.path)
                    for addr in self.addresses
                ]
                loop = asyncio.new_event_loop()
                # asyncio event loop should be manually set when running asyncio stuff in another thread
                asyncio.set_event_loop(loop)
                loop.run_until_complete(asyncio.gather(*jobs))
            finally:
                loop.close()
            logger.info(
                f"Loading weights done in {(time.time_ns() - load_timestamp)/1e6:.2f} ms"
            )
            self.set_version(meta.model_version)
        else:
            raise NotImplementedError(f"Unsupported weight update type: {meta.type}")

    def _init_distributed_weight_update(self, meta: WeightUpdateMeta):
        logger.info(f"Begin init distributed weight update")
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
        payload={
            "master_address": meta.master_address,
            "master_port": meta.master_port,
            "rank_offset": rank_offset,
            "world_size": meta.world_size,
            "group_name": meta.group_name,
            "backend": "nccl",
        }
        print(payload)
        response = await self.arequest_with_retry(
            endpoint="/init_weights_update_group",
            payload=payload,
            method="POST",
            max_retries=3,
            timeout=self.config.request_timeout,
        )
        res = await response.json()
        assert res["success"]
        if "num_paused_requests" in res:
            logger.info(
                f"{res['num_paused_requests']} requests are interrupted "
                f"during updating weights for server {addr}"
            )

    async def aupdate_weights_from_distributed(
        self, addr: str, meta: WeightUpdateMeta, parameter_name: str
    ):
        response = await self.arequest_with_retry(
            endpoint="/update_weights_from_distributed",
            payload={
                "name": parameter_name,
                "dtype": "bfloat16",
                "shape": meta.state_dict_key_to_shape[parameter_name],
            },
            method="POST",
            max_retries=3,
            timeout=self.config.request_timeout,
        )
        res = await response.json()
        assert res["success"]
        if "num_paused_requests" in res:
            logger.info(
                f"{res['num_paused_requests']} requests are interrupted "
                f"during updating weights for server {addr}"
            )

    async def aupdate_weights_from_disk(self, addr, path: str):
        response = await self.arequest_with_retry(
            endpoint="/update_weights_from_disk",
            payload=dict(model_path=str(path), allow_interrupt=True),
            method="POST",
            max_retries=3,
            timeout=self.config.request_timeout,
            target_addr=addr,
        )
        res = await response.json()
        assert res["success"]
        if "num_paused_requests" in res:
            logger.info(
                f"{res['num_paused_requests']} requests are interrupted "
                f"during updating weights for server {addr}"
            )

    def submit(self, data: Dict[str, Any], workflow: "RolloutWorkflow") -> None:
        try:
            self.input_queue.put_nowait((data, workflow))
        except Full:
            raise RuntimeError("Input queue full. Please increase queue_size.")

    def wait(self, count: int, timeout: float, should_accept: Callable) -> TensorDict:
        tik = time.perf_counter()
        accepted = len(self.result_cache)
        while (
            accepted < count
            and not self.exiting.is_set()
            and time.perf_counter() - tik < timeout
        ):
            try:
                result = self.output_queue.get(timeout=ROLLOUT_POLL_WAIT_TIME)
                if should_accept(result):
                    self.result_cache.append(result)
                    accepted += 1
                else:
                    with self.lock:
                        self.rollout_stat.accepted -= 1
            except Empty:
                time.sleep(ROLLOUT_POLL_WAIT_TIME)
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
        return TensorDict.cat(results, dim=0)

    def rollout(
        self, data: List[Dict[str, Any]], workflow: "RolloutWorkflow"
    ) -> TensorDict:
        """Submit a batch of requests to the inference engine and wait for the results."""
        for item in data:
            self.submit(item, workflow)
        return self.wait(
            count=len(data),
            timeout=self.config.request_timeout,
            should_accept=lambda x: True,
        )
