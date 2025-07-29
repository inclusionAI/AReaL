import asyncio
import os
import random
import json
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from queue import Empty, Full, Queue
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional
from dataclasses import dataclass, asdict

import aiohttp
import requests
import torch.distributed as dist
import uvloop
from tensordict import TensorDict
from concurrent.futures import ThreadPoolExecutor, as_completed

from arealite.api.cli_args import RemoteHybridInferenceConfig
from arealite.api.engine_api import InferenceEngine, Scheduling
from arealite.dataset.distributed_batch_memory import DistributedBatchMemory
from arealite.api.io_struct import (
    LLMRequest,
    LLMResponse,
    RolloutStat,
    WeightUpdateMeta,
)
from arealite.utils.data import concat_padded_tensors
from arealite.utils.http import arequest_with_retry, get_default_connector
from realhf.base import logging, name_resolve, names, pkg_version, seeding

if TYPE_CHECKING:
    from arealite.api.workflow_api import RolloutWorkflow
logger = logging.getLogger(__name__)

ROLLOUT_POLL_WAIT_TIME = 0.05
RID_CACHE_SIZE = 128


@dataclass
class RemoteHypidInferenceInitConfig:
    main_server_addrs: list[str]  # dp address
    free_addrs: list[str] # 给出每个 ip 的 free_port, 顺序和main_server_addrs一致
    world_size: int     # 总的 world size
    global_ranks: list[int] # 要求和 main_server_addrs 的 idx 一一对应


class RemoteHybridInferenceWorker(InferenceEngine):
    def __init__(self, config: RemoteHybridInferenceConfig):
        config.max_concurrent_rollouts = (
            config.max_concurrent_rollouts or config.consumer_batch_size
        )
        self.config = config
        self.qsize = config.queue_size or config.max_concurrent_rollouts * 16
        self.rid_to_address = {}
        # Maintain the addresses for the recent 128 requests
        self.rid_queue = []
        self.addresses = None  # dp head
        self.server_idx = 0
        self.result_cache = []
        self.rollout_stat = RolloutStat()

        self._version = 0
        logger.info(f"[RemoteHybridInferenceWorker] class __init__ success...")

    def initialize(self, initialize_cfg: RemoteHypidInferenceInitConfig):
        logger.info(f"[RemoteHybridInferenceWorker] begin exec initialize, config: {initialize_cfg}")
        seeding.set_random_seed(1, self.config.experiment_name)
        master_addr_info = initialize_cfg.free_addrs[0]
        master_addr, master_port = master_addr_info.split(":")
        world_size = initialize_cfg.world_size
        seeding.set_random_seed(1, self.config.experiment_name)

        self.addresses = []
        futures = []
        
        with ThreadPoolExecutor(max_workers=len(initialize_cfg.main_server_addrs)) as executor:
            for index, engine_addrs in enumerate(initialize_cfg.main_server_addrs):
                global_rank = initialize_cfg.global_ranks[index]
                server_ip_port = engine_addrs.split(":")
                server_ip = server_ip_port[0]
                server_port = server_ip_port[1]
                if index == 0:
                    self.addresses = [server_ip + ":" + server_port]

                # http body data
                body = dict(self.config.engine_config)
                body["model_path"] = self.config.model_path
                body["storage_path"] = self.config.storage_path
                body["random_seed"] = seeding.get_seed()
                body["engine_config"] = self.config.engine_config
                rank_config = {
                    "master_addr": master_addr,
                    "master_port": master_port,
                    "world_size": world_size,
                    "global_rank": global_rank,
                    "dp_size": self.config.dp_size,
                    "pp_size": self.config.pp_size,
                    "tp_size": self.config.tp_size,
                }
                body["rank_config"] = rank_config
                url = "http://" + initialize_cfg.main_server_addrs[index] + "/initialize"
                logger.info(
                    f"[RemoteHybridInferenceWorker] url: {url}, send hybrid inference initialize config to engine: {body}")
                
                futures.append(executor.submit(
                    requests.post,
                    url,
                    headers={"Content-Type": "application/json"},
                    json=body,
                    timeout=60
                ))

            try:
                for future in as_completed(futures):
                    response = future.result()
                    response.raise_for_status()  # 自动处理 4xx/5xx 状态码
                    result = response.json()
                    logger.info(f"[RemoteHybridInferenceWorker] initialize success, response: {result}")
            except Exception as e:
                logger.error(f"[RemoteHybridInferenceWorker] initialize failed: {str(e)}")
                raise

        self.exiting = threading.Event()
        self.paused = threading.Event()
        self.lock = threading.Lock()
        self.input_queue = Queue(maxsize=self.qsize)
        self.output_queue = Queue(maxsize=self.qsize)

        self.rollout_tasks: Dict[str, asyncio.Task] = {}
        self.executor = ProcessPoolExecutor(max_workers=1)
        self.rollout_thread = threading.Thread(target=self._rollout_thread)
        self.rollout_thread.start()
        logger.info("[RemoteHybridInferenceWorker] initialize exec success...")

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
                    logger.info(f"[RemoteHybridInferenceWorker] Get data from puller: {data}")
                    task = asyncio.create_task(
                        workflow.arun_episode(self, data), name=str(rid)
                    )
                    with self.lock:
                        rollout_tasks[str(rid)] = task
                        self.rollout_stat.submitted += 1
                        self.rollout_stat.running += 1
                        logger.info(
                            f"[RemoteHybridInferenceWorker] Submit rollout rid {rid}. "
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
                        logger.info(
                            f"[RemoteHybridInferenceWorker] Finish rollout {task_rid}. "
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

    async def agenerate(self, req: LLMRequest) -> LLMResponse:
        """Async version of generate using aiohttp."""
        # Prepare request payload
        gconfig = req.gconfig
        stop_token_ids = gconfig.stop_token_ids

        if gconfig.n_samples != 1:
            raise ValueError(
                "RemoteHybridInferenceWorker does not support n_samples > 1. "
                "Please call generate for multiple times with n_samples = 1."
            )
        sample_params = {
            "top_p": gconfig.top_p,
            "top_k": gconfig.top_k,
            "max_new_tokens": gconfig.max_new_tokens,
            "temperature": 0.0 if gconfig.greedy else gconfig.temperature,
            "stop_token_ids": stop_token_ids,
        }

        print(f"gconfig.max_new_tokens: {gconfig.max_new_tokens}", flush=True)
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
                endpoint="/async_generate_sequences",
                payload=payload,
                method="POST",
                max_retries=3,
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

            payload["input_ids"] += result["output_ids"].copy()
            sample_params["max_new_tokens"] -= len(output_tokens)

        latency = time.perf_counter() - start_time

        return LLMResponse(
            input_tokens=req.input_ids,
            output_tokens=accumulated_output_tokens,
            output_logprobs=accumulated_output_logprobs,
            output_versions=accumulated_versions,
            stop_reason=stop_reason,
            latency=latency,
            ttft=latency,  # Simplified for non-streaming
        )

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
        # logger.info(f"max_concurrent_rollouts: {max_concurrent_rollouts}, version: {version}, ofp: {ofp}, self.config.consumer_batch_size: {self.config.consumer_batch_size}, sample_cnt: {sample_cnt}")

        return capacity

    def update_weights(self, meta):
        # executor = ThreadPoolExecutor(max_workers=1)
        # return executor.submit(self._update_weights, meta)
        self._update_weights(meta)
        self._version = self._version + 1
        return True

    def _update_weights(self, meta: WeightUpdateMeta):
        if meta.type == "disk":
            load_timestamp = time.time_ns()
            logger.info(f"[RemoteHybridInferenceWorker] Begin update weights from {meta.path}")

            def update_single_server(addr):
                try:
                    response = requests.post(
                        f"http://{addr}/update_weights_from_disk",
                        json={"model_path": str(meta.path)},
                        timeout=self.config.request_timeout
                    )
                    response.raise_for_status()
                    res = response.json()
                    assert res["success"]
                except Exception as e:
                    logger.error(f"[RemoteHybridInferenceWorker] Failed to update weights on {addr}: {str(e)}")
                    raise

            with ThreadPoolExecutor(max_workers=len(self.addresses)) as executor:
                futures = [executor.submit(update_single_server, addr) for addr in self.addresses]
                for future in futures:
                    future.result()  # Wait for all to complete or raise first exception

            logger.info(
                f"[RemoteHybridInferenceWorker] Loading weights done in {(time.time_ns() - load_timestamp) / 1e6:.2f} ms"
            )
            self.set_version(meta.model_version)
        else:
            load_timestamp = time.time_ns()
            logger.info(f"[RemoteHybridInferenceWorkerer] Begin update weights from {meta.path}")

            def update_single_server(addr):
                try:
                    response = requests.post(
                        f"http://{addr}/update_weights",
                        json={"model_path": str(meta.path)},
                        timeout=self.config.request_timeout
                    )
                    response.raise_for_status()
                    res = response.json()
                    assert res["success"]
                except Exception as e:
                    logger.error(f"[RemoteHybridInferenceWorker] Failed to update weights on {addr}: {str(e)}")
                    raise

            with ThreadPoolExecutor(max_workers=len(self.addresses)) as executor:
                futures = [executor.submit(update_single_server, addr) for addr in self.addresses]
                for future in futures:
                    future.result()  # Wait for all to complete or raise first exception

            logger.info(
                f"[RemoteHybridInferenceWorker] Loading weights done in {(time.time_ns() - load_timestamp) / 1e6:.2f} ms"
            )
            self.set_version(meta.model_version)

    def pause(self):
        self.paused.set()

    def resume(self):
        self.paused.clear()

    def update_weights_from_disk(self, addr, path: str):
        response = requests.post(
            f"http://{addr}/update_weights_from_disk",
            json={"model_path": str(path)},
            timeout=self.config.request_timeout
        )
        response.raise_for_status()
        res = response.json()
        assert res["success"]

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

        logger.info(f"[RemoteHybridInferenceWorker] wait, get all results len: {len(results)}, details: {results}")
        group_size = 1
        if len(results) > 0:
            group_size = int(results[0]["input_ids"].shape[0])
        bs = group_size * len(results)

        padded = concat_padded_tensors(results)
        if isinstance(padded, dict):
            padded = TensorDict(padded, batch_size=[bs])
        print(f"[RemoteHybridInferenceWorker] wait, padded type: {type(padded)}, padded: {padded}")
        return padded

    def rollout(  # only dp head accept this request
        self, data: List[Dict[str, Any]], workflow: "RolloutWorkflow"
    ) -> TensorDict:
        """Submit a batch of requests to the inference engine and wait for the results."""
        for item in data:
            self.submit(item, workflow)
        return self.wait(count=len(data))

    def rollout_distributed_batch(
        self, batch: DistributedBatchMemory, workflow: "RolloutWorkflow"
    ) -> DistributedBatchMemory:
        # batch => list[Dict]
        data = []
        for i in range(len(batch)):
            data.append(batch[i])
        res = self.rollout(data, workflow)  # TensorDict
        # TensorDict=>DistributedBatchMemory
        return DistributedBatchMemory(res.to_dict())

    def get_scheduling_config(self):
        # one dp total resources
        return Scheduling(
            cpu=0,
            gpu=1,
            mem=0,
            type="engine",
        )

    def notify_inference_event(self, event: str, global_step: int) -> None:
        """Handle inference start/end events by sending HTTP notification.
        
        Args:
            event: "start" or "end" 
            global_step: Current global step
        """
        if event not in ["start", "end"]:
            raise ValueError(f"Invalid event type: {event}")
            
        logger.info(f"[RemoteHybridInferenceWorker] Sending inference {event} notification at global_step: {global_step}")
        
        try:
            target_url = f"http://{self.addresses[0]}/inference_events"
            headers = {"Content-Type": "application/json"}
            payload = {
                "event": event,
                "global_step": global_step
            }
            response = requests.post(
                target_url,
                data=json.dumps(payload),
                headers=headers,
                timeout=60
            )
            if response.status_code != 200:
                raise ValueError(
                    f"Failed to send inference event. Status code: {response.status_code}, "
                    f"Response: {response.text}"
                )
        except Exception as e:
            raise ValueError(f"Error sending inference event notification: {e}")
            
        return None
