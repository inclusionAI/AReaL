import asyncio
import os
import random
import shutil
import time
import uuid
from concurrent.futures import Future, ProcessPoolExecutor
from datetime import datetime
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Union

import aiohttp
import requests
import torch.distributed as dist
import uvloop
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.cli_args import InferenceEngineConfig
from areal.api.engine_api import InferenceEngine, Scheduling
from areal.api.io_struct import (
    ModelRequest,
    ModelResponse,
    WeightUpdateMeta,
)
from areal.api.workflow_api import RolloutWorkflow, WorkflowExecutor
from areal.platforms import current_platform
from areal.utils import logging, name_resolve, names
from areal.utils.http import arequest_with_retry, get_default_connector

RID_CACHE_SIZE = 128


class RemoteSGLangEngine(InferenceEngine):

    def __init__(self, config: InferenceEngineConfig):
        self.config = config

        self.rid_to_address = {}
        # Maintain the addresses for the recent 128 requests
        self.rid_queue = []
        self.addresses = []
        self.server_idx = 0

        self.distributed_weight_update_initialized = False
        self._version = 0

        self.lock = Lock()
        self.workflow_executor = WorkflowExecutor(
            config=config,
            inference_engine=self,
        )

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

    def initialize(
        self,
        engine_id: Optional[str] = None,
        addr: str | List[str] | None = None,
        train_data_parallel_size: int | None = None,
    ):
        if engine_id is None:
            if dist.is_initialized():
                engine_id = str(dist.get_rank())
            else:
                engine_id = uuid.uuid4().hex
        self.engine_id = engine_id
        self.logger = logging.getLogger(f"[SGLang Remote Engine Rank {engine_id}]")

        if addr:
            self.addresses = addr if isinstance(addr, list) else [addr]
        else:
            # When addr is not provided, fallback to reading addrs from env var
            self.addresses = os.getenv("AREAL_LLM_SERVER_ADDRS").split(",")
        if not self.addresses:
            raise RuntimeError(
                "No configured SGLang servers. Please pass in SGLang server addresses by arguments "
                "for `RemoteSGLangEngine.initialize` or environment variable `AREAL_LLM_SERVER_ADDRS`."
            )

        self.logger.info("Waiting for server ready...")
        for addr_ in self.addresses:
            self._wait_for_server(addr_)
        self.server_idx = random.randint(0, len(self.addresses) - 1)
        self.logger.info("Servers are all ready!")
        self.executor = ProcessPoolExecutor(max_workers=1)
        self.lora_init = False
        self.workflow_executor.initialize(
            logger=self.logger, train_data_parallel_size=train_data_parallel_size
        )

    def destroy(self):
        self.workflow_executor.destroy()
        self.executor.shutdown()

    def set_version(self, version):
        with self.lock:
            self._version = version

    def get_version(self):
        with self.lock:
            return self._version

    def choose_server(self) -> str:
        if self.config.schedule_policy == "round_robin":
            server = self.addresses[self.server_idx]
            self.server_idx = (self.server_idx + 1) % len(self.addresses)
            return server
        raise NotImplementedError("Only round-robin scheduling is implemented.")

    async def agenerate(self, req: ModelRequest) -> ModelResponse:
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

        max_new_tokens = min(
            gconfig.max_tokens - len(req.input_ids), gconfig.max_new_tokens
        )
        if max_new_tokens <= 0:
            raise RuntimeError(
                f"max_new_tokens ({max_new_tokens}) is non-positive! "
                f"max_tokens={gconfig.max_tokens}, prompt_len={len(req.input_ids)}, "
                f"max_new_tokens={gconfig.max_new_tokens}."
            )

        sample_params = {
            "top_p": gconfig.top_p,
            "top_k": gconfig.top_k,
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0 if gconfig.greedy else gconfig.temperature,
            "stop_token_ids": stop_token_ids,
            "frequency_penalty": gconfig.frequency_penalty,
        }
        if stop:
            sample_params["stop"] = stop

        payload = {
            "input_ids": req.input_ids.copy(),
            "image_data": req.image_data,  # ImageObject or str
            "sampling_params": sample_params,
            "return_logprob": True,
            "stream": False,
        }
        if self.lora_init:
            # Use the same lora name because we are unable to change
            # the lora_name of an inflight request during weight update.
            # If the lora_name mismatch, there'll be an error.
            payload["lora_path"] = f"lora_1"

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

        # Create a new session because we don't know whether this method
        # is called in the workflow thread or the main thread.
        session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=self.config.request_timeout,
                sock_connect=self.config.request_timeout,
                connect=self.config.request_timeout,
            ),
            read_bufsize=1024 * 1024 * 10,
            connector=get_default_connector(),
        )

        # Deal with rollout interruption
        # "abort" is the stop reason for later v0.4.9.post2 after
        # we call the pause_generation endpoint
        stop_reason = None
        while (
            stop_reason not in ["stop", "tool_calls", "length"]
            and len(accumulated_output_tokens) < gconfig.max_new_tokens
        ):
            # Request is interrupted, wait for some time to avoid interfering
            # with update weights requests
            while self.workflow_executor.paused.is_set():
                await asyncio.sleep(0.5)

            # loop until the generation is complete
            result = await arequest_with_retry(
                session=session,
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
            accumulated_versions.extend([self.get_version()] * len(output_tokens))

            payload["input_ids"] += output_tokens
            sample_params["max_new_tokens"] -= len(output_tokens)

        if stop_reason == "abort":
            # If stop_reason is "abort", the only reason we exit the loop is
            # len(accumulated_output_tokens) >= gconfig.max_new_tokens
            # so the actual reason is length
            stop_reason = "length"
        await session.close()
        latency = time.perf_counter() - start_time

        response = ModelResponse(
            input_tokens=req.input_ids,
            input_images=req.image_data,
            output_tokens=accumulated_output_tokens,
            output_logprobs=accumulated_output_logprobs,
            output_versions=accumulated_versions,
            stop_reason=stop_reason,
            latency=latency,
            ttft=latency,  # Simplified for non-streaming
            tokenizer=req.tokenizer,
            processor=req.processor,
        )
        return response

    def update_weights(self, meta: WeightUpdateMeta):
        for addr in self.addresses:
            res = requests.post(f"http://{addr}/pause_generation")
            res.raise_for_status()

        # The above http request may require some time to be scheduled and executed.
        # The following line waits until all requests are indeed dropped.
        time.sleep(self.config.pause_grace_period)

        tik = time.perf_counter()
        fut = Future()
        if meta.type == current_platform.communication_backend:
            fut = self.executor.submit(
                update_weights_from_distributed,
                meta,
                self.addresses,
                self.config.request_timeout,
                not self.distributed_weight_update_initialized,
            )

            def callback(fut):
                self.logger.info(
                    f"Distributed update weights done in {time.perf_counter() - tik}s"
                )
                self.distributed_weight_update_initialized = True

            fut.add_done_callback(callback)
        elif meta.type == "disk":
            # Update weights from disk
            # Use ProcessPool to bypass python GIL for running async coroutines
            if self.config.experiment_name is None or self.config.trial_name is None:
                raise RuntimeError(
                    f"Experiment and trial names must be set for disk-based weight updates."
                )
            endpoints = ["update_weights_from_disk"]
            payloads = [dict(model_path=str(meta.path), abort_all_requests=True)]
            lora_name = "lora_1"
            if meta.use_lora:
                endpoints = []
                payloads = []
                if self.lora_init:
                    endpoints.append("unload_lora_adapter")
                    payloads.append(dict(lora_name=lora_name))
                else:
                    self.lora_init = True
                endpoints.append("load_lora_adapter")
                payloads.append(dict(lora_name=lora_name, lora_path=str(meta.path)))

            fut = self.executor.submit(
                update_weights_from_disk,
                self.config.experiment_name,
                self.config.trial_name,
                self.get_version(),
                self.addresses,
                meta.path,
                self.config.request_retries,
                self.config.request_timeout,
                endpoints,
                payloads,
            )

            def callback(fut):
                respond_time = fut.result()
                self.logger.info(
                    f"Loading weights from disk done in {(time.perf_counter() - tik):.2f}s. "
                    f"Respond time: {respond_time:.2f}s."
                )
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

    def submit(
        self,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        workflow: Optional[RolloutWorkflow] = None,
        workflow_builder: Optional[Callable] = None,
        should_accept: Callable | None = None,
    ) -> None:
        if isinstance(data, Dict):
            return self.workflow_executor.submit(
                data,
                workflow=workflow,
                workflow_builder=workflow_builder,
                should_accept=should_accept,
            )
        else:
            for item in data:
                self.workflow_executor.submit(
                    item,
                    workflow=workflow,
                    workflow_builder=workflow_builder,
                    should_accept=should_accept,
                )
            return None


    def wait(self, count: int, timeout: float | None = None) -> Dict[str, Any]:
        return self.workflow_executor.wait(count, timeout=timeout)

    def rollout_batch(
        self,
        data: List[Dict[str, Any]],
        workflow: Optional["RolloutWorkflow"] = None,
        workflow_builder: Optional[Callable] = None,
        should_accept: Callable | None = None,
    ) -> Dict[str, Any]:
        return self.workflow_executor.rollout_batch(
            data=data,
            workflow=workflow,
            workflow_builder=workflow_builder,
            should_accept=should_accept,
        )

    def prepare_batch(
        self,
        dataloader: Union[StatefulDataLoader, List[Dict[str, Any]]],
        workflow: Optional[RolloutWorkflow] = None,
        workflow_builder: Optional[Callable] = None,
        should_accept: Callable | None = None,
    ):
        return self.workflow_executor.prepare_batch(
            dataloader=dataloader,
            workflow=workflow,
            workflow_builder=workflow_builder,
            should_accept=should_accept,
        )

    def pause(self):
        """Pause request submission for async rollout. Used during evaluation to prevent data over generation."""
        return self.workflow_executor.pause()

    def resume(self):
        """Resume request submission for async rollout."""
        return self.workflow_executor.resume()

    def get_scheduling_config(self) -> List[Scheduling]:
        # 部署 launcher/sglang_server.py, local_scheduler 注入一个ENGINE_PORTS的端口环境变量,里面有两个端口
        raise NotImplementedError()


def update_weights_from_disk(
    experiment_name,
    trial_name,
    model_version,
    addresses,
    path,
    request_retries,
    request_timeout,
    endpoints,
    payloads,
):
    async def _fn():
        update_name = names.update_weights_from_disk(
            experiment_name, trial_name, model_version
        )
        save_timestamp = float(name_resolve.wait(update_name, timeout=120))
        load_timestamp = datetime.now().timestamp()
        session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=request_timeout,
                sock_connect=request_timeout,
                connect=request_timeout,
            ),
            read_bufsize=1024 * 1024 * 10,
            connector=get_default_connector(),
        )
        for endpoint, payload in zip(endpoints, payloads):
            jobs = [
                arequest_with_retry(
                    addr=addr,
                    session=session,
                    endpoint=f"/{endpoint}",
                    payload=payload,
                    method="POST",
                    max_retries=request_retries,
                    timeout=request_timeout,
                )
                for addr in addresses
            ]
            await asyncio.gather(*jobs)
        await session.close()
        return load_timestamp - save_timestamp

    return uvloop.run(_fn())


def update_weights_from_distributed(
    meta: WeightUpdateMeta,
    addresses: List[str],
    request_timeout,
    init_group: bool,
):
    nccl_param_specs = [
        spec for param_specs in meta.nccl_param_specs for spec in param_specs
    ]

    async def _fn():
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
                        "names": [pspec.name for pspec in nccl_param_specs],
                        "dtypes": [pspec.dtype for pspec in nccl_param_specs],
                        "shapes": [pspec.shape for pspec in nccl_param_specs],
                        "group_name": meta.nccl_group_name,
                        "abort_all_requests": True,
                    },
                    method="POST",
                    max_retries=1,
                    timeout=request_timeout,
                )
                for addr in addresses
            ]
        )

    return uvloop.run(_fn())


async def ainit_weights_update_group(
    addr: str,
    server_idx: int,
    meta: WeightUpdateMeta,
    request_timeout: float,
):
    assert meta.alloc_mode is not None
    if meta.alloc_mode.gen.pp_size != 1:
        raise NotImplementedError(
            "NCCL weight update with PP size > 1 is not implemented yet."
        )
    rank_offset = 1 + server_idx * meta.alloc_mode.gen.tp_size
    payload = {
        "master_address": meta.nccl_master_address,
        "master_port": str(meta.nccl_master_port),
        "rank_offset": rank_offset,
        "world_size": meta.alloc_mode.gen.world_size + 1,
        "backend": current_platform.communication_backend,
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
