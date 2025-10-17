import asyncio
import os
import random
import shutil
import time
import uuid
from concurrent.futures import Future, ProcessPoolExecutor
from datetime import datetime
from threading import Lock
from typing import Any, Callable, Dict, List, Optional

import aiohttp
import requests
import torch.distributed as dist
import uvloop
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.cli_args import InferenceEngineConfig
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import (
    ModelRequest,
    ModelResponse,
    ParamSpec,
    WeightUpdateMeta,
)
from areal.api.workflow_api import RolloutWorkflow, WorkflowExecutor
from areal.platforms import current_platform
from areal.utils import logging, name_resolve, names
from areal.utils.http import arequest_with_retry, get_default_connector
from areal.utils.launcher import wait_llm_server_addrs

RID_CACHE_SIZE = 128


class RemotevLLMEngine(InferenceEngine):
    """
    A remote inference engine that communicates with vLLM servers to perform model inference.
    This class manages multiple vLLM server instances and routes requests accordingly.
    """

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
        """
        Initialize the engine by waiting for all servers to be ready.
        """
        if engine_id is None:
            if dist.is_initialized():
                engine_id = str(dist.get_rank())
            else:
                engine_id = uuid.uuid4().hex
        self.engine_id = engine_id
        self.logger = logging.getLogger(f"[vLLM Remote Engine Rank {engine_id}]")

        if addr:
            self.addresses = addr if isinstance(addr, list) else [addr]
            self.logger.info(f"Get server addresses from the `addr` argument.")
        else:
            if (
                self.config.experiment_name is not None
                and self.config.trial_name is not None
            ):
                try:
                    self.addresses = wait_llm_server_addrs(
                        experiment_name=self.config.experiment_name,
                        trial_name=self.config.trial_name,
                        timeout=1,
                    )
                    self.logger.info(f"Get server addresses from name_resolve.")
                except (TimeoutError, RuntimeError):
                    # RuntimeError happens when name_resolve is not properly configured.
                    pass
        if not self.addresses and os.getenv("AREAL_LLM_SERVER_ADDRS"):
            # When addr is not provided, fallback to reading addrs from env var
            self.addresses = os.environ["AREAL_LLM_SERVER_ADDRS"].split(",")
            self.logger.info(f"Get server addresses from environment variable.")
        if not self.addresses:
            raise RuntimeError(
                "No configured vLLM servers. Please pass in vLLM server addresses by arguments "
                "for `RemotevLLMEngine.initialize` or environment variable `AREAL_LLM_SERVER_ADDRS`."
            )

        self.logger.info("Waiting for server ready...")
        for addr_ in self.addresses:
            self._wait_for_server(addr_)
        self.server_idx = random.randint(0, len(self.addresses) - 1)
        self.logger.info("Servers are all ready!")
        self.executor = ProcessPoolExecutor(max_workers=1)
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
        """
        Choose a server based on the scheduling policy.

        Returns:
            str: Selected server address.

        Raises:
            NotImplementedError: If schedule policy other than round-robin is used.
        """
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

        if gconfig.n_samples != 1:
            raise ValueError(
                "RemotevLLMEngine does not support n_samples > 1. "
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

        # NOTE: rid should NOT be passed in payload
        payload = {
            "prompt": req.input_ids.copy(),
            "top_p": gconfig.top_p,
            "top_k": gconfig.top_k,
            "max_tokens": max_new_tokens,
            "temperature": 0.0 if gconfig.greedy else gconfig.temperature,
            "stop_token_ids": stop_token_ids,
            "return_tokens_as_token_ids": True,
            "logprobs": 0,
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
                endpoint="/v1/completions",
                payload=payload,
                method="POST",
                max_retries=self.config.request_retries,
                timeout=self.config.request_timeout,
            )

            meta_info = result["choices"][0]
            # Check if generation is complete
            finish_reason = meta_info["finish_reason"]
            stop_reason = finish_reason
            # Parse response
            output_tokens = meta_info["logprobs"]["tokens"]
            output_tokens = [int(t.split(":")[1]) for t in output_tokens]
            output_logprobs = meta_info["logprobs"]["token_logprobs"]

            output_len = len(output_tokens)
            if stop_reason == "abort" and output_len <= 0:
                continue

            # Update accumulated outputs
            accumulated_output_tokens.extend(output_tokens)
            accumulated_output_logprobs.extend(output_logprobs)
            accumulated_versions.extend([self.get_version()] * len(output_tokens))

            payload["prompt"] += output_tokens
            payload["max_tokens"] -= len(output_tokens)

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

    def init_weights_update_group(self, meta: WeightUpdateMeta) -> Future[None]:
        # No need to init group for non-NCCL update
        assert meta.type == current_platform.communication_backend
        assert (
            not self.distributed_weight_update_initialized
        ), "Weight update group already initialized."

        fut = self.executor.submit(
            init_weights_update_group_remote,
            meta,
            self.addresses,
            self.config.request_timeout,
        )

        def callback(fut):
            self.logger.info(
                f"Initialized XCCL group for distributed weight update for {meta.nccl_group_name}."
            )
            self.distributed_weight_update_initialized = True

        fut.add_done_callback(callback)

        return fut

    def update_weights_from_distributed(
        self, meta: WeightUpdateMeta, param_specs: List[ParamSpec]
    ) -> Future[None]:
        assert meta.type == current_platform.communication_backend

        fut = self.executor.submit(
            update_weights_from_distributed,
            meta,
            param_specs,
            self.addresses,
            self.config.request_timeout,
        )

        return fut

    def update_weights_from_disk(self, meta: WeightUpdateMeta) -> Future[None]:
        assert meta.type == "disk"

        tik = time.perf_counter()

        # Use ProcessPool to bypass python GIL for running async coroutines
        if self.config.experiment_name is None or self.config.trial_name is None:
            raise RuntimeError(
                f"Experiment and trial names must be set for disk-based weight updates."
            )
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
            respond_time = fut.result()
            self.logger.info(
                f"Loading weights from disk done in {(time.perf_counter() - tik):.2f}s. "
                f"Respond time: {respond_time:.2f}s."
            )
            shutil.rmtree(meta.path, ignore_errors=True)

        fut.add_done_callback(callback)

        return fut

    def submit(
        self,
        data: Dict[str, Any],
        workflow: Optional[RolloutWorkflow] = None,
        workflow_builder: Optional[Callable] = None,
        should_accept: Callable | None = None,
    ) -> None:
        return self.workflow_executor.submit(
            data,
            workflow=workflow,
            workflow_builder=workflow_builder,
            should_accept=should_accept,
        )

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
        dataloader: StatefulDataLoader,
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

    def pause_generation(self):
        """Pause the generation of inference engine.

        Used during updating weights from distributed or disk.
        """
        for addr in self.addresses:
            res = requests.post(f"http://{addr}/areal_pause_generation")
            res.raise_for_status()

        # The above http request may require some time to be scheduled and executed.
        # The following line waits until all requests are indeed dropped.
        time.sleep(self.config.pause_grace_period)

    def continue_generation(self):
        """Continue the generation of inference engine."""
        for addr in self.addresses:
            res = requests.post(f"http://{addr}/areal_continue_generation")
            res.raise_for_status()

    def pause(self):
        """Pause request submission for async rollout. Used during evaluation to prevent data over generation."""
        return self.workflow_executor.pause()

    def resume(self):
        """Resume request submission for async rollout."""
        return self.workflow_executor.resume()


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
        jobs = [
            arequest_with_retry(
                addr=addr,
                session=session,
                endpoint="/areal_update_weights",
                payload=dict(model_path=str(path)),
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


def init_weights_update_group_remote(
    meta: WeightUpdateMeta,
    addresses: List[str],
    request_timeout,
):
    async def _fn():
        await asyncio.gather(
            *[
                ainit_weights_update_group(addr, i, meta, request_timeout)
                for i, addr in enumerate(addresses)
            ]
        )

    return uvloop.run(_fn())


def update_weights_from_distributed(
    meta: WeightUpdateMeta,
    param_specs: List[ParamSpec],
    addresses: List[str],
    request_timeout,
):
    async def _fn():
        await asyncio.gather(
            *[
                arequest_with_retry(
                    addr=addr,
                    endpoint="/areal_set_update_weight_meta",
                    payload={
                        "names": [pspec.name for pspec in param_specs],
                        "dtypes": [pspec.dtype for pspec in param_specs],
                        "shapes": [pspec.shape for pspec in param_specs],
                        "group_name": meta.nccl_group_name,
                    },
                    method="POST",
                    max_retries=1,
                    timeout=request_timeout,
                )
                for addr in addresses
            ]
        )
        await asyncio.gather(
            *[
                arequest_with_retry(
                    addr=addr,
                    endpoint="/areal_update_weights_xccl",
                    payload={},
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
        endpoint="/areal_init_weights_update_group",
        payload=payload,
        method="POST",
        max_retries=1,
        timeout=request_timeout,
    )
    assert res["success"]
