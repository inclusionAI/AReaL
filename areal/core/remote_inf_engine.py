import asyncio
import os
import random
import shutil
import time
import uuid
from concurrent.futures import Future, ProcessPoolExecutor
from datetime import datetime
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Protocol

import aiohttp
import requests
import torch.distributed as dist
import uvloop
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.cli_args import InferenceEngineConfig
from areal.api.io_struct import (
    HttpGenerationResult,
    HttpRequest,
    ModelRequest,
    ModelResponse,
    ParamSpec,
    WeightUpdateMeta,
    WeightUpdateRequests,
)
from areal.api.workflow_api import RolloutWorkflow
from areal.platforms import current_platform
from areal.utils import logging, name_resolve, names
from areal.utils.http import arequest_with_retry, get_default_connector
from areal.utils.launcher import wait_llm_server_addrs

from .workflow_executor import WorkflowExecutor

RID_CACHE_SIZE = 128


class RemoteInfBackendProtocol(Protocol):
    """Protocol defining backend-specific operations for remote inference engines.

    This protocol abstracts the differences between various remote inference servers
    (SGLang, vLLM, etc.) by defining a common interface for:
    - Building HTTP requests with backend-specific formats
    - Parsing backend-specific responses
    - Handling weight updates
    - Managing control flow (pause/resume)
    - Supporting optional features (LoRA)

    Implementations can raise NotImplementedError for unsupported features.
    """

    def build_generation_request(
        self, req: ModelRequest, with_lora: bool
    ) -> HttpRequest:
        """Build HTTP request for text generation.

        Parameters
        ----------
        req : ModelRequest
            The generation request containing input and parameters
        with_lora : bool
            Whether to specify a LoRA to use

        Returns
        -------
        HttpRequest
            The HTTP request with endpoint and payload
        """
        ...

    def parse_generation_response(
        self, response: Dict[str, Any]
    ) -> HttpGenerationResult:
        """Parse generation response into standard format.

        Parameters
        ----------
        response : Dict[str, Any]
            The raw JSON response from the server

        Returns
        -------
        HttpGenerationResult
            Parsed result with tokens, logprobs, and stop reason
        """
        ...

    def build_disk_weight_update_requests(
        self, meta: WeightUpdateMeta, lora_initialized: bool
    ) -> WeightUpdateRequests:
        """Build requests for loading weights from disk.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Metadata containing path and configuration
        lora_initialized : bool
            Whether LoRA has been initialized in the server. If so, we need to unload the previous LoRA before uploading a new one.

        Returns
        -------
        WeightUpdateRequests
            Collection of HTTP requests (may be multiple for LoRA workflows)
        """
        ...

    def build_distributed_weight_update_requests(
        self, meta: WeightUpdateMeta, param_specs: List[ParamSpec]
    ) -> WeightUpdateRequests:
        """Build requests for distributed weight update via NCCL/XCCL.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Metadata containing communication group info
        param_specs : List[ParamSpec]
            Specifications for parameters to be updated

        Returns
        -------
        WeightUpdateRequests
            Collection of HTTP requests for distributed update
        """
        ...

    def build_init_weights_group_request(
        self, addr: str, server_idx: int, meta: WeightUpdateMeta
    ) -> HttpRequest:
        """Build request to initialize weight update communication group.

        Parameters
        ----------
        addr : str
            Server address
        server_idx : int
            Index of this server in the server list
        meta : WeightUpdateMeta
            Metadata containing communication backend configuration

        Returns
        -------
        HttpRequest
            The HTTP request to initialize the group
        """
        ...

    def get_pause_request(self) -> HttpRequest:
        """Get request to pause generation.

        Returns
        -------
        HttpRequest
            The HTTP request to pause generation

        Raises
        ------
        NotImplementedError
            If pause is not supported by this backend
        """
        ...

    def get_resume_request(self) -> HttpRequest:
        """Get request to resume generation.

        Returns
        -------
        HttpRequest
            The HTTP request to resume generation

        Raises
        ------
        NotImplementedError
            If resume is not supported by this backend
        """
        ...

    def get_health_check_request(self) -> HttpRequest:
        """Get the health check request.

        Returns
        -------
        HttpRequest
            The HTTP request for health checks
        """
        ...


class RemoteInfEngine:
    """
    Base implementation for HTTP-based remote inference engines.

    This class provides common functionality for communicating with remote
    inference servers via HTTP REST APIs. Backend-specific behaviors are
    delegated to an injected RemoteInfBackendProtocol implementation.

    Uses composition pattern - instantiate directly with a backend rather
    than inheriting from this class.

    Parameters
    ----------
    config : InferenceEngineConfig
        Configuration for the inference engine
    backend : RemoteInfBackendProtocol
        Backend implementation providing server-specific behavior
    """

    def __init__(
        self, config: InferenceEngineConfig, backend: RemoteInfBackendProtocol
    ):
        self.config = config
        self.backend = backend

        self.rid_to_address = {}
        # Maintain the addresses for the recent 128 requests
        self.rid_queue = []
        self.addresses = []
        self.server_idx = 0

        self.distributed_weight_update_initialized = False
        self._version = 0

        self.lock = Lock()

        self.lora_initialized = False

        self.workflow_executor: WorkflowExecutor

    def _wait_for_server(self, address):
        """Wait for a server to become healthy."""
        base_url = f"http://{address}"
        tik = time.time()
        while time.time() - tik < self.config.setup_timeout:
            if self.check_health(base_url):
                return
            time.sleep(1)
        raise RuntimeError("server launch failed")

    def check_health(self, base_url):
        """Check if server is healthy."""
        try:
            health_req = self.backend.get_health_check_request()
            url = f"{base_url}{health_req.endpoint}"
            response = requests.request(
                health_req.method, url, json=health_req.payload, timeout=30
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def initialize(
        self,
        engine_id: Optional[str] = None,
        addr: str | List[str] | None = None,
        train_data_parallel_size: int | None = None,
    ):
        """Initialize the engine by discovering and connecting to servers.

        Parameters
        ----------
        engine_id : Optional[str]
            Unique identifier for this engine instance
        addr : str | List[str] | None
            Server address(es) to connect to. If None, will auto-discover.
        train_data_parallel_size : int | None
            Data parallel size of the training engine
        """
        if engine_id is None:
            if dist.is_initialized():
                engine_id = str(dist.get_rank())
            else:
                engine_id = uuid.uuid4().hex
        self.engine_id = engine_id
        self.logger = logging.getLogger(f"[Remote Inference Engine Rank {engine_id}]")

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
                "No configured inference servers. Please pass in server addresses by arguments "
                "for `initialize` or environment variable `AREAL_LLM_SERVER_ADDRS`."
            )

        self.logger.info("Waiting for server ready...")
        for addr_ in self.addresses:
            self._wait_for_server(addr_)
        self.server_idx = random.randint(0, len(self.addresses) - 1)
        self.logger.info("Servers are all ready!")
        self.executor = ProcessPoolExecutor(max_workers=1)

        self.workflow_executor = WorkflowExecutor(
            config=self.config,
            inference_engine=self,
        )
        self.workflow_executor.initialize(
            logger=self.logger, train_data_parallel_size=train_data_parallel_size
        )

    def destroy(self):
        """Destroy the engine and clean up resources."""
        self.workflow_executor.destroy()
        self.executor.shutdown()

    def set_version(self, version):
        """Set the current weight version."""
        with self.lock:
            self._version = version

    def get_version(self):
        """Get the current weight version."""
        with self.lock:
            return self._version

    def choose_server(self) -> str:
        """Choose a server based on the scheduling policy.

        Returns
        -------
        str
            Selected server address

        Raises
        ------
        NotImplementedError
            If schedule policy other than round-robin is used
        """
        if self.config.schedule_policy == "round_robin":
            server = self.addresses[self.server_idx]
            self.server_idx = (self.server_idx + 1) % len(self.addresses)
            return server
        raise NotImplementedError("Only round-robin scheduling is implemented.")

    async def agenerate(self, req: ModelRequest) -> ModelResponse:
        """Asynchronously generate a response for the given request.

        Parameters
        ----------
        req : ModelRequest
            The model request containing input data and generation parameters

        Returns
        -------
        ModelResponse
            The generated response from the model
        """
        # Create a shallow copy of the input request
        # we are going to modify it in-place
        req = req.copy()

        # Validate n_samples
        gconfig = req.gconfig
        if gconfig.n_samples != 1:
            raise ValueError(
                "Inference engines do not support n_samples > 1. "
                "Please call generate multiple times with n_samples = 1."
            )

        # Validate max_new_tokens
        max_new_tokens = min(
            gconfig.max_tokens - len(req.input_ids), gconfig.max_new_tokens
        )
        if max_new_tokens <= 0:
            raise RuntimeError(
                f"max_new_tokens ({max_new_tokens}) is non-positive! "
                f"max_tokens={gconfig.max_tokens}, prompt_len={len(req.input_ids)}, "
                f"max_new_tokens={gconfig.max_new_tokens}."
            )

        # Update max_new_tokens in request
        req.gconfig.max_new_tokens = max_new_tokens

        # Make request
        start_time = time.perf_counter()
        accumulated_output_tokens = []
        accumulated_output_logprobs = []
        accumulated_versions = []

        # A single "rid" shares the same server to allow KV cache reuse
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
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=self.config.request_timeout,
                sock_connect=self.config.request_timeout,
                connect=self.config.request_timeout,
            ),
            read_bufsize=1024 * 1024 * 10,
            connector=get_default_connector(),
        ) as session:

            # Deal with rollout interruption
            stop_reason = None
            while (
                stop_reason not in ["stop", "tool_calls", "length"]
                and len(accumulated_output_tokens) < gconfig.max_new_tokens
            ):
                # Request is interrupted, wait for some time to avoid interfering
                # with update weights requests
                while self.workflow_executor.paused.is_set():
                    await asyncio.sleep(0.5)

                # Build request using backend
                http_req = self.backend.build_generation_request(
                    req, self.lora_initialized
                )

                # Loop until the generation is complete
                result = await arequest_with_retry(
                    session=session,
                    addr=server_addr,
                    endpoint=http_req.endpoint,
                    payload=http_req.payload,
                    method=http_req.method,
                    max_retries=self.config.request_retries,
                    timeout=self.config.request_timeout,
                )

                # Parse response using backend
                gen_result = self.backend.parse_generation_response(result)
                stop_reason = gen_result.stop_reason

                # Update accumulated outputs
                accumulated_output_tokens.extend(gen_result.output_tokens)
                accumulated_output_logprobs.extend(gen_result.output_logprobs)
                accumulated_versions.extend(
                    [self.get_version()] * len(gen_result.output_tokens)
                )

                # Update request for next iteration
                req.input_ids += gen_result.output_tokens
                req.gconfig.max_new_tokens -= len(gen_result.output_tokens)
                assert req.gconfig.max_new_tokens >= 0, (
                    req.gconfig.max_new_tokens,
                    len(gen_result.output_tokens),
                    len(req.input_ids),
                )

            # Final abort handling
            if stop_reason == "abort":
                # If stop_reason is "abort", the only reason we exit the loop is
                # len(accumulated_output_tokens) >= gconfig.max_new_tokens
                # so the actual reason is length
                stop_reason = "length"

        latency = time.perf_counter() - start_time

        response = ModelResponse(
            input_tokens=req.input_ids[
                : len(req.input_ids) - len(accumulated_output_tokens)
            ],
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
        """Initialize the weight update process group for distributed weight updates.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Metadata containing information about the weight update

        Returns
        -------
        Future[None]
            A future object representing the asynchronous initialization operation
        """
        assert meta.type == current_platform.communication_backend
        assert (
            not self.distributed_weight_update_initialized
        ), "Weight update group already initialized."

        fut = self.executor.submit(
            _init_weights_update_group_remote,
            self.backend,
            meta,
            self.addresses,
            self.config.request_timeout,
        )

        def callback(fut):
            self.logger.info(
                f"Initialized {current_platform.communication_backend.upper()} group "
                f"for distributed weight update for {meta.nccl_group_name}."
            )
            self.distributed_weight_update_initialized = True

        fut.add_done_callback(callback)

        return fut

    def update_weights_from_distributed(
        self, meta: WeightUpdateMeta, param_specs: List[ParamSpec]
    ) -> Future[None]:
        """Update weights in the inference engine from distributed memory.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Metadata containing information about the weight update
        param_specs : List[ParamSpec]
            A list of parameter specifications for the weights to be updated

        Returns
        -------
        Future[None]
            A future object representing the asynchronous weight update operation
        """
        assert meta.type == current_platform.communication_backend

        fut = self.executor.submit(
            _update_weights_from_distributed,
            self.backend,
            meta,
            param_specs,
            self.addresses,
            self.config.request_timeout,
        )

        return fut

    def update_weights_from_disk(self, meta: WeightUpdateMeta) -> Future[None]:
        """Update weights in the inference engine from disk.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Metadata containing information about the weight update

        Returns
        -------
        Future[None]
            A future object representing the asynchronous weight update operation
        """
        assert meta.type == "disk"

        tik = time.perf_counter()

        # Use ProcessPool to bypass python GIL for running async coroutines
        if self.config.experiment_name is None or self.config.trial_name is None:
            raise RuntimeError(
                f"Experiment and trial names must be set for disk-based weight updates."
            )

        fut = self.executor.submit(
            _update_weights_from_disk,
            self.backend,
            self.lora_initialized,
            self.config.experiment_name,
            self.config.trial_name,
            self.get_version(),
            self.addresses,
            meta,
            self.config.request_retries,
            self.config.request_timeout,
        )

        def callback(fut):
            respond_time = fut.result()
            self.logger.info(
                f"Loading weights from disk done in {(time.perf_counter() - tik):.2f}s. "
                f"Respond time: {respond_time:.2f}s."
            )
            # Update LoRA state if this was a LoRA update
            if meta.use_lora:
                self.lora_initialized = True
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
        """Submit a request to the inference engine and return immediately.

        Parameters
        ----------
        data : Dict[str, Any]
            The input data for rollout
        workflow : RolloutWorkflow, optional
            The workflow instance to run
        workflow_builder : Callable, optional
            A builder to create a workflow instance
        should_accept : Callable, optional
            A function to decide whether to accept a trajectory
        """
        return self.workflow_executor.submit(
            data,
            workflow=workflow,
            workflow_builder=workflow_builder,
            should_accept=should_accept,
        )

    def wait(self, count: int, timeout: float | None = None) -> Dict[str, Any]:
        """Wait for a specified number of requests to complete.

        Parameters
        ----------
        count : int
            The number of accepted trajectories to wait for
        timeout : float, optional
            Timeout in seconds

        Returns
        -------
        Dict[str, Any]
            A concatenated batch of trajectories
        """
        return self.workflow_executor.wait(count, timeout=timeout)

    def rollout_batch(
        self,
        data: List[Dict[str, Any]],
        workflow: Optional["RolloutWorkflow"] = None,
        workflow_builder: Optional[Callable] = None,
        should_accept: Callable | None = None,
    ) -> Dict[str, Any]:
        """Submit a batch of requests and wait for results.

        Parameters
        ----------
        data : List[Dict[str, Any]]
            A list of input data dictionaries for rollout
        workflow : RolloutWorkflow, optional
            The workflow instance to run
        workflow_builder : Callable, optional
            A builder to create a workflow instance
        should_accept : Callable, optional
            A function to decide whether to accept a trajectory

        Returns
        -------
        Dict[str, Any]
            A concatenated batch of trajectory results
        """
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
        """Asynchronously submit and wait until a full batch is ready.

        Parameters
        ----------
        dataloader : StatefulDataLoader
            The data loader to pull data from
        workflow : RolloutWorkflow, optional
            The workflow instance to run
        workflow_builder : Callable, optional
            A builder to create a workflow instance
        should_accept : Callable, optional
            A function to decide whether to accept a trajectory

        Returns
        -------
        Dict[str, Any]
            A full batch of trajectory results
        """
        return self.workflow_executor.prepare_batch(
            dataloader=dataloader,
            workflow=workflow,
            workflow_builder=workflow_builder,
            should_accept=should_accept,
        )

    def pause_generation(self):
        """Pause request submission for async rollout."""
        try:
            pause_req = self.backend.get_pause_request()
            for addr in self.addresses:
                res = requests.post(
                    f"http://{addr}{pause_req.endpoint}",
                    json=pause_req.payload,
                )
                res.raise_for_status()
        except NotImplementedError:
            self.logger.warning("Backend does not support pause operation")

        # The above http request may require some time to be scheduled and executed.
        # The following line waits until all requests are indeed dropped.
        time.sleep(self.config.pause_grace_period)

    def continue_generation(self):
        """Resume request submission for async rollout."""
        try:
            resume_req = self.backend.get_resume_request()
            for addr in self.addresses:
                res = requests.post(
                    f"http://{addr}{resume_req.endpoint}",
                    json=resume_req.payload,
                )
                res.raise_for_status()
        except NotImplementedError:
            self.logger.warning("Backend does not support resume operation")

    def pause(self):
        """Pause request submission for async rollout. Used during evaluation to prevent data over generation."""
        return self.workflow_executor.pause()

    def resume(self):
        """Resume request submission for async rollout."""
        return self.workflow_executor.resume()


# Helper functions that run in ProcessPoolExecutor


def _update_weights_from_disk(
    backend: RemoteInfBackendProtocol,
    lora_initialized: bool,
    experiment_name: str,
    trial_name: str,
    model_version: int,
    addresses: List[str],
    meta: WeightUpdateMeta,
    request_retries: int,
    request_timeout: float,
):
    """Helper to update weights from disk in a separate process."""

    async def _fn():
        update_name = names.update_weights_from_disk(
            experiment_name, trial_name, model_version
        )
        save_timestamp = float(name_resolve.wait(update_name, timeout=120))
        load_timestamp = datetime.now().timestamp()

        # Get requests from backend
        weight_reqs = backend.build_disk_weight_update_requests(meta, lora_initialized)

        # Execute all requests
        for http_req in weight_reqs.requests:
            jobs = [
                arequest_with_retry(
                    addr=addr,
                    endpoint=http_req.endpoint,
                    payload=http_req.payload,
                    method=http_req.method,
                    max_retries=request_retries,
                    timeout=request_timeout,
                )
                for addr in addresses
            ]
            await asyncio.gather(*jobs)

        return load_timestamp - save_timestamp

    return uvloop.run(_fn())


def _init_weights_update_group_remote(
    backend: RemoteInfBackendProtocol,
    meta: WeightUpdateMeta,
    addresses: List[str],
    request_timeout: float,
):
    """Helper to initialize weight update group in a separate process."""

    async def _fn():
        jobs = []
        for i, addr in enumerate(addresses):
            http_req = backend.build_init_weights_group_request(addr, i, meta)
            jobs.append(
                arequest_with_retry(
                    addr=addr,
                    endpoint=http_req.endpoint,
                    payload=http_req.payload,
                    method=http_req.method,
                    max_retries=1,
                    timeout=request_timeout,
                )
            )
        await asyncio.gather(*jobs)

    return uvloop.run(_fn())


def _update_weights_from_distributed(
    backend: RemoteInfBackendProtocol,
    meta: WeightUpdateMeta,
    param_specs: List[ParamSpec],
    addresses: List[str],
    request_timeout: float,
):
    """Helper to update weights from distributed memory in a separate process."""

    async def _fn():
        # Get requests from backend
        weight_reqs = backend.build_distributed_weight_update_requests(
            meta, param_specs
        )

        # Execute all requests sequentially (they may have dependencies)
        for http_req in weight_reqs.requests:
            jobs = [
                arequest_with_retry(
                    addr=addr,
                    endpoint=http_req.endpoint,
                    payload=http_req.payload,
                    method=http_req.method,
                    max_retries=1,
                    timeout=request_timeout,
                )
                for addr in addresses
            ]
            await asyncio.gather(*jobs)

    return uvloop.run(_fn())
