import asyncio
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from typing import Any, Protocol

import torch.distributed as dist
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.cli_args import InferenceEngineConfig
from areal.api.io_struct import (
    ModelRequest,
    ModelResponse,
    ParamSpec,
    WeightUpdateMeta,
)
from areal.api.workflow_api import RolloutWorkflow
from areal.platforms import current_platform
from areal.utils import logging, name_resolve, names

from .workflow_executor import WorkflowExecutor


class LocalInfBackendProtocol(Protocol):
    """Protocol defining backend-specific operations for local inference engines.

    This protocol abstracts the differences between various local inference engines
    (SGLang, vLLM, etc.) by defining a common interface for:
    - Creating and managing local engine instances
    - Performing async generation
    - Handling weight updates (both disk and distributed)
    - Managing engine lifecycle

    Implementations can raise NotImplementedError for unsupported features.
    """

    def create_engine(self, engine_args: dict[str, Any]) -> Any:
        """Create a local inference engine instance.

        Parameters
        ----------
        engine_args : Dict[str, Any]
            Arguments to pass to the engine constructor

        Returns
        -------
        Any
            The created engine instance
        """
        ...

    async def async_generation(self, engine: Any, req: ModelRequest) -> ModelResponse:
        """Perform async generation using the local engine.

        Parameters
        ----------
        engine : Any
            The engine instance
        req : ModelRequest
            The generation request containing input and parameters

        Returns
        -------
        ModelResponse
            The generated response with tokens, logprobs, and metadata
        """
        ...

    def update_weight_disk(self, engine: Any, model_path: str) -> None:
        """Update weights from disk synchronously.

        Parameters
        ----------
        engine : Any
            The engine instance
        model_path : str
            Path to the model weights on disk
        """
        ...

    def update_weight_xccl(
        self,
        engine: Any,
        meta: WeightUpdateMeta,
        param_specs: list[ParamSpec],
    ) -> None:
        """Update weights from distributed memory via NCCL/XCCL synchronously.

        Parameters
        ----------
        engine : Any
            The engine instance
        meta : WeightUpdateMeta
            Metadata containing communication group info
        param_specs : List[ParamSpec]
            Specifications for parameters to be updated
        """
        ...

    def init_update_weight_group(
        self, engine: Any, meta: WeightUpdateMeta, rank_offset: int
    ) -> None:
        """Initialize weight update communication group synchronously.

        Parameters
        ----------
        engine : Any
            The engine instance
        meta : WeightUpdateMeta
            Metadata containing communication backend configuration
        rank_offset : int
            Rank offset for this engine in the communication group
        """
        ...

    def destroy(self, engine: Any) -> None:
        """Destroy the engine and release resources.

        Parameters
        ----------
        engine : Any
            The engine instance to destroy
        """
        ...


class LocalInfEngine:
    """
    Base implementation for local in-process inference engines.

    This class provides common functionality for running inference engines
    within the same process. Backend-specific behaviors are delegated to
    an injected LocalInfBackendProtocol implementation.

    Uses composition pattern - instantiate directly with a backend rather
    than inheriting from this class.

    Parameters
    ----------
    config : InferenceEngineConfig
        Configuration for the inference engine
    backend : LocalInfBackendProtocol
        Backend implementation providing engine-specific behavior
    """

    def __init__(self, config: InferenceEngineConfig, backend: LocalInfBackendProtocol):
        self.config = config
        self.backend = backend

        self.engine = None
        self.distributed_weight_update_initialized = False
        self._version = 0

        self.lock = Lock()

        self.workflow_executor: WorkflowExecutor

    def initialize(
        self,
        engine_id: str | None = None,
        engine_args: dict[str, Any] | None = None,
        train_data_parallel_size: int | None = None,
    ):
        """Initialize the engine by creating the local inference engine.

        Parameters
        ----------
        engine_id : Optional[str]
            Unique identifier for this engine instance
        engine_args : Optional[Dict[str, Any]]
            Arguments to pass to the backend engine constructor
        train_data_parallel_size : int | None
            Data parallel size of the training engine
        """
        if engine_id is None:
            if dist.is_initialized():
                engine_id = str(dist.get_rank())
            else:
                engine_id = uuid.uuid4().hex
        self.engine_id = engine_id
        self.logger = logging.getLogger(f"[Local Inference Engine Rank {engine_id}]")

        # Create the local engine via backend
        engine_args = engine_args or {}
        self.logger.info(f"Creating local inference engine with args: {engine_args}")
        self.engine = self.backend.create_engine(engine_args)
        self.logger.info("Local inference engine created successfully!")

        # Initialize thread pool for non-blocking weight updates
        self.executor = ThreadPoolExecutor(max_workers=1)

        # Initialize workflow executor
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
        if self.engine is not None:
            self.backend.destroy(self.engine)
            self.engine = None
        self.executor.shutdown()

    def set_version(self, version: int):
        """Set the current weight version."""
        with self.lock:
            self._version = version

    def get_version(self) -> int:
        """Get the current weight version."""
        with self.lock:
            return self._version

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
        if self.engine is None:
            raise RuntimeError(
                "Local inference engine is not initialized, cannot generate."
            )

        # Create a shallow copy of the input request
        # we are going to modify it in-place
        req = req.copy()

        # Validate n_samples
        gconfig = req.gconfig
        if gconfig.n_samples != 1:
            raise ValueError(
                "Local inference engines do not support n_samples > 1. "
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

        # Loop until generation is complete
        stop_reason = None
        while (
            stop_reason not in ["stop", "tool_calls", "length"]
            and len(accumulated_output_tokens) < gconfig.max_new_tokens
        ):
            # Handle rollout interruption
            while self.workflow_executor.paused.is_set():
                await asyncio.sleep(0.5)

            # Call backend async_generation
            response = await self.backend.async_generation(self.engine, req)

            # Extract result
            output_tokens = response.output_tokens
            output_logprobs = response.output_logprobs
            stop_reason = response.stop_reason

            # Update accumulated outputs
            accumulated_output_tokens.extend(output_tokens)
            accumulated_output_logprobs.extend(output_logprobs)
            accumulated_versions.extend([self.get_version()] * len(output_tokens))

            # Update request for next iteration
            req.input_ids += output_tokens
            req.gconfig.max_new_tokens -= len(output_tokens)
            assert req.gconfig.max_new_tokens >= 0, (
                req.gconfig.max_new_tokens,
                len(output_tokens),
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
        assert not self.distributed_weight_update_initialized, (
            "Weight update group already initialized."
        )

        if self.engine is None:
            raise RuntimeError(
                "Local inference engine is not initialized, "
                "cannot init weight update group."
            )

        # Compute rank offset for this engine
        # For local engines, we assume single instance per process
        rank_offset = 1  # Offset by 1 to leave rank 0 for the training engine

        fut = self.executor.submit(
            self._init_weights_update_group_sync, meta, rank_offset
        )

        def callback(fut):
            self.logger.info(
                f"Initialized {current_platform.communication_backend.upper()} group "
                f"for distributed weight update for {meta.nccl_group_name}."
            )
            self.distributed_weight_update_initialized = True

        fut.add_done_callback(callback)

        return fut

    def _init_weights_update_group_sync(self, meta: WeightUpdateMeta, rank_offset: int):
        """Synchronously initialize weight update group in thread pool."""
        self.backend.init_update_weight_group(self.engine, meta, rank_offset)

    def update_weights_from_distributed(
        self, meta: WeightUpdateMeta, param_specs: list[ParamSpec]
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

        if self.engine is None:
            raise RuntimeError(
                "Local inference engine is not initialized, cannot update weights."
            )

        fut = self.executor.submit(
            self._update_weights_from_distributed_sync, meta, param_specs
        )

        return fut

    def _update_weights_from_distributed_sync(
        self, meta: WeightUpdateMeta, param_specs: list[ParamSpec]
    ):
        """Synchronously update weights from distributed memory in thread pool."""
        self.backend.update_weight_xccl(self.engine, meta, param_specs)

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

        if self.engine is None:
            raise RuntimeError(
                "Local inference engine is not initialized, cannot update weights."
            )

        tik = time.perf_counter()

        # Validate experiment and trial names
        if self.config.experiment_name is None or self.config.trial_name is None:
            raise RuntimeError(
                "Experiment and trial names must be set for disk-based weight updates."
            )

        fut = self.executor.submit(self._update_weights_from_disk_sync, meta)

        def callback(fut):
            respond_time = fut.result()
            self.logger.info(
                f"Loading weights from disk done in "
                f"{(time.perf_counter() - tik):.2f}s. "
                f"Respond time: {respond_time:.2f}s."
            )

        fut.add_done_callback(callback)

        return fut

    def _update_weights_from_disk_sync(self, meta: WeightUpdateMeta) -> float:
        """Synchronously update weights from disk in thread pool."""
        # Wait for training engine to signal that weights are ready
        update_name = names.update_weights_from_disk(
            self.config.experiment_name,
            self.config.trial_name,
            meta.model_version,
        )
        save_timestamp = float(name_resolve.wait(update_name, timeout=120))
        load_timestamp = time.time()

        self.logger.info(
            f"Begin update weights from {meta.path}, "
            f"responded in {(load_timestamp - save_timestamp) * 1000:.2f} ms"
        )

        # Update weights from disk via backend
        self.backend.update_weight_disk(self.engine, str(meta.path))

        self.logger.info(
            f"Loading weights done in {(time.time() - load_timestamp) * 1000:.2f} ms"
        )
        self.set_version(meta.model_version)

        return load_timestamp - save_timestamp

    def submit(
        self,
        data: dict[str, Any],
        workflow: RolloutWorkflow | None = None,
        workflow_builder: Callable | None = None,
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

    def wait(self, count: int, timeout: float | None = None) -> dict[str, Any]:
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
        data: list[dict[str, Any]],
        workflow: RolloutWorkflow | None = None,
        workflow_builder: Callable | None = None,
        should_accept: Callable | None = None,
    ) -> dict[str, Any]:
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
        workflow: RolloutWorkflow | None = None,
        workflow_builder: Callable | None = None,
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

    def pause(self):
        """Pause request submission for async rollout.

        Used during evaluation to prevent data over generation.
        """
        return self.workflow_executor.pause()

    def resume(self):
        """Resume request submission for async rollout."""
        return self.workflow_executor.resume()
