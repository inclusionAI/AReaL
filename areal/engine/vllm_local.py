import asyncio
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any

from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.cli_args import InferenceEngineConfig
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import (
    ModelRequest,
    ModelResponse,
    ParamSpec,
    WeightUpdateMeta,
)
from areal.api.workflow_api import RolloutWorkflow
from areal.core.local_inf_engine import LocalInfEngine
from areal.platforms import current_platform


class VLLMLocalBackend:
    """vLLM-specific backend implementation for local inference.

    This backend wraps vLLM's native AsyncLLMEngine API for in-process inference.
    """

    def create_engine(self, engine_args: dict[str, Any]) -> Any:
        """Create a local vLLM engine instance.

        Parameters
        ----------
        engine_args : Dict[str, Any]
            Arguments to pass to vLLM AsyncLLMEngine constructor

        Returns
        -------
        Any
            The created vLLM AsyncLLMEngine instance
        """
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        engine_args.pop("host", None)
        engine_args.pop("port", None)
        engine_args.pop("uvicorn_log_level", None)

        engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**engine_args))
        return engine

    async def async_generation(self, engine: Any, req: ModelRequest) -> ModelResponse:
        """Perform async generation using the local vLLM engine.

        Parameters
        ----------
        engine : Any
            The vLLM AsyncLLMEngine instance
        req : ModelRequest
            The generation request containing input and parameters

        Returns
        -------
        ModelResponse
            The generated response with tokens, logprobs, and metadata
        """
        from vllm import SamplingParams

        # Prepare request payload
        gconfig = req.gconfig
        stop_token_ids = gconfig.stop_token_ids

        sampling_params = SamplingParams(
            top_p=gconfig.top_p,
            top_k=gconfig.top_k,
            max_tokens=gconfig.max_new_tokens,
            temperature=0.0 if gconfig.greedy else gconfig.temperature,
            stop_token_ids=stop_token_ids,
            logprobs=0,  # Request logprobs
        )

        # Make request
        start_time = time.perf_counter()

        # Generate unique request ID
        request_id = uuid.uuid4().hex

        # Call vLLM's generate method which returns an async generator
        from vllm.inputs.data import TokensPrompt

        results_generator = engine.generate(
            prompt=TokensPrompt(prompt_token_ids=req.input_ids),
            sampling_params=sampling_params,
            request_id=request_id,
        )

        # Iterate through the generator to get the final result
        final_output = None  # RequestOutput
        async for request_output in results_generator:
            final_output = request_output

        # Parse response
        if final_output is None:
            latency = time.perf_counter() - start_time
            return ModelResponse(
                input_tokens=req.input_ids,
                input_images=req.image_data,
                output_tokens=[],
                output_logprobs=[],
                output_versions=[],
                stop_reason="abort",
                latency=latency,
                ttft=latency,
                tokenizer=req.tokenizer,
                processor=req.processor,
            )

        # Extract first completion output
        assert len(final_output.outputs) == 1
        completion_output = final_output.outputs[0]
        stop_reason = completion_output.finish_reason

        # Extract output tokens from token_ids
        output_tokens = completion_output.token_ids

        # Extract logprobs - vLLM returns logprobs as a list of dicts
        output_logprobs = []
        for token_logprobs, token_id in zip(completion_output.logprobs, output_tokens):
            output_logprobs.append(token_logprobs[token_id].logprob)

        latency = time.perf_counter() - start_time

        return ModelResponse(
            input_tokens=req.input_ids,
            input_images=req.image_data,
            output_tokens=output_tokens,
            output_logprobs=output_logprobs,
            output_versions=[],  # Will be filled by LocalInfEngine
            stop_reason=stop_reason,
            latency=latency,
            ttft=latency,
            tokenizer=req.tokenizer,
            processor=req.processor,
        )

    def update_weight_disk(self, engine: Any, model_path: str) -> None:
        """Update weights from disk synchronously.

        Parameters
        ----------
        engine : Any
            The vLLM AsyncLLMEngine instance
        model_path : str
            Path to the model weights on disk
        """
        loop = asyncio.new_event_loop()
        loop.run_until_complete(
            engine.collective_rpc("areal_injected_update_weight", model_path)
        )
        return None

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
            The vLLM AsyncLLMEngine instance
        meta : WeightUpdateMeta
            Metadata containing communication group info
        param_specs : List[ParamSpec]
            Specifications for parameters to be updated
        """
        loop = asyncio.new_event_loop()
        task = engine.collective_rpc(
            "set_weight_meta",
            args=(
                [pspec.name for pspec in param_specs],
                [pspec.dtype for pspec in param_specs],
                [pspec.shape for pspec in param_specs],
            ),
        )
        loop.run_until_complete(task)
        loop.run_until_complete(
            engine.collective_rpc("areal_injected_update_weight_xccl")
        )
        return None

    def init_update_weight_group(
        self, engine: Any, meta: WeightUpdateMeta, rank_offset: int
    ) -> None:
        """Initialize weight update communication group synchronously.

        Parameters
        ----------
        engine : Any
            The vLLM AsyncLLMEngine instance
        meta : WeightUpdateMeta
            Metadata containing communication backend configuration
        rank_offset : int
            Rank offset for this engine in the communication group
        """
        task = engine.collective_rpc(
            "init_update_weight_group",
            args=(
                meta.nccl_master_address,
                str(meta.nccl_master_port),
                rank_offset,
                meta.alloc_mode.gen.world_size + 1,
                current_platform.communication_backend,
                meta.nccl_group_name,
            ),
        )
        loop = asyncio.new_event_loop()
        loop.run_until_complete(task)
        return None

    def destroy(self, engine: Any) -> None:
        """Destroy the engine and release resources.

        Parameters
        ----------
        engine : Any
            The vLLM AsyncLLMEngine instance to destroy
        """
        # vLLM engines typically don't need explicit cleanup
        # but we include this for consistency with the protocol
        if hasattr(engine, "shutdown"):
            engine.shutdown()

    def pause_generation(self):
        raise NotImplementedError()

    def continue_generation(self):
        raise NotImplementedError()


class LocalvLLMEngine(InferenceEngine):
    """vLLM local inference engine.

    This class delegates all functionality to LocalInfEngine with
    a VLLMLocalBackend implementation. It maintains the same public API.

    Note: vLLM does not support weight updates, so update_weights_from_disk
    and update_weights_from_distributed will raise NotImplementedError.

    Parameters
    ----------
    config : InferenceEngineConfig
        Configuration for the inference engine
    """

    def __init__(self, config: InferenceEngineConfig):
        self.config = config
        # Pure composition - create internal engine with vLLM backend
        self._engine = LocalInfEngine(config, VLLMLocalBackend())

    def configure(self, config):
        self.config = config
        self._engine.configure(config)

    def create_engine(self, engine_args):
        return self._engine.create_engine(engine_args)

    def destroy_engine(self):
        self._engine.destroy_engine()

    def initialize(
        self,
        engine_id: str | None = None,
        train_data_parallel_size: int | None = None,
    ):
        """Initialize the engine by creating the local vLLM engine.

        Parameters
        ----------
        engine_id : Optional[str]
            Unique identifier for this engine instance
        engine_args : Optional[Dict[str, Any]]
            Arguments to pass to vLLM AsyncLLMEngine constructor
        train_data_parallel_size : int | None
            Data parallel size of the training engine
        """
        return self._engine.initialize(engine_id, train_data_parallel_size)

    def destroy(self):
        """Destroy the engine and clean up resources."""
        return self._engine.destroy()

    def set_version(self, version: int):
        """Set the current weight version."""
        return self._engine.set_version(version)

    def get_version(self) -> int:
        """Get the current weight version."""
        return self._engine.get_version()

    async def agenerate(self, req: ModelRequest) -> ModelResponse:
        """Asynchronously generate a response for the given request."""
        return await self._engine.agenerate(req)

    def init_weights_update_group(self, meta: WeightUpdateMeta) -> Future[None]:
        """Initialize the weight update process group.

        Note: Not supported by vLLM.
        """
        return self._engine.init_weights_update_group(meta)

    def update_weights_from_distributed(
        self, meta: WeightUpdateMeta, param_specs: list[ParamSpec]
    ) -> Future[None]:
        """Update weights from distributed memory.

        Note: Not supported by vLLM.
        """
        return self._engine.update_weights_from_distributed(meta, param_specs)

    def update_weights_from_disk(self, meta: WeightUpdateMeta) -> Future[None]:
        """Update weights from disk.

        Note: Not supported by vLLM.
        """
        return self._engine.update_weights_from_disk(meta)

    def submit(
        self,
        data: dict[str, Any],
        workflow: RolloutWorkflow | None = None,
        workflow_builder: Callable | None = None,
        should_accept: Callable | None = None,
    ) -> None:
        """Submit a request to the inference engine."""
        return self._engine.submit(data, workflow, workflow_builder, should_accept)

    def wait(self, count: int, timeout: float | None = None) -> dict[str, Any]:
        """Wait for a specified number of requests to complete."""
        return self._engine.wait(count, timeout)

    def rollout_batch(
        self,
        data: list[dict[str, Any]],
        workflow: RolloutWorkflow | None = None,
        workflow_builder: Callable | None = None,
        should_accept: Callable | None = None,
    ) -> dict[str, Any]:
        """Submit a batch of requests and wait for results."""
        return self._engine.rollout_batch(
            data, workflow, workflow_builder, should_accept
        )

    def prepare_batch(
        self,
        dataloader: StatefulDataLoader,
        workflow: RolloutWorkflow | None = None,
        workflow_builder: Callable | None = None,
        should_accept: Callable | None = None,
    ):
        """Asynchronously submit and wait until a full batch is ready."""
        return self._engine.prepare_batch(
            dataloader, workflow, workflow_builder, should_accept
        )

    def pause(self):
        """Pause request submission for async rollout."""
        return self._engine.pause()

    def resume(self):
        """Resume request submission for async rollout."""
        return self._engine.resume()
