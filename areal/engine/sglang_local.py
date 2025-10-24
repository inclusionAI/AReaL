import time
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


class SGLangLocalBackend:
    """SGLang-specific backend implementation for local inference.

    This backend wraps SGLang's native Engine API for in-process inference.
    """

    def create_engine(self, engine_args: dict[str, Any]) -> Any:
        """Create a local SGLang engine instance.

        Parameters
        ----------
        engine_args : Dict[str, Any]
            Arguments to pass to sglang.Engine constructor

        Returns
        -------
        Any
            The created SGLang Engine instance
        """
        import sglang as sgl

        engine = sgl.Engine(**engine_args)
        return engine

    async def async_generation(self, engine: Any, req: ModelRequest) -> ModelResponse:
        """Perform async generation using the local SGLang engine.

        Parameters
        ----------
        engine : Any
            The SGLang Engine instance
        req : ModelRequest
            The generation request containing input and parameters

        Returns
        -------
        ModelResponse
            The generated response with tokens, logprobs, and metadata
        """
        # Prepare request payload
        gconfig = req.gconfig
        stop_token_ids = gconfig.stop_token_ids

        sampling_params = {
            "top_p": gconfig.top_p,
            "top_k": gconfig.top_k,
            "max_new_tokens": gconfig.max_new_tokens,
            "temperature": 0.0 if gconfig.greedy else gconfig.temperature,
            "stop_token_ids": stop_token_ids,
            "frequency_penalty": gconfig.frequency_penalty,
        }

        if gconfig.stop:
            sampling_params["stop"] = gconfig.stop

        # Make request
        start_time = time.perf_counter()

        # Call SGLang's async_generate method
        outputs = await engine.async_generate(
            input_ids=req.input_ids,
            sampling_params=sampling_params,
            return_logprob=True,
        )

        # Parse response
        meta_info = outputs["meta_info"]
        finish_reason = meta_info["finish_reason"]
        stop_reason = finish_reason["type"]
        stop_message = finish_reason.get("message", "")

        # Handle early abort
        if stop_reason == "abort" and stop_message.startswith("Abort before prefill"):
            latency = time.perf_counter() - start_time
            return ModelResponse(
                input_tokens=req.input_ids,
                input_images=req.image_data,
                output_tokens=[],
                output_logprobs=[],
                output_versions=[],
                stop_reason=stop_reason,
                latency=latency,
                ttft=latency,
                tokenizer=req.tokenizer,
                processor=req.processor,
            )

        # Extract output tokens and logprobs
        output_tokens = [x[1] for x in meta_info["output_token_logprobs"]]
        output_logprobs = [x[0] for x in meta_info["output_token_logprobs"]]

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
            The SGLang Engine instance
        model_path : str
            Path to the model weights on disk
        """
        # Call SGLang's update_weights_from_disk method
        engine.update_weights_from_disk(model_path=model_path)

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
            The SGLang Engine instance
        meta : WeightUpdateMeta
            Metadata containing communication group info
        param_specs : List[ParamSpec]
            Specifications for parameters to be updated
        """
        # Call SGLang's update_weights_from_distributed method
        engine.update_weights_from_distributed(
            names=[pspec.name for pspec in param_specs],
            dtypes=[pspec.dtype for pspec in param_specs],
            shapes=[pspec.shape for pspec in param_specs],
            group_name=meta.nccl_group_name,
        )

    def init_update_weight_group(
        self, engine: Any, meta: WeightUpdateMeta, rank_offset: int
    ) -> None:
        """Initialize weight update communication group synchronously.

        Parameters
        ----------
        engine : Any
            The SGLang Engine instance
        meta : WeightUpdateMeta
            Metadata containing communication backend configuration
        rank_offset : int
            Rank offset for this engine in the communication group
        """
        assert meta.alloc_mode is not None
        if meta.alloc_mode.gen.pp_size != 1:
            raise NotImplementedError(
                "NCCL weight update with PP size > 1 is not implemented yet."
            )

        # Call SGLang's init_weights_update_group method
        engine.init_weights_update_group(
            master_address=meta.nccl_master_address,
            master_port=str(meta.nccl_master_port),
            rank_offset=rank_offset,
            world_size=meta.alloc_mode.gen.world_size + 1,
            backend=current_platform.communication_backend,
            group_name=meta.nccl_group_name,
        )

    def destroy(self, engine: Any) -> None:
        """Destroy the engine and release resources.

        Parameters
        ----------
        engine : Any
            The SGLang Engine instance to destroy
        """
        # SGLang engines typically don't need explicit cleanup
        # but we include this for consistency with the protocol
        if hasattr(engine, "shutdown"):
            engine.shutdown()


class LocalSGLangEngine(InferenceEngine):
    """SGLang local inference engine.

    This class delegates all functionality to LocalInfEngine with
    an SGLangLocalBackend implementation. It maintains the same public API.

    Parameters
    ----------
    config : InferenceEngineConfig
        Configuration for the inference engine
    """

    def __init__(self, config: InferenceEngineConfig):
        self.config = config
        # Pure composition - create internal engine with SGLang backend
        self._engine = LocalInfEngine(config, SGLangLocalBackend())

    def initialize(
        self,
        engine_id: str | None = None,
        engine_args: dict[str, Any] | None = None,
        train_data_parallel_size: int | None = None,
    ):
        """Initialize the engine by creating the local SGLang engine.

        Parameters
        ----------
        engine_id : Optional[str]
            Unique identifier for this engine instance
        engine_args : Optional[Dict[str, Any]]
            Arguments to pass to sglang.Engine constructor
        train_data_parallel_size : int | None
            Data parallel size of the training engine
        """
        return self._engine.initialize(engine_id, engine_args, train_data_parallel_size)

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
        """Initialize the weight update process group."""
        return self._engine.init_weights_update_group(meta)

    def update_weights_from_distributed(
        self, meta: WeightUpdateMeta, param_specs: list[ParamSpec]
    ) -> Future[None]:
        """Update weights from distributed memory."""
        return self._engine.update_weights_from_distributed(meta, param_specs)

    def update_weights_from_disk(self, meta: WeightUpdateMeta) -> Future[None]:
        """Update weights from disk."""
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
