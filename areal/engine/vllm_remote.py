from concurrent.futures import Future
from typing import Any, Callable, Dict, List, Optional

from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.cli_args import InferenceEngineConfig
from areal.api.engine_api import InferenceEngine
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
from areal.core import RemoteInfEngine
from areal.platforms import current_platform


class VLLMBackend:
    """vLLM-specific backend implementation for remote inference."""

    def build_generation_request(
        self, req: ModelRequest, with_lora: bool
    ) -> HttpRequest:
        """Build vLLM generation request."""
        if with_lora:
            raise NotImplementedError("vLLM does not support LoRA training.")
        gconfig = req.gconfig
        stop_token_ids = gconfig.stop_token_ids

        # NOTE: vLLM uses flat payload structure, not nested sampling_params
        payload = {
            "prompt": req.input_ids.copy(),
            "top_p": gconfig.top_p,
            "top_k": gconfig.top_k,
            "max_tokens": gconfig.max_new_tokens,
            "temperature": 0.0 if gconfig.greedy else gconfig.temperature,
            "stop_token_ids": stop_token_ids,
            "return_tokens_as_token_ids": True,
            "logprobs": 0,
            "stream": False,
        }

        return HttpRequest(endpoint="/v1/completions", payload=payload)

    def parse_generation_response(
        self, response: Dict[str, Any]
    ) -> HttpGenerationResult:
        """Parse vLLM generation response."""
        meta_info = response["choices"][0]
        stop_reason = meta_info["finish_reason"]

        # Parse tokens from "token:123" format
        output_tokens = meta_info["logprobs"]["tokens"]
        if stop_reason == "abort" and len(output_tokens) == 0:
            return HttpGenerationResult(
                output_tokens=[],
                output_logprobs=[],
                stop_reason=stop_reason,
            )
        output_tokens = [int(t.split(":")[1]) for t in output_tokens]
        output_logprobs = meta_info["logprobs"]["token_logprobs"]

        return HttpGenerationResult(
            output_tokens=output_tokens,
            output_logprobs=output_logprobs,
            stop_reason=stop_reason,
        )

    def build_disk_weight_update_requests(
        self, meta: WeightUpdateMeta, lora_initialized: bool
    ) -> WeightUpdateRequests:
        """Build vLLM disk weight update requests."""
        # vLLM uses a single endpoint for disk updates
        if lora_initialized:
            raise NotImplementedError("vLLM does not support updating LoRA weights.")
        return WeightUpdateRequests(
            requests=[
                HttpRequest(
                    endpoint="/areal_update_weights",
                    payload={"model_path": str(meta.path)},
                )
            ]
        )

    def build_distributed_weight_update_requests(
        self, meta: WeightUpdateMeta, param_specs: List[ParamSpec]
    ) -> WeightUpdateRequests:
        """Build vLLM distributed weight update requests."""
        # vLLM uses two-step process: set metadata, then update
        return WeightUpdateRequests(
            requests=[
                HttpRequest(
                    endpoint="/areal_set_update_weight_meta",
                    payload={
                        "names": [pspec.name for pspec in param_specs],
                        "dtypes": [pspec.dtype for pspec in param_specs],
                        "shapes": [pspec.shape for pspec in param_specs],
                        "group_name": meta.nccl_group_name,
                    },
                ),
                HttpRequest(
                    endpoint="/areal_update_weights_xccl",
                    payload={},
                ),
            ]
        )

    def build_init_weights_group_request(
        self, addr: str, server_idx: int, meta: WeightUpdateMeta
    ) -> HttpRequest:
        """Build vLLM init weights group request."""
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
        return HttpRequest(endpoint="/areal_init_weights_update_group", payload=payload)

    def get_pause_request(self) -> HttpRequest:
        """Get vLLM pause request."""
        return HttpRequest(endpoint="/areal_pause_generation", payload={})

    def get_resume_request(self) -> HttpRequest:
        """Get vLLM resume request."""
        return HttpRequest(endpoint="/areal_continue_generation", payload={})

    def get_health_check_request(self) -> HttpRequest:
        """Get vLLM health check request."""
        return HttpRequest(endpoint="/health", payload={}, method="GET")


class RemotevLLMEngine(InferenceEngine):
    """vLLM remote inference engine.

    This class delegates all functionality to RemoteInfEngine with
    a VLLMBackend implementation. It maintains the same public API for
    backward compatibility.

    Parameters
    ----------
    config : InferenceEngineConfig
        Configuration for the inference engine
    """

    def __init__(self, config: InferenceEngineConfig):
        self.config = config
        # Pure composition - create internal engine with vLLM backend
        self._engine = RemoteInfEngine(config, VLLMBackend())

    def initialize(
        self,
        engine_id: Optional[str] = None,
        addr: str | List[str] | None = None,
        train_data_parallel_size: int | None = None,
    ):
        """Initialize the engine by discovering and connecting to servers."""
        return self._engine.initialize(engine_id, addr, train_data_parallel_size)

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
        self, meta: WeightUpdateMeta, param_specs: List[ParamSpec]
    ) -> Future[None]:
        """Update weights from distributed memory."""
        return self._engine.update_weights_from_distributed(meta, param_specs)

    def update_weights_from_disk(self, meta: WeightUpdateMeta) -> Future[None]:
        """Update weights from disk."""
        return self._engine.update_weights_from_disk(meta)

    def submit(
        self,
        data: Dict[str, Any],
        workflow: Optional[RolloutWorkflow] = None,
        workflow_builder: Optional[Callable] = None,
        should_accept: Callable | None = None,
    ) -> None:
        """Submit a request to the inference engine."""
        return self._engine.submit(data, workflow, workflow_builder, should_accept)

    def wait(self, count: int, timeout: float | None = None) -> Dict[str, Any]:
        """Wait for a specified number of requests to complete."""
        return self._engine.wait(count, timeout)

    def rollout_batch(
        self,
        data: List[Dict[str, Any]],
        workflow: Optional["RolloutWorkflow"] = None,
        workflow_builder: Optional[Callable] = None,
        should_accept: Callable | None = None,
    ) -> Dict[str, Any]:
        """Submit a batch of requests and wait for results."""
        return self._engine.rollout_batch(
            data, workflow, workflow_builder, should_accept
        )

    def prepare_batch(
        self,
        dataloader: StatefulDataLoader,
        workflow: Optional[RolloutWorkflow] = None,
        workflow_builder: Optional[Callable] = None,
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
