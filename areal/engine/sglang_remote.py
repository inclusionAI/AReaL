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
from areal.core.remote_inf_engine import RemoteInfEngine
from areal.platforms import current_platform


class SGLangBackend:
    """SGLang-specific backend implementation for remote inference."""

    def build_generation_request(
        self, req: ModelRequest, with_lora: bool
    ) -> HttpRequest:
        """Build SGLang generation request."""
        gconfig = req.gconfig
        stop_token_ids = gconfig.stop_token_ids
        stop = gconfig.stop

        sample_params = {
            "top_p": gconfig.top_p,
            "top_k": gconfig.top_k,
            "max_new_tokens": gconfig.max_new_tokens,
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

        # Add LoRA if initialized
        if with_lora:
            payload["lora_path"] = "lora_1"

        return HttpRequest(endpoint="/generate", payload=payload)

    def parse_generation_response(
        self, response: Dict[str, Any]
    ) -> HttpGenerationResult:
        """Parse SGLang generation response."""
        meta_info = response["meta_info"]
        finish_reason = meta_info["finish_reason"]
        stop_reason = finish_reason["type"]
        stop_message = finish_reason.get("message", "")
        if stop_reason == "abort" and stop_message.startswith("Abort before prefill"):
            return HttpGenerationResult(
                output_tokens=[],
                output_logprobs=[],
                stop_reason=stop_reason,
            )

        output_tokens = [x[1] for x in meta_info["output_token_logprobs"]]
        output_logprobs = [x[0] for x in meta_info["output_token_logprobs"]]

        return HttpGenerationResult(
            output_tokens=output_tokens,
            output_logprobs=output_logprobs,
            stop_reason=stop_reason,
        )

    def build_disk_weight_update_requests(
        self, meta: WeightUpdateMeta, lora_initialized: bool
    ) -> WeightUpdateRequests:
        """Build SGLang disk weight update requests."""
        lora_name = "lora_1"

        if meta.use_lora:
            # LoRA workflow
            requests = []
            if lora_initialized:
                # Unload existing LoRA
                requests.append(
                    HttpRequest(
                        endpoint="/unload_lora_adapter",
                        payload={"lora_name": lora_name},
                    )
                )
            # Load new LoRA
            requests.append(
                HttpRequest(
                    endpoint="/load_lora_adapter",
                    payload={"lora_name": lora_name, "lora_path": str(meta.path)},
                )
            )
            return WeightUpdateRequests(requests=requests)
        else:
            # Full model update
            return WeightUpdateRequests(
                requests=[
                    HttpRequest(
                        endpoint="/update_weights_from_disk",
                        payload={
                            "model_path": str(meta.path),
                            "abort_all_requests": True,
                        },
                    )
                ]
            )

    def build_distributed_weight_update_requests(
        self, meta: WeightUpdateMeta, param_specs: List[ParamSpec]
    ) -> WeightUpdateRequests:
        """Build SGLang distributed weight update requests."""
        return WeightUpdateRequests(
            requests=[
                HttpRequest(
                    endpoint="/update_weights_from_distributed",
                    payload={
                        "names": [pspec.name for pspec in param_specs],
                        "dtypes": [pspec.dtype for pspec in param_specs],
                        "shapes": [pspec.shape for pspec in param_specs],
                        "group_name": meta.nccl_group_name,
                        "abort_all_requests": True,
                    },
                )
            ]
        )

    def build_init_weights_group_request(
        self, addr: str, server_idx: int, meta: WeightUpdateMeta
    ) -> HttpRequest:
        """Build SGLang init weights group request."""
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
        return HttpRequest(endpoint="/init_weights_update_group", payload=payload)

    def get_pause_request(self) -> HttpRequest:
        """Get SGLang pause request."""
        return HttpRequest(endpoint="/pause_generation", payload={})

    def get_resume_request(self) -> HttpRequest:
        """Get SGLang resume request."""
        return HttpRequest(endpoint="/continue_generation", payload={})

    def get_health_check_request(self) -> HttpRequest:
        """Get SGLang health check request."""
        return HttpRequest(endpoint="/health", payload={}, method="GET")


class RemoteSGLangEngine(InferenceEngine):
    """SGLang remote inference engine.

    This class delegates all functionality to RemoteInfEngine with
    an SGLangBackend implementation. It maintains the same public API.

    Parameters
    ----------
    config : InferenceEngineConfig
        Configuration for the inference engine
    """

    def __init__(self, config: InferenceEngineConfig):
        # Pure composition - create internal engine with SGLang backend
        self._engine = RemoteInfEngine(config, SGLangBackend())

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
        return self._engine.pause()

    def resume(self):
        return self._engine.resume()

    def pause_generation(self):
        return self._engine.pause_generation()

    def resume_generation(self):
        return self._engine.resume_generation()
