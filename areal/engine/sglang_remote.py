# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
import uuid
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any

import numpy as np
import pybase64
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api import (
    InferenceEngine,
    LocalInfServerInfo,
    ModelRequest,
    ModelResponse,
    ParamSpec,
    Scheduler,
    WeightUpdateMeta,
    WorkflowLike,
)
from areal.api.cli_args import InferenceEngineConfig, PerfTracerConfig, SGLangConfig
from areal.api.io_struct import (
    HttpGenerationResult,
    HttpRequest,
    WeightUpdateRequests,
    get_versioned_lora_name,
)
from areal.infra import RemoteInfEngine, RolloutController, WorkflowExecutor
from areal.infra.platforms import current_platform
from areal.infra.utils.launcher import TRITON_CACHE_PATH
from areal.utils import logging, perf_tracer, stats_tracker
from areal.utils.network import format_host_for_url

logger = logging.getLogger("SGLangBackend")


class SGLangBackend:
    """SGLang-specific backend implementation for remote inference."""

    def build_generation_request(
        self, req: ModelRequest, with_lora: bool, version: int
    ) -> HttpRequest:
        """Build SGLang generation request."""
        gconfig = req.gconfig
        stop_token_ids = gconfig.stop_token_ids
        stop = gconfig.stop

        if gconfig.use_beam_search:
            raise NotImplementedError(
                "Currently Beam search is not supported in SGLang backend."
            )

        sample_params = {
            "top_p": gconfig.top_p,
            "top_k": gconfig.top_k,
            "max_new_tokens": gconfig.max_new_tokens,
            "temperature": 0.0 if gconfig.greedy else gconfig.temperature,
            "stop_token_ids": stop_token_ids,
            "ignore_eos": gconfig.ignore_eos,
            "skip_special_tokens": gconfig.skip_special_tokens,
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

        # Add return_routed_experts to payload if set
        if req.metadata.get("return_routed_experts", False):
            payload["return_routed_experts"] = True
        # Add LoRA if initialized
        if with_lora:
            lora_name = gconfig.lora_name
            if not lora_name:
                raise ValueError(
                    "LoRA name (gconfig.lora_name) is required when use_lora is enabled."
                )
            payload["lora_path"] = get_versioned_lora_name(lora_name, version)

        return HttpRequest(endpoint="/generate", payload=payload)

    def parse_generation_response(
        self, response: dict[str, Any]
    ) -> HttpGenerationResult:
        """Parse SGLang generation response."""
        meta_info = response["meta_info"]
        finish_reason = meta_info["finish_reason"]
        stop_reason = finish_reason["type"]
        stop_message = finish_reason.get("message", "")

        # Extract routed_experts information if available
        routed_experts = meta_info.get("routed_experts", None)
        if routed_experts is not None:
            num_sgl_token = (
                meta_info["prompt_tokens"] + meta_info["completion_tokens"] - 1
            )
            # Extract expert_id and reshape to (num_sgl_token, num_layers*expert_top_k)
            routed_experts = np.frombuffer(
                pybase64.b64decode(routed_experts.encode("utf-8")), dtype=np.int32
            ).reshape(num_sgl_token, -1)

        if stop_reason == "abort" and stop_message.startswith("Abort before prefill"):
            return HttpGenerationResult(
                output_tokens=[],
                output_logprobs=[],
                stop_reason=stop_reason,
                routed_experts=routed_experts,
            )

        output_tokens = [x[1] for x in meta_info["output_token_logprobs"]]
        output_logprobs = [x[0] for x in meta_info["output_token_logprobs"]]

        return HttpGenerationResult(
            output_tokens=output_tokens,
            output_logprobs=output_logprobs,
            stop_reason=stop_reason,
            routed_experts=routed_experts,
        )

    def build_disk_weight_update_requests(
        self, meta: WeightUpdateMeta
    ) -> WeightUpdateRequests:
        """Build SGLang disk weight update requests."""
        if meta.use_lora:
            if not meta.lora_name:
                raise ValueError("LoRA name is required for LoRA update.")
            if meta.version is None:
                raise ValueError("Version is required for LoRA update.")
            lora_name = get_versioned_lora_name(meta.lora_name, meta.version)
            # Load new LoRA
            requests = [
                HttpRequest(
                    endpoint="/load_lora_adapter",
                    payload={"lora_name": lora_name, "lora_path": str(meta.path)},
                )
            ]
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
        self, meta: WeightUpdateMeta, param_specs: list[ParamSpec]
    ) -> WeightUpdateRequests:
        """Build SGLang distributed weight update requests.

        When lora_delta_sync is enabled on the meta, LoRA is handled in two
        phases orchestrated by the FSDP engine:
          - base_sync_done=False: base model params are pushed via the normal
            ``/update_weights_from_distributed`` endpoint.
          - base_sync_done=True: LoRA adapter params are pushed via the
            ``/load_lora_adapter_from_distributed`` endpoint (or tensor API).

        For non-delta-sync LoRA, the original error is preserved to guide
        users towards disk-based updates.
        """
        if meta.use_lora and not meta.lora_delta_sync:
            raise ValueError(
                "SGLang distributed (XCCL/NCCL) weight update does not support LoRA "
                "without lora_delta_sync. Use weight_update_mode='disk' for LoRA "
                "weight updates with SGLang, or enable lora_delta_sync=True."
            )

        if meta.lora_delta_sync and meta.base_sync_done:
            # Adapter-only phase: this should no longer be reached because
            # the FSDP engine now saves adapters to disk and uses
            # /load_lora_adapter directly.  If we get here, something is wrong.
            logger.error(
                "[LoRA Delta Sync] build_distributed_weight_update_requests called "
                f"with base_sync_done=True for {len(param_specs)} adapter params. "
                "This path is no longer supported; adapter loading should go "
                "through the disk-based path in fsdp_engine.py."
            )
            raise NotImplementedError(
                "Distributed LoRA adapter loading is not supported. "
                "The FSDP engine should use disk-based adapter loading instead."
            )

        # Base model phase (or non-LoRA): standard distributed weight update
        param_names = [pspec.name for pspec in param_specs]
        if meta.lora_delta_sync:
            logger.info(
                "[LoRA Delta Sync] Building distributed base-model weight update "
                f"requests for {len(param_specs)} base params"
            )

        # Log first 5 param names for debugging name-matching issues
        logger.info(
            f"[Weight Update] First 5 param names being sent to SGLang: "
            f"{param_names[:5]}"
        )
        # Estimate total payload size
        total_elements = 0
        for pspec in param_specs:
            if pspec.shape:
                elems = 1
                for s in pspec.shape:
                    elems *= s
                total_elements += elems
            else:
                total_elements += 1
        estimated_size_mb = total_elements * 2 / 1024 / 1024
        logger.info(
            f"[Weight Update] Total params: {len(param_specs)}, "
            f"estimated payload (fp16): {estimated_size_mb:.2f} MB"
        )

        return WeightUpdateRequests(
            requests=[
                HttpRequest(
                    endpoint="/update_weights_from_distributed",
                    payload={
                        "names": param_names,
                        "dtypes": [pspec.dtype for pspec in param_specs],
                        "shapes": [pspec.shape for pspec in param_specs],
                        "group_name": meta.nccl_group_name,
                        "abort_all_requests": True,
                    },
                )
            ]
        )

    def build_lora_adapter_load_requests(
        self, meta: WeightUpdateMeta, param_specs: list[ParamSpec]
    ) -> WeightUpdateRequests:
        """Build requests to load LoRA adapter weights.

        .. note::

           SGLang v0.5.9 does NOT have a ``/load_lora_adapter_from_distributed``
           endpoint.  LoRA adapters should be loaded via the disk-based path
           (``/load_lora_adapter``) instead.  The FSDP engine now saves the
           adapter to disk and calls ``update_weights_from_disk`` directly.

           This method is kept as a safety net and raises ``NotImplementedError``
           to prevent silent misuse.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Must have ``peft_config`` populated with LoRA hyper-parameters.
        param_specs : list[ParamSpec]
            Specifications for the adapter tensors being sent.

        Raises
        ------
        NotImplementedError
            Always raised.  Use the disk-based adapter loading path instead.
        """
        raise NotImplementedError(
            "SGLang v0.5.9 does not support /load_lora_adapter_from_distributed. "
            "LoRA adapters should be loaded via the disk-based path "
            "(/load_lora_adapter). The FSDP engine handles this automatically "
            "when lora_delta_sync is enabled."
        )

    def build_lora_adapter_unload_requests(
        self, lora_name: str, version: int = 0
    ) -> WeightUpdateRequests:
        """Build requests to unload a LoRA adapter from the SGLang server.

        Parameters
        ----------
        lora_name : str
            Base LoRA adapter name.
        version : int
            Version number for versioned adapter naming.

        Returns
        -------
        WeightUpdateRequests
            A single request targeting the adapter unload endpoint.
        """
        versioned_name = get_versioned_lora_name(lora_name, version)
        logger.info(
            f"[LoRA Delta Sync] Building adapter unload request: "
            f"lora_name='{versioned_name}', "
            f"payload={{'lora_name': '{versioned_name}'}}"
        )
        return WeightUpdateRequests(
            requests=[
                HttpRequest(
                    endpoint="/unload_lora_adapter",
                    payload={"lora_name": versioned_name},
                )
            ]
        )

    def build_init_weights_group_request(
        self, addr: str, server_idx: int, meta: WeightUpdateMeta
    ) -> HttpRequest:
        """Build SGLang init weights group request."""
        assert meta.gen_allocation is not None
        gen_parallel = meta.gen_allocation.parallel
        if gen_parallel.pp_size != 1:
            raise NotImplementedError(
                "NCCL weight update with PP size > 1 is not implemented yet."
            )
        rank_offset = 1 + server_idx * gen_parallel.tp_size
        payload = {
            "master_address": format_host_for_url(meta.nccl_master_address),
            "master_port": str(meta.nccl_master_port),
            "rank_offset": rank_offset,
            "world_size": gen_parallel.world_size + 1,
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

    def get_offload_request(self) -> HttpRequest:
        """Get SGLang offload request."""
        return HttpRequest(endpoint="/release_memory_occupation", payload={})

    def get_onload_request(self, tags: list[str] | None = None) -> HttpRequest:
        """Get SGLang onload request.

        Parameters:
        ----------
        tags: list[str], optional
            Available tags for multi-stage resume: weights, kv_cache
        """
        payload = {"tags": tags} if tags is not None else {}
        return HttpRequest(endpoint="/resume_memory_occupation", payload=payload)

    def launch_server(self, server_args: dict[str, Any]) -> subprocess.Popen:
        """Launch SGLang server subprocess."""
        cmd = SGLangConfig.build_cmd_from_args(server_args)
        _env = os.environ.copy()
        triton_cache_path = _env.get("TRITON_CACHE_PATH", TRITON_CACHE_PATH)
        _env["TRITON_CACHE_PATH"] = os.path.join(triton_cache_path, str(uuid.uuid4()))

        return subprocess.Popen(
            cmd,
            env=_env,
            stdout=sys.stdout,
            stderr=sys.stdout,
        )


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
        self.config = config
        # Pure composition - create internal engine with SGLang backend
        self._engine = RemoteInfEngine(config, SGLangBackend())

    def initialize(
        self,
        engine_id: str | None = None,
        addr: str | list[str] | None = None,
        train_data_parallel_size: int | None = None,
    ):
        """Initialize the engine by discovering and connecting to servers."""
        return self._engine.initialize(engine_id, addr, train_data_parallel_size)

    def destroy(self):
        """Destroy the engine and clean up resources."""
        return self._engine.destroy()

    @property
    def initialized(self) -> bool:
        return self._engine.initialized

    @property
    def workflow_executor(self) -> WorkflowExecutor:
        """Get the workflow executor of the inference engine."""
        return self._engine.workflow_executor

    def set_version(self, version: int):
        """Set the current weight version."""
        return self._engine.set_version(version)

    def get_version(self) -> int:
        """Get the current weight version."""
        return self._engine.get_version()

    def set_proxy_gateway_addr(self, addr: str) -> None:
        return self._engine.set_proxy_gateway_addr(addr)

    async def agenerate(self, req: ModelRequest) -> ModelResponse:
        """Asynchronously generate a response for the given request."""
        return await self._engine.agenerate(req)

    def init_weights_update_group(
        self, meta: WeightUpdateMeta, xccl_group_ranks: list[int] | None = None
    ) -> Future[None]:
        """Initialize the weight update process group."""
        return self._engine.init_weights_update_group(
            meta, xccl_group_ranks=xccl_group_ranks
        )

    def update_weights_from_distributed(
        self, meta: WeightUpdateMeta, param_specs: list[ParamSpec]
    ) -> Future[None]:
        """Update weights from distributed memory."""
        return self._engine.update_weights_from_distributed(meta, param_specs)

    def update_weights_from_disk(self, meta: WeightUpdateMeta) -> Future[None]:
        """Update weights from disk."""
        return self._engine.update_weights_from_disk(meta)


    def load_lora_adapter(
        self, lora_name: str, lora_path: str,
        prev_lora_name: str | None = None,
    ) -> Future[None]:
        """Load a LoRA adapter from a local path on all SGLang servers."""
        return self._engine.load_lora_adapter(
            lora_name, lora_path, prev_lora_name=prev_lora_name,
        )

    def unload_lora_adapter(
        self, lora_name: str
    ) -> Future[None]:
        """Unload a LoRA adapter from all SGLang servers."""
        return self._engine.unload_lora_adapter(lora_name)

    def submit(
        self,
        data: dict[str, Any],
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: Callable[[dict[str, Any]], bool] | str | None = None,
        group_size: int = 1,
        task_id: int | None = None,
        callback_addr: str | None = None,
        is_eval: bool = False,
        proxy_addr: str | None = None,
    ) -> int:
        """Submit a request to the inference engine."""
        return self._engine.submit(
            data=data,
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            should_accept_fn=should_accept_fn,
            group_size=group_size,
            task_id=task_id,
            callback_addr=callback_addr,
            is_eval=is_eval,
            proxy_addr=proxy_addr,
        )

    def wait(
        self, count: int, timeout: float | None = None, raise_timeout: bool = True
    ) -> list[dict[str, Any] | None]:
        """Wait for a specified number of requests to complete."""
        return self._engine.wait(count, timeout, raise_timeout)

    def wait_for_task(
        self, task_id: int, timeout: float | None = None, raise_timeout: bool = True
    ) -> dict[str, Any] | None:
        """Wait for a specific task to complete by task_id."""
        return self._engine.wait_for_task(task_id, timeout, raise_timeout)

    def rollout_batch(
        self,
        data: list[dict[str, Any]],
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        group_size: int = 1,
    ) -> dict[str, Any]:
        """Submit a batch of requests and wait for results.

        This method does not support asynchronous rollout and should be used for offline
        data collection or debugging, not in production experiments.
        """
        return self._engine.rollout_batch(
            data=data,
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            group_size=group_size,
        )

    def prepare_batch(
        self,
        dataloader: StatefulDataLoader,
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: Callable[[dict[str, Any]], bool] | str | None = None,
        group_size: int = 1,
        dynamic_bs: bool = False,
    ):
        """Asynchronously submit and wait until a full batch is ready."""
        return self._engine.prepare_batch(
            dataloader=dataloader,
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            should_accept_fn=should_accept_fn,
            group_size=group_size,
            dynamic_bs=dynamic_bs,
        )

    def pause(self):
        return self._engine.pause()

    def resume(self):
        return self._engine.resume()

    def pause_generation(self):
        return self._engine.pause_generation()

    def continue_generation(self):
        return self._engine.continue_generation()

    def launch_server(self, server_args: dict[str, Any]) -> LocalInfServerInfo:
        return self._engine.launch_server(server_args)

    def teardown_server(self):
        return self._engine.teardown_server()

    def offload(self):
        return self._engine.offload()

    def onload(self, tags: list[str] | None = None):
        return self._engine.onload(tags=tags)

    def export_stats(self) -> dict[str, float]:
        return stats_tracker.export_all(reduce_group=None)

    @classmethod
    def as_controller(
        cls, config: InferenceEngineConfig, scheduler: Scheduler
    ) -> RolloutController:
        return RolloutController(cls, config=config, scheduler=scheduler)

    def clear_batches(self, *args):
        """Placeholder method of single-controller API."""

    def save_perf_tracer(self, step: int | None = None, force: bool = False) -> None:
        perf_tracer.save(step=step, force=force)

    def config_perf_tracer(
        self, config: PerfTracerConfig, rank: int, role: str
    ) -> None:
        if perf_tracer.is_configured():
            return
        perf_tracer.configure(config, rank=rank, role=role)
