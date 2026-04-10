from __future__ import annotations

import base64
import pickle
from typing import Any

from areal.utils import logging

logger = logging.getLogger("SGLangWorkerExtension")

_PICKLE_PREFIX = "__awex_pickle_base64__:"

_AWEX_WORKER_METHODS = {
    "_get_model_param_info": (
        "awex.meta.infer_meta_resolver",
        "InferParamMetaResolver._get_model_param_info",
    ),
    "_init_in_tp_worker": (
        "awex.reader.weights_reader",
        "WeightsReader._init_in_tp_worker",
    ),
    "_update_parameters_in_tp_worker": (
        "awex.reader.weights_reader",
        "WeightsReader._update_parameters_in_tp_worker",
    ),
    "_pre_update_weights_in_tp_worker": (
        "awex.reader.weights_reader",
        "WeightsReader._pre_update_weights_in_tp_worker",
    ),
}


def _patch_awex_sglang_converter() -> None:
    """Patch awex SGLang converter in worker process."""

    try:
        from awex.converter.sglang_converter import SGlangToHFWeightConverter
    except ImportError:
        return

    if getattr(SGlangToHFWeightConverter, "_areal_qnorm_patch", False):
        return

    original = SGlangToHFWeightConverter._convert_layer_norm_param

    def _patched_convert_layer_norm_param(this, name, parameter, layer_number):
        if "self_attn.q_norm" in name:
            name = name.replace("self_attn.q_norm", "self_attn.query_layernorm")
        if "self_attn.k_norm" in name:
            name = name.replace("self_attn.k_norm", "self_attn.key_layernorm")
        return original(this, name, parameter, layer_number)

    SGlangToHFWeightConverter._convert_layer_norm_param = (
        _patched_convert_layer_norm_param
    )
    SGlangToHFWeightConverter._areal_qnorm_patch = True
    logger.info("Patched awex SGlangToHFWeightConverter in worker")


def _sanitize_for_ipc(obj: Any) -> Any:
    try:
        import torch

        if isinstance(obj, torch.dtype):
            return str(obj).replace("torch.", "")
        if isinstance(obj, torch.device):
            return str(obj)
    except Exception:
        pass
    if isinstance(obj, dict):
        return {k: _sanitize_for_ipc(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_ipc(v) for v in obj]
    return obj


def _build_awex_model_context(scheduler, infer_engine_config=None) -> dict[str, Any]:
    server_args = getattr(scheduler, "server_args", None)
    tp_rank = int(getattr(scheduler, "tp_rank", 0) or 0)
    tp_size = int(getattr(server_args, "tp_size", 1) or 1)
    pp_rank = int(getattr(scheduler, "pp_rank", 0) or 0)
    pp_size = int(getattr(server_args, "pp_size", 1) or 1)
    dp_rank = int(getattr(scheduler, "dp_rank", 0) or 0)
    dp_size = int(getattr(server_args, "dp_size", 1) or 1)
    local_rank = int(getattr(scheduler, "gpu_id", 0) or 0)

    engine_rank = int(getattr(infer_engine_config, "engine_rank", 0) or 0)
    local_world_size = tp_size * pp_size
    global_rank = dp_rank * local_world_size + (pp_rank * tp_size + tp_rank)

    return {
        "tp_rank": tp_rank,
        "tp_size": tp_size,
        "pp_rank": pp_rank,
        "pp_size": pp_size,
        "dp_rank": dp_rank,
        "dp_size": dp_size,
        "ep_rank": 0,
        "ep_size": 1,
        "ep_tp_rank": 0,
        "ep_tp_size": 1,
        "local_rank": local_rank,
        "global_rank": global_rank,
        "world_size": tp_size * pp_size * dp_size,
        "engine_rank": engine_rank,
        "is_infer": True,
        "attn_tp_rank": tp_rank,
        "attn_tp_size": tp_size,
        "attn_dp_rank": 0,
        "cp_rank": 0,
        "cp_size": 1,
        "cp_mode": "none",
        "scheduler": scheduler,
        "infer_engine_config": infer_engine_config,
    }


def patch_scheduler_for_awex() -> None:
    """Patch SGLang Scheduler to support awex RPC entrypoints."""

    try:
        from sglang.srt.managers.io_struct import RpcReqOutput
        from sglang.srt.managers.scheduler import Scheduler, barrier
    except ImportError as exc:  # pragma: no cover - runtime dependency
        logger.warning("Skip awex scheduler patch: %s", exc)
        return

    if getattr(Scheduler, "_areal_awex_patched", False):
        return

    def awex_execute(self, task_module: str, task_qualname: str, task_kwargs=None):
        _patch_awex_sglang_converter()

        module = __import__(task_module, fromlist=["__dummy__"])
        target = module
        for attr in task_qualname.split("."):
            target = getattr(target, attr)

        task_kwargs = task_kwargs or {}
        infer_engine_config = task_kwargs.get("infer_engine_config")
        if isinstance(infer_engine_config, dict):
            from awex.config import InferenceConfig

            infer_engine_config = InferenceConfig.from_dict(infer_engine_config)
            task_kwargs["infer_engine_config"] = infer_engine_config

        task_kwargs["model"] = self.tp_worker.model_runner.model
        task_kwargs["model_context"] = _build_awex_model_context(
            self, infer_engine_config
        )
        result = target(**task_kwargs)
        return _sanitize_for_ipc(result)

    def _make_awex_worker_method(task_module: str, task_qualname: str):
        def _method(self, **kwargs):
            return awex_execute(self, task_module, task_qualname, kwargs)

        return _method

    original_handle_rpc_request = Scheduler.handle_rpc_request

    def handle_rpc_request_with_result(self, recv_req):
        if (
            not str(getattr(recv_req, "method", "")).startswith("awex_")
            and str(getattr(recv_req, "method", "")) not in _AWEX_WORKER_METHODS
        ):
            return original_handle_rpc_request(self, recv_req)

        success = True
        err: Exception | None = None
        result = None
        try:
            func = getattr(self, recv_req.method)
            if recv_req.parameters is not None:
                result = func(**recv_req.parameters)
            else:
                result = func()
        except Exception as exc:  # pragma: no cover - runtime behavior
            success = False
            err = exc

        barrier()
        if not success:
            return RpcReqOutput(False, str(err))

        if result is None:
            return RpcReqOutput(True, "")
        encoded = base64.b64encode(pickle.dumps(result)).decode("utf-8")
        return RpcReqOutput(True, f"{_PICKLE_PREFIX}{encoded}")

    Scheduler.awex_execute = awex_execute
    for method_name, (task_module, task_qualname) in _AWEX_WORKER_METHODS.items():
        setattr(
            Scheduler,
            method_name,
            _make_awex_worker_method(task_module, task_qualname),
        )
    Scheduler.handle_rpc_request = handle_rpc_request_with_result
    Scheduler._areal_awex_patched = True

    logger.info("Patched SGLang Scheduler for awex RPC methods")
