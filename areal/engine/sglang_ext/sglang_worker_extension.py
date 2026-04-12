from __future__ import annotations

import base64
import copy
import pickle
import traceback
import types
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

_PARAM_NAME_ALIAS_RULES = (
    (".self_attn.o_proj.", ".attention.dense."),
    (".attention.dense.", ".self_attn.o_proj."),
    (".attention.o_proj.", ".attention.dense."),
    (".attention.dense.", ".attention.o_proj."),
    (".self_attn.qkv_proj.", ".attention.query_key_value_proj."),
    (".attention.query_key_value_proj.", ".self_attn.qkv_proj."),
    (".self_attn.q_proj.", ".attention.q_proj."),
    (".attention.q_proj.", ".self_attn.q_proj."),
    (".self_attn.k_proj.", ".attention.k_proj."),
    (".attention.k_proj.", ".self_attn.k_proj."),
    (".self_attn.v_proj.", ".attention.v_proj."),
    (".attention.v_proj.", ".self_attn.v_proj."),
    (".self_attn.query_layernorm.", ".attention.query_layernorm."),
    (".attention.query_layernorm.", ".self_attn.query_layernorm."),
    (".self_attn.key_layernorm.", ".attention.key_layernorm."),
    (".attention.key_layernorm.", ".self_attn.key_layernorm."),
)


def _safe_exc_message(exc: Exception) -> str:
    """Best-effort stringification for cross-process error transport."""
    try:
        return repr(exc)
    except Exception:
        try:
            return str(type(exc))
        except Exception:
            return f"{type(exc).__name__}: <unprintable>"


def _normalize_awex_param_meta_keys(result: Any) -> Any:
    """Normalize infer param-meta keys to awex Megatron-style naming.

    awex writer/reader strict meta validation compares training-side Megatron names
    (e.g. ``attention.dense``, ``attention.query_key_value_proj``) with infer-side
    resolved names. For Qwen3 SGLang side, infer meta can appear as
    ``self_attn.o_proj`` / ``self_attn.qkv_proj``. Normalize those variants to the
    Megatron-style names expected by awex checks.
    """

    def _is_param_name(name: str) -> bool:
        return (
            name.endswith(".weight")
            or name.endswith(".bias")
            or name == "lm_head.weight"
        )

    def _is_param_map(d: dict[Any, Any]) -> bool:
        return any(isinstance(k, str) and _is_param_name(k) for k in d.keys())

    def _map_param_key(key: str) -> str:
        mapped = key
        if ".self_attn." in mapped:
            mapped = mapped.replace(".self_attn.", ".attention.")
        mapped = mapped.replace(".attention.o_proj.", ".attention.dense.")
        mapped = mapped.replace(
            ".attention.qkv_proj.", ".attention.query_key_value_proj."
        )
        return mapped

    def _normalize(obj: Any) -> Any:
        if isinstance(obj, str) and _is_param_name(obj):
            return _map_param_key(obj)
        if isinstance(obj, list):
            return [_normalize(item) for item in obj]
        if not isinstance(obj, dict):
            return obj

        normalized: dict[Any, Any] = {}
        param_map = _is_param_map(obj)

        for key, value in obj.items():
            mapped_key = (
                _map_param_key(key) if (param_map and isinstance(key, str)) else key
            )
            normalized.setdefault(mapped_key, _normalize(value))

        # Some awex structures carry parameter names in values
        # (e.g. {"name": "model.layers..."}) instead of dict keys.
        for name_field in ("name", "param_name", "weight_name"):
            field_val = normalized.get(name_field)
            if isinstance(field_val, str) and _is_param_name(field_val):
                normalized[name_field] = _map_param_key(field_val)

        # For parameter maps, ensure lm_head.weight exists when tied embeddings are used.
        if param_map and "lm_head.weight" not in normalized:
            for candidate in normalized.keys():
                if not isinstance(candidate, str):
                    continue
                if candidate.endswith("embed_tokens.weight") or candidate.endswith(
                    "tok_embeddings.weight"
                ):
                    normalized["lm_head.weight"] = normalized[candidate]
                    break
            else:
                for candidate in (
                    "model.output_layer.weight",
                    "transformer.wte.weight",
                ):
                    if candidate in normalized:
                        normalized["lm_head.weight"] = normalized[candidate]
                        break

        return normalized

    def _add_lm_head_alias_in_name_entries(obj: Any) -> Any:
        if isinstance(obj, list):
            names: set[str] = set()
            template_item: dict[str, Any] | None = None
            template_field: str | None = None

            for item in obj:
                if not isinstance(item, dict):
                    continue
                for field in ("name", "param_name", "weight_name"):
                    val = item.get(field)
                    if isinstance(val, str):
                        names.add(val)
                        if val.endswith("embed_tokens.weight") or val.endswith(
                            "tok_embeddings.weight"
                        ):
                            template_item = item
                            template_field = field

            if (
                "lm_head.weight" not in names
                and template_item is not None
                and template_field is not None
            ):
                alias_item = copy.deepcopy(template_item)
                alias_item[template_field] = "lm_head.weight"
                obj = [*obj, alias_item]

            return [_add_lm_head_alias_in_name_entries(x) for x in obj]

        if isinstance(obj, dict):
            return {k: _add_lm_head_alias_in_name_entries(v) for k, v in obj.items()}
        return obj

    normalized = _normalize(result)
    normalized = _add_lm_head_alias_in_name_entries(normalized)
    return normalized


def _gather_awex_param_meta_across_workers(result: Any) -> list[dict[str, Any]]:
    """Collect infer parameter metadata from all distributed TP workers.

    SGLang awex RPC returns a single worker response to the caller. For
    ``InferParamMetaResolver._get_model_param_info`` we must aggregate metadata
    across all TP workers so awex can reconstruct global parameter shapes.
    """

    if isinstance(result, list):
        local_entries = [x for x in result if isinstance(x, dict)]
    elif isinstance(result, dict):
        local_entries = [result]
    else:
        return []

    try:
        import torch.distributed as dist
    except Exception:
        return local_entries

    if not (dist.is_available() and dist.is_initialized()):
        return local_entries

    world_size = dist.get_world_size()
    if world_size <= 1:
        return local_entries

    gathered: list[Any] = [None] * world_size
    dist.all_gather_object(gathered, local_entries)

    merged: list[dict[str, Any]] = []
    for item in gathered:
        if isinstance(item, list):
            merged.extend(x for x in item if isinstance(x, dict))
        elif isinstance(item, dict):
            merged.append(item)

    dedup: dict[int, dict[str, Any]] = {}
    fallback: list[dict[str, Any]] = []
    for meta in merged:
        rank_info = meta.get("rank_info")
        global_rank = getattr(rank_info, "global_rank", None)
        if isinstance(global_rank, int):
            if global_rank in dedup:
                prev = dedup[global_rank]
                prev_rank_info = prev.get("rank_info")
                prev_tp_rank = getattr(prev_rank_info, "tp_rank", None)
                curr_tp_rank = getattr(rank_info, "tp_rank", None)
                logger.warning(
                    "Duplicate awex infer meta for global_rank=%s (prev_tp_rank=%s curr_tp_rank=%s). "
                    "Keeping first entry.",
                    global_rank,
                    prev_tp_rank,
                    curr_tp_rank,
                )
                continue
            dedup[global_rank] = meta
        else:
            fallback.append(meta)

    ordered = [dedup[k] for k in sorted(dedup.keys())]
    ordered.extend(fallback)
    if ordered:
        expected_world_size = 0
        for meta in ordered:
            rank_info = meta.get("rank_info")
            expected_world_size = max(
                expected_world_size, int(getattr(rank_info, "world_size", 0) or 0)
            )
        if expected_world_size > 0 and len(dedup) < expected_world_size:
            logger.warning(
                "Incomplete awex infer meta gather: got %s unique ranks, expected at least %s. "
                "This may cause TP shard-size mismatch.",
                len(dedup),
                expected_world_size,
            )
    return ordered if ordered else local_entries


def _inject_awex_parameter_aliases(parameters: dict[str, Any]) -> int:
    """Inject missing parameter aliases for awex transfer-plan name variants.

    awex transfer plans may use either SGLang-style names (``self_attn.*``)
    or canonical attention names (``attention.*``). Ensure both spellings are
    present in ``parameters`` so recv-side lookups do not fail.
    """

    if not isinstance(parameters, dict) or not parameters:
        return 0

    added = 0
    existing_names = list(parameters.keys())
    for name in existing_names:
        if not isinstance(name, str):
            continue
        value = parameters[name]
        for src, dst in _PARAM_NAME_ALIAS_RULES:
            if src not in name:
                continue
            alias = name.replace(src, dst)
            if alias not in parameters:
                parameters[alias] = value
                added += 1

    # Keep tied embedding aliases symmetric for readers/writers expecting either key.
    if "lm_head.weight" not in parameters:
        for emb in (
            "model.embed_tokens.weight",
            "model.tok_embeddings.weight",
            "model.output_layer.weight",
            "transformer.wte.weight",
        ):
            if emb in parameters:
                parameters["lm_head.weight"] = parameters[emb]
                added += 1
                break
    if "model.output_layer.weight" not in parameters and "lm_head.weight" in parameters:
        parameters["model.output_layer.weight"] = parameters["lm_head.weight"]
        added += 1
    if "model.embed_tokens.weight" not in parameters and "lm_head.weight" in parameters:
        parameters["model.embed_tokens.weight"] = parameters["lm_head.weight"]
        added += 1

    return added


def _patch_awex_reader_parameter_aliases() -> None:
    """Patch awex WorkerWeightsReader.initialize to inject param-name aliases."""

    try:
        from awex.reader.weights_reader import WorkerWeightsReader
    except ImportError:
        return

    if getattr(WorkerWeightsReader, "_areal_param_alias_patch", False):
        return

    original_initialize = WorkerWeightsReader.initialize

    def _patched_initialize(self, *args, **kwargs):
        out = original_initialize(self, *args, **kwargs)
        params = getattr(self, "parameters", None)
        added = (
            _inject_awex_parameter_aliases(params) if isinstance(params, dict) else 0
        )
        if added > 0:
            logger.info(
                "Injected %s awex parameter aliases for rank %s",
                added,
                getattr(getattr(self, "rank_info", None), "global_rank", "unknown"),
            )
        return out

    WorkerWeightsReader.initialize = _patched_initialize
    WorkerWeightsReader._areal_param_alias_patch = True
    logger.info("Patched awex WorkerWeightsReader.initialize parameter aliases")


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


def _patch_awex_nccl_barrier_device_ids() -> None:
    """Patch awex NCCL initialize paths to avoid barrier(device_ids=...).

    In some containerized environments with custom CUDA visibility mappings,
    ``dist.barrier(..., device_ids=[...])`` can trigger NCCL/Torch runtime errors.
    For awex reader/writer initialize only, drop ``device_ids`` from barrier calls.
    """

    try:
        from awex.reader.nccl_reader import NCCLWeightsReader
    except ImportError:
        NCCLWeightsReader = None

    try:
        from awex.writer.nccl_writer import NCCLWeightsWriter
    except ImportError:
        NCCLWeightsWriter = None

    def _wrap_initialize(cls, tag: str):
        if cls is None or getattr(cls, "_areal_barrier_device_patch", False):
            return

        original_initialize = cls.initialize

        def _patched_initialize(self, *args, **kwargs):
            import torch.distributed as dist

            original_barrier = dist.barrier

            def _barrier_without_device_ids(*b_args, **b_kwargs):
                b_kwargs.pop("device_ids", None)
                return original_barrier(*b_args, **b_kwargs)

            dist.barrier = _barrier_without_device_ids
            try:
                return original_initialize(self, *args, **kwargs)
            finally:
                dist.barrier = original_barrier

        cls.initialize = _patched_initialize
        cls._areal_barrier_device_patch = True
        logger.info("Patched awex %s initialize barrier device_ids", tag)

    _wrap_initialize(NCCLWeightsReader, "NCCLWeightsReader")
    _wrap_initialize(NCCLWeightsWriter, "NCCLWeightsWriter")


def _run_with_barrier_device_ids_stripped(fn, *args, **kwargs):
    """Run callable while stripping ``device_ids`` from dist.barrier kwargs.

    Preserve barrier synchronization semantics, but avoid passing ``device_ids``
    which can trigger runtime/device-mapping failures in some environments.
    """

    try:
        import torch.distributed as dist
    except Exception:
        return fn(*args, **kwargs)

    original_barrier = dist.barrier

    def _barrier_without_device_ids(*b_args, **b_kwargs):
        b_kwargs.pop("device_ids", None)
        return original_barrier(*b_args, **b_kwargs)

    dist.barrier = _barrier_without_device_ids
    try:
        return fn(*args, **kwargs)
    finally:
        dist.barrier = original_barrier


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


def _build_fallback_infer_engine_config(scheduler) -> Any:
    """Build minimal config object for awex worker hooks when missing.

    Some awex worker code paths expect ``infer_engine_config`` to exist and expose
    attributes like ``tp_size``/``pp_size``/``dp_size``. If upstream call sites do
    not pass it through, synthesize a best-effort object from scheduler/server args.
    """

    server_args = getattr(scheduler, "server_args", None)
    return types.SimpleNamespace(
        tp_size=int(getattr(server_args, "tp_size", 1) or 1),
        pp_size=int(getattr(server_args, "pp_size", 1) or 1),
        dp_size=int(getattr(server_args, "dp_size", 1) or 1),
        ep_size=1,
        nnodes=int(getattr(server_args, "nnodes", 1) or 1),
        node_rank=int(getattr(server_args, "node_rank", 0) or 0),
        num_engines=1,
        engine_rank=0,
        comm_backend="nccl",
        enable_dp_attention=False,
        enable_dp_lm_head=True,
    )


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
        _patch_awex_nccl_barrier_device_ids()
        _patch_awex_reader_parameter_aliases()

        logger.info(
            "awex_execute start: task=%s.%s kwargs_keys=%s",
            task_module,
            task_qualname,
            sorted((task_kwargs or {}).keys()),
        )

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
        if infer_engine_config is None:
            infer_engine_config = _build_fallback_infer_engine_config(self)
            task_kwargs["infer_engine_config"] = infer_engine_config

        task_kwargs["model"] = self.tp_worker.model_runner.model
        task_kwargs["model_context"] = _build_awex_model_context(
            self, infer_engine_config
        )
        result = _run_with_barrier_device_ids_stripped(target, **task_kwargs)

        # awex InferParamMetaResolver expects execute_task_in_model_worker() to
        # return List[Dict[str, Any]] for _get_model_param_info, even with one
        # worker/rank.
        if task_qualname.endswith("._get_model_param_info"):
            result = _normalize_awex_param_meta_keys(result)
            result = _gather_awex_param_meta_across_workers(result)
            if isinstance(result, dict):
                result = [result]
        logger.info(
            "awex_execute done: task=%s.%s result_type=%s",
            task_module,
            task_qualname,
            type(result).__name__,
        )
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
        err_tb = ""
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
            err_tb = traceback.format_exc()

        barrier()
        if not success:
            msg = (
                _safe_exc_message(err) if err is not None else "awex worker task failed"
            )
            if err_tb:
                msg = f"{msg}\n--- worker traceback ---\n{err_tb}"
            return RpcReqOutput(False, msg)

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
