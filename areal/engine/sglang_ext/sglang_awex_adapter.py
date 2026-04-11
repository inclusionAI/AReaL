from __future__ import annotations

import base64
import pickle
import threading
from typing import TYPE_CHECKING, Any

from areal.utils import logging

if TYPE_CHECKING:
    import sglang


logger = logging.getLogger("SGLangAwexAdapter")

_PICKLE_PREFIX = "__awex_pickle_base64__:"


class AwexSGLangServerAdapter:
    """Server-side Awex adapter that dispatches through ``sgl.Engine`` RPC.

    The adapter mirrors Awex's vLLM server adapter shape while using SGLang's
    scheduler-side RPC mechanism.
    """

    def __init__(
        self,
        sgl_engine: sglang.Engine,
        meta_server_addr: str,
        engine_rank: int = 0,
        num_engines: int = 1,
        comm_backend: str = "file",
        enable_debug_mode: bool = False,
        debug_mode_config: dict[str, Any] | None = None,
        disable_weights_exchange_pipeline: bool = False,
        enable_colocate_mode: bool = False,
        weights_exchange_ipc_backend: str = "cuda",
        weights_comm_nccl_group_size: int = 1,
        nnodes: int | None = None,
        node_rank: int | None = None,
        weights_validation_steps: int = 0,
        validate_weights_every_n_steps: int = 1,
        dump_weights_list_for_validation: list[str] | None = None,
        dump_weights_dir_for_validation: str | None = None,
    ):
        try:
            from awex.config import InferenceConfig
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "awex is required for SGLang awex adapter. Install areal[awex]."
            ) from exc

        self._sgl_engine = sgl_engine
        self._initialized = False
        self._initializing = False
        self._rpc_lock = threading.Lock()

        server_args = getattr(sgl_engine, "server_args", None)
        tp_size = int(getattr(server_args, "tp_size", 1) or 1)
        dp_size = int(getattr(server_args, "dp_size", 1) or 1)
        pp_size = int(getattr(server_args, "pp_size", 1) or 1)
        resolved_nnodes = int(
            nnodes if nnodes is not None else getattr(server_args, "nnodes", 1) or 1
        )
        resolved_node_rank = int(
            node_rank
            if node_rank is not None
            else getattr(server_args, "node_rank", 0) or 0
        )

        self._config = InferenceConfig(
            tp_size=tp_size,
            pp_size=pp_size,
            dp_size=dp_size,
            ep_size=1,
            enable_dp_attention=False,
            # Keep lm_head metadata visible for writer/reader strict key checks.
            enable_dp_lm_head=True,
            moe_dense_tp_size=None,
            nnodes=resolved_nnodes,
            node_rank=resolved_node_rank,
            num_engines=num_engines,
            engine_rank=engine_rank,
            meta_server_addr=meta_server_addr,
            comm_backend=comm_backend,
            enable_debug_mode=enable_debug_mode,
            debug_mode_config=debug_mode_config or {},
            disable_weights_exchange_pipeline=disable_weights_exchange_pipeline,
            enable_colocate_mode=enable_colocate_mode,
            weights_exchange_ipc_backend=weights_exchange_ipc_backend,
            weights_comm_nccl_group_size=weights_comm_nccl_group_size,
            weights_validation_steps=weights_validation_steps,
            validate_weights_every_n_steps=validate_weights_every_n_steps,
            dump_weights_list_for_validation=dump_weights_list_for_validation or [],
            dump_weights_dir_for_validation=dump_weights_dir_for_validation,
        )
        self.hf_config = self._resolve_hf_config(server_args)
        self.engine_name = "sglang"
        self.weights_exchange_reader = None

    def _resolve_hf_config(self, server_args):
        """Resolve HF config object expected by awex readers.

        awex calls ``simple_hf_config(self.hf_config)`` and requires a config-like
        object with ``to_dict()`` (e.g., ``transformers.PretrainedConfig``).
        SGLang wrappers (e.g., ModelConfig) may contain nested HF config but are not
        themselves accepted by awex.
        """

        def _has_to_dict(obj: Any) -> bool:
            return hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict"))

        def _unwrap_to_hf_config(candidate: Any, *, depth: int = 0) -> Any | None:
            if candidate is None or depth > 5:
                return None
            # Prefer nested canonical HF config when wrappers expose both wrapper
            # attrs and nested hf_config/config.
            for attr in ("hf_config", "config", "model_config"):
                nested = getattr(candidate, attr, None)
                if nested is None or nested is candidate:
                    continue
                resolved = _unwrap_to_hf_config(nested, depth=depth + 1)
                if resolved is not None:
                    return resolved
            if _has_to_dict(candidate):
                return candidate
            return None

        # Preferred: use already-built config from the live SGLang engine.
        tokenizer_manager = getattr(self._sgl_engine, "tokenizer_manager", None)
        tm_model_config = getattr(tokenizer_manager, "model_config", None)
        resolved = _unwrap_to_hf_config(tm_model_config)
        if resolved is not None:
            return resolved

        model_config = getattr(self._sgl_engine, "model_config", None)
        resolved = _unwrap_to_hf_config(model_config)
        if resolved is not None:
            return resolved

        direct_cfg = getattr(self._sgl_engine, "hf_config", None)
        resolved = _unwrap_to_hf_config(direct_cfg)
        if resolved is not None:
            return resolved

        model_path = getattr(server_args, "model_path", None)
        if not model_path:
            raise RuntimeError(
                "Cannot resolve hf_config for awex adapter: model_path missing"
            )

        try:
            from transformers import AutoConfig
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "transformers is required to resolve hf_config for awex adapter"
            ) from exc

        logger.info("Loading hf_config from model_path for awex adapter")
        loaded = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if _has_to_dict(loaded):
            return loaded
        raise RuntimeError(
            "Cannot resolve awex-compatible hf_config with to_dict(); "
            f"loaded type={type(loaded).__name__}"
        )

    @property
    def config(self):
        return self._config

    @property
    def num_engines(self):
        return self._config.num_engines

    @property
    def engine_rank(self):
        return self._config.engine_rank

    def initialize(self) -> None:
        if self._initialized or self._initializing:
            return
        self._initializing = True
        try:
            from awex.reader.weights_reader import get_weights_exchange_reader

            self._patch_awex_sglang_converter()
            logger.info("Initializing Awex weights reader in SGLang server process")
            self.weights_exchange_reader = get_weights_exchange_reader(self)
            self.weights_exchange_reader.initialize()
            self._initialized = True
        finally:
            self._initializing = False

    def _patch_awex_sglang_converter(self) -> None:
        """Patch awex SGLang converter for q_norm/k_norm naming variants."""

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
        logger.info("Patched awex SGlangToHFWeightConverter for q_norm/k_norm")

    def update_weights(self, step_id: int, **kwargs):
        if not self._initialized:
            raise RuntimeError("Awex adapter not initialized.")
        if self.weights_exchange_reader is None:
            raise RuntimeError("Awex weights reader is unavailable")
        self.weights_exchange_reader.update_weights(step_id=step_id, **kwargs)

    def update_weights_from_disk(self, model_path: str, load_format: str | None = None):
        updater = getattr(self._sgl_engine, "update_weights_from_disk", None)
        if updater is None:
            raise RuntimeError("SGLang engine has no update_weights_from_disk")
        return updater(model_path=model_path, load_format=load_format)

    def execute_task_in_model_worker(self, fn, **kwargs):
        if not self._initialized and not self._initializing:
            raise RuntimeError("Awex adapter not initialized.")

        if isinstance(fn, str):
            method = fn
            payload = kwargs
        else:
            method = "awex_execute"
            infer_engine_config = kwargs.get("infer_engine_config")
            if infer_engine_config is not None and hasattr(
                infer_engine_config, "__dict__"
            ):
                kwargs = dict(kwargs)
                kwargs["infer_engine_config"] = infer_engine_config.__dict__
            payload = {
                "task_module": fn.__module__,
                "task_qualname": fn.__qualname__,
                "task_kwargs": kwargs,
            }
        return self._collective_rpc_with_result(method, **payload)

    def release_memory_occupation(self, tags=None) -> None:
        self._sgl_engine.release_memory_occupation(tags=tags)

    def resume_memory_occupation(self, tags=None) -> None:
        self._sgl_engine.resume_memory_occupation(tags=tags)

    def _collective_rpc_with_result(self, method: str, **kwargs):
        try:
            from sglang.srt.managers.io_struct import RpcReqInput, RpcReqOutput
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "SGLang RPC structs unavailable for awex dispatch"
            ) from exc

        rpc_socket = getattr(self._sgl_engine, "send_to_rpc", None)
        if rpc_socket is None:
            raise RuntimeError("SGLang engine has no send_to_rpc socket")

        request = RpcReqInput(method=method, parameters=kwargs)
        with self._rpc_lock:
            rpc_socket.send_pyobj(request)
            response = rpc_socket.recv_pyobj()

        if not isinstance(response, RpcReqOutput):
            return response
        if not response.success:
            raise RuntimeError(response.message)

        message = response.message
        if isinstance(message, bytes):
            return pickle.loads(message)
        if isinstance(message, str) and message.startswith(_PICKLE_PREFIX):
            encoded = message[len(_PICKLE_PREFIX) :]
            return pickle.loads(base64.b64decode(encoded))
        if message in (None, ""):
            return None
        return message
