import dataclasses
from typing import Any

import torch
from mbridge.core.bridge import Bridge
from megatron.core import parallel_state as mpu
from megatron.core import tensor_parallel
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import DistributedDataParallelConfig as MCoreDDPConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer import TransformerConfig
from transformers import AutoConfig, PretrainedConfig

from areal.api.cli_args import MegatronEngineConfig
from areal.models.mcore.bailing_moe import (
    hf_to_mcore_config_bailing_moe,
    make_mcore_layer_specs_bailing_moe,
)
from areal.models.mcore.qwen3 import (
    hf_to_mcore_config_qwen3_dense,
    make_mcore_layer_specs_qwen3_dense,
)
from areal.utils import logging

logger = logging.getLogger("MCoreRegistry")


class ValueHead(torch.nn.Linear):
    def __init__(
        self,
        input_size: int,
        output_size: int = 1,
        *,
        config: TransformerConfig,
        bias: bool = False,
    ) -> None:
        super().__init__(in_features=input_size, out_features=output_size, bias=bias)
        self.sequence_parallel = config.sequence_parallel
        if self.sequence_parallel:
            self.weight.sequence_parallel = True

        self.weight.data.normal_(mean=0.0, std=0.02)
        if bias:
            self.bias.data.zero_()

    def forward(
        self,
        input_: torch.Tensor,
        weight: torch.Tensor | None = None,
        runtime_gather_output: bool | None = None,
    ) -> tuple[torch.Tensor, None]:
        logits = super().forward(input_)
        logits = logits.float()
        if self.sequence_parallel:
            logits = tensor_parallel.gather_from_sequence_parallel_region(
                logits, tensor_parallel_output_grad=False
            )
        return logits, None


def _replace_output_layer_with_value_head(
    model: GPTModel,
    tf_config: TransformerConfig,
) -> None:
    """Replace model's output_layer with ValueHead.

    This function can be used on any GPTModel instance, whether created
    via mbridge or directly. After replacement:
    - model.output_layer becomes a ValueHead instance
    - model.vocab_size is set to 1

    Args:
        model: The GPTModel instance to modify
        tf_config: Transformer configuration containing hidden_size and SP settings
    """
    if not hasattr(model, "output_layer"):
        raise ValueError(
            "Model does not have output_layer. Ensure post_process=True when creating GPTModel."
        )

    dtype = tf_config.params_dtype

    model.output_layer = ValueHead(
        input_size=tf_config.hidden_size,
        output_size=1,
        config=tf_config,
        bias=False,
    ).to(dtype=dtype)

    model.vocab_size = 1


def unwrap_to_gpt_model(model: torch.nn.Module) -> GPTModel:
    """Unwraps a model to the underlying GPTModel instance."""
    _model = model
    while not isinstance(_model, GPTModel) and hasattr(_model, "module"):
        _model = _model.module
    if not isinstance(_model, GPTModel):
        raise TypeError(f"Model could not be unwrapped to GPTModel. Got {type(_model)}")
    return _model


def _ensure_mtp_spec_compat():
    """Patch MTP block-spec functions to gracefully handle TransformerConfig as *spec*.

    **Why multi-level patching is needed**

    ``mbridge.models.mimo`` does::

        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec

    at module load time, which creates a *local* binding to the original
    function object.  Simply replacing ``gpt_layer_specs.get_gpt_mtp_block_spec``
    does NOT affect that already-bound local reference — the original function
    will still be called by mimo.

    However, the original ``get_gpt_mtp_block_spec`` internally calls
    ``get_gpt_mtp_block_spec_for_backend`` through the module's **global
    namespace**, which IS resolved at call time.  Therefore we apply patches
    at three levels for maximum robustness:

    1. ``get_gpt_mtp_block_spec_for_backend`` on the module — catches calls
       coming through *any* import path (including mimo's local reference).
       This is the **critical** patch that actually fixes the bug.
    2. ``get_gpt_mtp_block_spec`` on the module — catches future callers that
       access it via ``gpt_layer_specs.get_gpt_mtp_block_spec``.
    3. The local reference inside ``mbridge.models.mimo`` (if importable) —
       belt-and-suspenders for the direct ``from-import`` case.
    """
    try:
        from megatron.core.models.gpt import gpt_layer_specs as _specs_mod
    except ImportError:
        logger.warning(
            "[MTPCompat] Cannot import gpt_layer_specs from megatron.core; "
            "skipping MTP spec compatibility patch."
        )
        return

    if getattr(_specs_mod, "_areal_mtp_compat_patched", False):
        return

    # ----- helper: convert TransformerConfig → proper ModuleSpec -----
    def _convert_spec_if_needed(config, spec, use_transformer_engine=True):
        if not isinstance(spec, TransformerConfig):
            return spec
        logger.info(
            "[MTPCompat] Auto-converting TransformerConfig -> ModuleSpec "
            "for get_gpt_mtp_block_spec (use_transformer_engine=%s).",
            use_transformer_engine,
        )
        _get_decoder = getattr(_specs_mod, "get_gpt_decoder_block_spec", None)
        if _get_decoder is not None:
            decoder_block_spec = _get_decoder(
                config=config, use_transformer_engine=use_transformer_engine
            )
            spec = decoder_block_spec.layer_specs[-1]
            logger.info(
                "[MTPCompat] Resolved spec via get_gpt_decoder_block_spec."
            )
        elif use_transformer_engine:
            spec = _specs_mod.get_gpt_layer_with_transformer_engine_spec()
            logger.info(
                "[MTPCompat] Resolved spec via "
                "get_gpt_layer_with_transformer_engine_spec."
            )
        else:
            spec = _specs_mod.get_gpt_layer_local_spec()
            logger.info("[MTPCompat] Resolved spec via get_gpt_layer_local_spec.")
        return spec

    # get_gpt_mtp_block_spec_for_backend ---
    # This is the lowest-level function that validates the spec type.
    # Because the original get_gpt_mtp_block_spec resolves this name
    # through the module's global dict at call time, patching here
    # intercepts ALL callers — including mimo's from-imported reference.
    _orig_backend_fn = _specs_mod.get_gpt_mtp_block_spec_for_backend

    def _compat_backend(config, spec, use_transformer_engine=True, **kwargs):
        spec = _convert_spec_if_needed(config, spec, use_transformer_engine)
        return _orig_backend_fn(config, spec, use_transformer_engine, **kwargs)

    _specs_mod.get_gpt_mtp_block_spec_for_backend = _compat_backend

    # --- Patch 2: get_gpt_mtp_block_spec (top-level entry point) ---
    _orig_fn = _specs_mod.get_gpt_mtp_block_spec

    def _compat_fn(config, spec, use_transformer_engine=True, **kwargs):
        spec = _convert_spec_if_needed(config, spec, use_transformer_engine)
        return _orig_fn(config, spec, use_transformer_engine, **kwargs)

    _specs_mod.get_gpt_mtp_block_spec = _compat_fn

    # mbridge.models.mimo local reference (if available) ---
    try:
        import mbridge.models.mimo as _mimo_mod

        if hasattr(_mimo_mod, "get_gpt_mtp_block_spec"):
            _mimo_mod.get_gpt_mtp_block_spec = _compat_fn
            logger.info(
                "[MTPCompat] Also patched mbridge.models.mimo."
                "get_gpt_mtp_block_spec direct reference."
            )
    except (ImportError, AttributeError):
        logger.info(
            "[MTPCompat] mbridge.models.mimo not importable; "
            "relying on backend-level patch only."
        )

    _specs_mod._areal_mtp_compat_patched = True
    logger.info(
        "[MTPCompat] Patched get_gpt_mtp_block_spec AND "
        "get_gpt_mtp_block_spec_for_backend for TransformerConfig compat."
    )


# Model registry for different architectures
def make_hf_and_mcore_config(
    hf_path: str,
    dtype: torch.dtype,
    bridge=None,
    bridge_type: str = "mbridge",
) -> tuple[PretrainedConfig, TransformerConfig]:
    if bridge is not None and bridge_type == "mbridge":
        hf_config = bridge.hf_config
        hf_config._name_or_path = hf_path
        return hf_config, bridge.config
    elif bridge is not None and bridge_type == "megatron-bridge":
        hf_config = getattr(bridge.hf_pretrained, "config", bridge.hf_pretrained)
        if hasattr(hf_config, "_name_or_path"):
            hf_config._name_or_path = hf_path
        return hf_config, bridge.transformer_config
    else:
        hf_config: PretrainedConfig = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=hf_path,
            trust_remote_code=True,
        )
        assert len(hf_config.architectures) == 1
        architecture = hf_config.architectures[0]
        if architecture == "Qwen3ForCausalLM":
            return hf_config, hf_to_mcore_config_qwen3_dense(hf_config, dtype)
        elif architecture in (
            "BailingMoeV2_5ForCausalLM",
            "BailingMoeLinearForCausalLM",
            "BailingHybridForCausalLM",
        ):
            return hf_config, hf_to_mcore_config_bailing_moe(hf_config, dtype)
        else:
            raise ValueError(
                f"Architecture not registered for config conversion: {architecture}."
            )


def make_mcore_layer_specs(hf_config: PretrainedConfig, tf_config: TransformerConfig):
    assert len(hf_config.architectures) == 1
    architecture = hf_config.architectures[0]
    if architecture == "Qwen3ForCausalLM":
        return make_mcore_layer_specs_qwen3_dense(tf_config, use_te=True)
    elif architecture in (
        "BailingMoeV2_5ForCausalLM",
        "BailingMoeLinearForCausalLM",
        "BailingHybridForCausalLM",
    ):
        return make_mcore_layer_specs_bailing_moe(tf_config, hf_config, use_te=True)
    else:
        raise ValueError(
            f"Architecture not registered for config conversion: {architecture}."
        )


def make_mcore_model(
    hf_config: PretrainedConfig,
    tf_config: TransformerConfig,
    mcore_config: MegatronEngineConfig | None = None,
    bridge: Bridge | Any | None = None,
    bridge_type: str = "mbridge",
    is_critic: bool = False,
    use_lora: bool = False,
    enable_mtp: bool = False,
) -> list[GPTModel | DDP]:
    if bridge is not None and bridge_type == "mbridge":
        # Patch get_gpt_mtp_block_spec before mbridge calls it so that a
        # TransformerConfig passed as ``spec`` is auto-converted to the
        # correct ModuleSpec type expected by megatron-core.
        if enable_mtp:
            _ensure_mtp_spec_compat()
            logger.info(
                "[MTPTrain] Applied MTP spec compatibility patch before mbridge model creation."
            )

        models = bridge.get_model(
            # TODO: Add DDP options when supporting training
            wrap_with_ddp=mcore_config.wrap_with_ddp,
            ddp_config=dataclasses.asdict(mcore_config.ddp),
            use_torch_fsdp2=mcore_config.use_torch_fsdp2,
            use_custom_fsdp=mcore_config.use_custom_fsdp,
            fp16=tf_config.fp16,
            bf16=tf_config.bf16,
            use_precision_aware_optimizer=mcore_config.use_precision_aware_optimizer,
            overlap_param_gather_with_optimizer_step=mcore_config.overlap_param_gather_with_optimizer_step,
        )
        models = list(models)

        # Replace output_layer with ValueHead for critic models
        if is_critic:
            for model in models:
                _model = unwrap_to_gpt_model(model)
                _replace_output_layer_with_value_head(_model, tf_config)

        return models

    if bridge is not None and bridge_type == "megatron-bridge":
        provider = bridge.to_megatron_provider(load_weights=False)
        vpp_size = mcore_config.virtual_pipeline_parallel_size or 0

        provider.tensor_model_parallel_size = mpu.get_tensor_model_parallel_world_size()
        provider.pipeline_model_parallel_size = (
            mpu.get_pipeline_model_parallel_world_size()
        )
        provider.virtual_pipeline_model_parallel_size = (
            vpp_size if vpp_size > 1 else None
        )
        provider.context_parallel_size = mpu.get_context_parallel_world_size()
        provider.expert_model_parallel_size = mpu.get_expert_model_parallel_world_size()
        provider.expert_tensor_parallel_size = (
            mpu.get_expert_tensor_parallel_world_size()
        )
        provider.sequence_parallel = mpu.get_tensor_model_parallel_world_size() > 1
        provider.pipeline_dtype = tf_config.params_dtype

        provider.recompute_granularity = mcore_config.recompute_granularity
        provider.recompute_method = mcore_config.recompute_method
        provider.recompute_num_layers = mcore_config.recompute_num_layers
        provider.distribute_saved_activations = (
            mcore_config.distribute_saved_activations
        )
        provider.recompute_modules = mcore_config.recompute_modules

        provider.account_for_embedding_in_pipeline_split = False
        provider.account_for_loss_in_pipeline_split = False

        # LoRA params are injected after model materialization and do not carry
        # Megatron main_grad buffers required by fused grad accumulation kernels.
        if use_lora:
            provider.gradient_accumulation_fusion = False

        # Keep these four flags aligned with mbridge base defaults.
        provider.variable_seq_lengths = True
        logger.warning(
            "Ignoring mcore_config.moe_token_dispatcher_type=%s for bridge_type='megatron-bridge'; "
            "using 'alltoall' and variable_seq_lengths=True.",
            mcore_config.moe_token_dispatcher_type,
        )
        provider.moe_token_dispatcher_type = "alltoall"
        provider.batch_p2p_comm = False
        provider.overlap_p2p_comm = (
            vpp_size > 1 and provider.pipeline_model_parallel_size > 1
        )

        # Aligning tf config settings with provider for consistency.
        tf_config.variable_seq_lengths = provider.variable_seq_lengths
        tf_config.moe_token_dispatcher_type = provider.moe_token_dispatcher_type
        tf_config.batch_p2p_comm = provider.batch_p2p_comm
        tf_config.overlap_p2p_comm = provider.overlap_p2p_comm

        provider.finalize()

        ddp_config = MCoreDDPConfig(**dataclasses.asdict(mcore_config.ddp))
        if use_lora:
            ddp_config.use_distributed_optimizer = False
            ddp_config.overlap_grad_reduce = False
            ddp_config.overlap_param_gather = False

        models = provider.provide_distributed_model(
            ddp_config=ddp_config,
            fp16=tf_config.fp16,
            bf16=tf_config.bf16,
            use_megatron_fsdp=mcore_config.use_custom_fsdp,
            use_torch_fsdp2=mcore_config.use_torch_fsdp2,
            wrap_with_ddp=mcore_config.wrap_with_ddp,
            overlap_param_gather_with_optimizer_step=mcore_config.overlap_param_gather_with_optimizer_step,
        )
        models = list(models)

        if is_critic:
            for model in models:
                _model = unwrap_to_gpt_model(model)
                _replace_output_layer_with_value_head(_model, tf_config)

        return models

    else:
        if (
            mcore_config is not None
            and mcore_config.virtual_pipeline_parallel_size is not None
            and mcore_config.virtual_pipeline_parallel_size > 1
        ):
            raise NotImplementedError(
                "Virtual pipeline parallelism requires mbridge-backed models."
            )
        transformer_layer_spec = make_mcore_layer_specs(hf_config, tf_config)

        # Build MTP block spec if MTP is configured
        mtp_block_spec = None
        mtp_num_layers = getattr(tf_config, "mtp_num_layers", 0)
        if mtp_num_layers > 0:
            try:
                from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec
                mtp_block_spec = get_gpt_mtp_block_spec(
                    tf_config, transformer_layer_spec, use_transformer_engine=True
                )
                logger.info(
                    f"[MTPTrain] Created MTP block spec with {mtp_num_layers} layers"
                )
            except ImportError:
                logger.warning(
                    "[MTPTrain] Cannot import get_gpt_mtp_block_spec from megatron.core. "
                    "MTP layers will not be created. Ensure megatron-core >= 0.11.0."
                )
        rope_scaling_args = {}
        if hf_config.rope_scaling is not None:
            if hf_config.rope_scaling["type"] != "linear":
                raise NotImplementedError(
                    f"Rope scaling type {hf_config.rope_scaling['type']} not supported yet."
                )
            rope_scaling_args["seq_len_interpolation_factor"] = hf_config.rope_scaling[
                "factor"
            ]

        model = GPTModel(
            config=tf_config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=hf_config.vocab_size,
            max_sequence_length=hf_config.max_position_embeddings,
            pre_process=True,  # TODO: pipeline parallel
            post_process=True,  # TODO: pipeline parallel
            share_embeddings_and_output_weights=False,  # TODO: implement share output weights
            position_embedding_type="rope",
            rotary_base=hf_config.rope_theta,
            **rope_scaling_args,
            # vp_stage=None TODO: virtual pipeline parallel
            **({"mtp_block_spec": mtp_block_spec} if mtp_block_spec is not None else {}),
        )

        # Replace output_layer with ValueHead for critic models
        if is_critic:
            _replace_output_layer_with_value_head(model, tf_config)

        if mcore_config.wrap_with_ddp:
            ddp_config = MCoreDDPConfig(**dataclasses.asdict(mcore_config.ddp))
            wrapped = DDP(
                config=tf_config,
                ddp_config=ddp_config,
                module=model,
                disable_bucketing=False,
            )
            return [wrapped]
        return [model]
