# SPDX-License-Identifier: Apache-2.0

"""GLM-5.3 layer specifications for a Megatron runtime with native mHC slots."""

import copy
import dataclasses

from areal.models.mcore.glm5_next_config import (
    Glm5NextConfig,
    validate_glm5_next_runtime,
)


def check_glm5_next_capabilities(config_class) -> None:
    from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
    from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules

    required = {
        "enable_hyper_connections",
        "num_residual_streams",
        "mhc_sinkhorn_iterations",
        "use_fused_mhc",
        "activation_func_clamp_value",
    }
    missing = required - {f.name for f in dataclasses.fields(config_class)}
    if (
        missing
        or not hasattr(TransformerBlockSubmodules, "hc_head_contraction")
        or not hasattr(TransformerLayerSubmodules, "self_attention_hyper_connection")
    ):
        raise RuntimeError(
            "GLM-5.3 requires a Megatron runtime with mHC layer slots, configurable "
            "hc_head_contraction and SwiGLU clamping (e.g. radixark/Megatron-LM "
            "e8f57451). Stock megatron-core 0.19 is insufficient. "
            f"Missing config fields: {sorted(missing)}"
        )


def make_glm5_next_layer_spec(config, hf_config, vp_stage=None):
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
    from megatron.core.transformer.spec_utils import ModuleSpec
    from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

    from areal.models.mcore.glm5_next_dsa import Glm5NextDSAAttention
    from areal.models.mcore.glm5_next_hyper_connection import (
        Glm5NextHyperConnection,
        Glm5NextMeanOutputContraction,
    )
    from areal.models.mcore.glm5_next_kda import Glm5NextKDAAttention

    validate_glm5_next_runtime(config)
    check_glm5_next_capabilities(type(config))
    glm = Glm5NextConfig.from_hf(hf_config)
    spec = get_gpt_decoder_block_spec(
        config, use_transformer_engine=True, vp_stage=vp_stage
    )
    spec.hc_head_contraction = ModuleSpec(module=Glm5NextMeanOutputContraction)
    offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
    for index, original in enumerate(spec.layer_specs):
        layer = copy.deepcopy(original)
        layer.submodules.self_attention_hyper_connection = ModuleSpec(
            module=Glm5NextHyperConnection
        )
        layer.submodules.mlp_hyper_connection = ModuleSpec(
            module=Glm5NextHyperConnection
        )
        if glm.layer_types[offset + index] == "linear_attention":
            layer.submodules.self_attention = ModuleSpec(
                module=Glm5NextKDAAttention, params={"glm_config": glm}
            )
        else:
            layer.submodules.self_attention.module = Glm5NextDSAAttention
            layer.submodules.self_attention.params = {
                **(layer.submodules.self_attention.params or {}),
                "glm_config": glm,
            }
        spec.layer_specs[index] = layer
    return spec
