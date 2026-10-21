# SPDX-License-Identifier: Apache-2.0

"""GLM-5.3 mbridge conversion, including lossless mHC scale export."""

import copy

import torch
from mbridge.core import LLMBridge, register_model
from mbridge.models import DeepseekV3Bridge
from megatron.core.transformer import MLATransformerConfig
from transformers import PretrainedConfig

from areal.models.mcore.glm5_next import (
    check_glm5_next_capabilities,
    make_glm5_next_layer_spec,
)
from areal.models.mcore.glm5_next_config import (
    Glm5NextConfig,
    glm5_next_quantization_config,
    validate_glm5_next_runtime,
)


def glm5_next_keep_fp32(name: str) -> bool:
    return (
        "_hyper_connection." in name
        or name.endswith(
            (".A_log", ".dt_bias", ".expert_bias", ".index_kpool_compress_ape")
        )
        or ".k_norm." in name
    )


@register_model("glm5_next")
class Glm5NextBridge(DeepseekV3Bridge):
    TransformerConfigClass = MLATransformerConfig

    _ATTENTION_MAPPING = {
        **DeepseekV3Bridge._ATTENTION_MAPPING,
        **{
            "self_attention.kda." + name: [
                "model.layers.{layer_number}.self_attn." + name
            ]
            for name in (
                "q_proj.weight",
                "k_proj.weight",
                "v_proj.weight",
                "b_proj.weight",
                "f_a_proj.weight",
                "f_b_proj.weight",
                "g_a_proj.weight",
                "g_b_proj.weight",
                "A_log",
                "dt_bias",
                "o_norm.weight",
                "o_proj.weight",
            )
        },
        "self_attention.kda.conv1d.weight": [
            "model.layers.{layer_number}.self_attn." + name + "_conv1d.weight"
            for name in ("q", "k", "v")
        ],
        **{
            "self_attention." + name: [
                "model.layers.{layer_number}.self_attn.indexer." + name
            ]
            for name in (
                "wq_b.weight",
                "wk.weight",
                "k_norm.weight",
                "k_norm.bias",
                "weights_proj.weight",
                "index_kpool_compress_gate",
                "index_kpool_compress_ape",
            )
        },
    }

    def __init__(self, hf_config, *args, **kwargs):
        self.glm_config = Glm5NextConfig.from_hf(hf_config)
        # Retain the root config, notably FP8 quantization metadata and the HF
        # architecture. PretrainedConfig works with older Transformers releases.
        hf_config = copy.deepcopy(hf_config)
        if getattr(hf_config, "quantization_config", None):
            hf_config.quantization_config = glm5_next_quantization_config(
                hf_config.quantization_config
            )
        hf_config.text_config = PretrainedConfig.from_dict(self.glm_config.text)
        hf_config.tie_word_embeddings = self.glm_config.text.get(
            "tie_word_embeddings", False
        )
        super().__init__(hf_config, *args, **kwargs)

    def _build_config(self):
        check_glm5_next_capabilities(self.TransformerConfigClass)
        config = self._build_base_config(
            text_config_key="text_config", **self.glm_config.mcore_args()
        )
        validate_glm5_next_runtime(config)
        return config

    def _get_gptmodel_args(self):
        t = self.glm_config.text
        return dict(
            vocab_size=t["vocab_size"],
            max_sequence_length=t["max_position_embeddings"],
            position_embedding_type="rope",
            rotary_base=10000.0,
        )

    # DeepSeek's provider materializes MTP from HF metadata and bypasses the
    # common vocab padding logic. GLM uses the base provider with MTP disabled.
    _model_provider = LLMBridge._model_provider

    def _get_transformer_layer_spec(self, vp_stage=None):
        self.has_vp_stage = True
        return make_glm5_next_layer_spec(self.config, self.hf_config, vp_stage)

    def _weight_name_mapping_mcore_to_hf(self, name: str) -> list[str]:
        if "_hyper_connection." in name:
            parts = name.split(".")
            layer = parts[2]
            site = "attn" if parts[3] == "self_attention_hyper_connection" else "ffn"
            field = {"mapping_proj.weight": "fn", "bias": "base", "scale": "scale"}.get(
                ".".join(parts[4:])
            )
            if field is None:
                raise NotImplementedError(f"Unsupported GLM mHC parameter: {name}")
            return [f"model.language_model.layers.{layer}.hc_{site}_{field}"]
        names = super()._weight_name_mapping_mcore_to_hf(name)
        return [
            "model.language_model." + n[len("model.") :]
            if n.startswith("model.")
            else n
            for n in names
        ]

    def _weight_to_mcore_format(
        self, name: str, weights: list[torch.Tensor]
    ) -> torch.Tensor:
        if name.endswith("self_attention.kda.conv1d.weight"):
            if len(weights) != 3 or len({tuple(w.shape) for w in weights}) != 1:
                raise ValueError(
                    "KDA convolution requires equally shaped Q/K/V tensors"
                )
            result = torch.cat(weights, dim=0).contiguous()
            return result.to(self.dtype) if self.dtype is not None else result
        if glm5_next_keep_fp32(name):
            if len(weights) != 1:
                raise ValueError(f"Expected one FP32 tensor for {name}")
            return weights[0].float()
        return super()._weight_to_mcore_format(name, weights)

    def _weight_to_hf_format(self, name: str, weight: torch.Tensor):
        if name.endswith("self_attention.kda.conv1d.weight"):
            if weight.shape[0] % 3:
                raise ValueError(
                    "Fused KDA convolution size must be divisible by three"
                )
            return self._weight_name_mapping_mcore_to_hf(name), list(
                weight.chunk(3, dim=0)
            )
        return super()._weight_to_hf_format(name, weight)

    def _weight_merge_across_tp(self, name, weights, param):
        if not getattr(param, "tensor_model_parallel", False):
            return weights[0]
        return super()._weight_merge_across_tp(name, weights, param)
