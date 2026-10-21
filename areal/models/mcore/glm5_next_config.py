# SPDX-License-Identifier: Apache-2.0

"""GLM-5.3 configuration validation, independent of CUDA and Transformers versions."""

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


def _as_dict(config: Any) -> dict:
    if isinstance(config, Mapping):
        return dict(config)
    return config.to_dict() if hasattr(config, "to_dict") else vars(config).copy()


def is_glm5_next(config: Any) -> bool:
    config = _as_dict(config)
    return config.get("model_type") in ("glm5_next", "glm5_next_text") or (
        "Glm5NextForConditionalGeneration" in (config.get("architectures") or [])
    )


def glm5_next_quantization_config(config: dict | None) -> dict | None:
    """Expand legacy text prefixes without losing the original FP8 exclusions."""
    if config is None:
        return None
    config = copy.deepcopy(config)
    excluded = config.get("modules_to_not_convert") or []
    config["modules_to_not_convert"] = list(
        dict.fromkeys(
            [
                *excluded,
                *(
                    "model.language_model." + name[len("model.") :]
                    for name in excluded
                    if name.startswith("model.")
                    and not name.startswith(("model.language_model.", "model.visual."))
                ),
            ]
        )
    )
    return config


@dataclass(frozen=True)
class Glm5NextConfig:
    """Validated text architecture; preserves the checkpoint's explicit layer order."""

    text: dict[str, Any]
    layer_types: tuple[str, ...]
    moe_layers: tuple[int, ...]
    linear_num_heads: int
    linear_head_dim: int
    linear_conv_kernel_dim: int
    linear_lower_bound: float

    @classmethod
    def from_hf(cls, config: Any) -> "Glm5NextConfig":
        root = _as_dict(config)
        text = _as_dict(root.get("text_config") or root)
        if not is_glm5_next(root) and not is_glm5_next(text):
            raise ValueError("Expected a glm5_next or glm5_next_text configuration")
        layers = tuple(text["layer_types"])
        count = text["num_hidden_layers"]
        if count <= 0 or len(layers) != count:
            raise ValueError("layer_types must contain num_hidden_layers entries")
        if set(layers) - {"linear_attention", "deepseek_sparse_attention"}:
            raise ValueError("Unsupported GLM-5.3 attention layer type")
        mlp = text.get("mlp_layer_types")
        if mlp is None:
            dense = text["first_k_dense_replace"]
            if not 0 <= dense <= count:
                raise ValueError("first_k_dense_replace must be within the layer count")
            mlp = ["dense"] * dense + ["sparse"] * (count - dense)
        if len(mlp) != count or set(mlp) - {"dense", "sparse"}:
            raise ValueError(
                "mlp_layer_types must specify dense/sparse for every layer"
            )
        if text.get("qk_rope_head_dim", 0) != 0 or not text.get("mla_use_nope", True):
            raise ValueError("GLM-5.3 requires NoPE (qk_rope_head_dim=0)")
        if text.get("hc_eps") != 1e-6 or not text.get("mhc", True):
            raise ValueError("GLM-5.3 requires mHC with hc_eps=1e-6")
        if text["hc_mult"] <= 0 or text["hc_sinkhorn_iters"] <= 0:
            raise ValueError("mHC streams and Sinkhorn iterations must be positive")
        pool = text["index_kpool"]
        if pool <= 1 or pool & (pool - 1) or pool > 256:
            raise ValueError("index_kpool must be a power of two between 2 and 256")
        if not text.get("index_kpool_compress") or not text.get(
            "index_kpool_always_select_tail"
        ):
            raise ValueError(
                "GLM-5.3 requires kpool compression and causal tail selection"
            )
        if text["index_topk"] < pool or text["index_topk"] % pool:
            raise ValueError("index_topk must be a positive multiple of index_kpool")
        if not 0 < text["num_experts_per_tok"] <= text["n_routed_experts"]:
            raise ValueError("num_experts_per_tok must be within the expert count")
        if text.get("scoring_func", "sigmoid") != "sigmoid" or not text.get(
            "norm_topk_prob", True
        ):
            raise ValueError("GLM-5.3 requires normalized sigmoid top-k routing")
        linear = text.get("linear_attn_config", {})

        def field(flat: str, nested: str) -> Any:
            value = text.get(flat, linear.get(nested))
            if value is None:
                raise ValueError(f"Missing KDA configuration: {flat} / {nested}")
            return value

        heads = field("linear_num_heads", "num_heads")
        dim = field("linear_head_dim", "head_dim")
        kernel = field("linear_conv_kernel_dim", "short_conv_kernel_size")
        lower = field("linear_lower_bound", "gate_lower_bound")
        if min(heads, dim, kernel) <= 0 or lower >= 0:
            raise ValueError(
                "KDA dimensions must be positive and gate lower bound negative"
            )
        return cls(
            text,
            layers,
            tuple(int(x == "sparse") for x in mlp),
            heads,
            dim,
            kernel,
            lower,
        )

    def mcore_args(self) -> dict[str, Any]:
        """Architecture settings; parallelism is supplied by the engine/mbridge."""
        t = self.text
        return dict(
            num_layers=len(self.layer_types),
            hidden_size=t["hidden_size"],
            num_attention_heads=t["num_attention_heads"],
            ffn_hidden_size=t["intermediate_size"],
            kv_channels=t["qk_head_dim"],
            layernorm_epsilon=t["rms_norm_eps"],
            attention_dropout=0.0,
            hidden_dropout=0.0,
            add_bias_linear=False,
            qk_layernorm=True,
            multi_latent_attention=True,
            q_lora_rank=t["q_lora_rank"],
            kv_lora_rank=t["kv_lora_rank"],
            qk_head_dim=t["qk_head_dim"],
            qk_pos_emb_head_dim=0,
            v_head_dim=t["v_head_dim"],
            rope_type="rope",
            rotary_base=10000.0,
            rotary_scaling_factor=1.0,
            apply_rope_fusion=False,
            num_moe_experts=t["n_routed_experts"],
            moe_router_topk=t["num_experts_per_tok"],
            moe_router_score_function="sigmoid",
            moe_router_pre_softmax=True,
            moe_router_num_groups=t.get("n_group", 1),
            moe_router_group_topk=t.get("topk_group", 1),
            moe_router_topk_scaling_factor=t["routed_scaling_factor"],
            moe_router_enable_expert_bias=True,
            moe_router_bias_update_rate=0.0,
            moe_router_dtype="fp32",
            moe_router_load_balancing_type="none",
            moe_aux_loss_coeff=0.0,
            moe_z_loss_coeff=0.0,
            moe_grouped_gemm=True,
            moe_ffn_hidden_size=t["moe_intermediate_size"],
            moe_shared_expert_intermediate_size=(
                t.get("n_shared_experts", 0) * t["moe_intermediate_size"] or None
            ),
            moe_layer_freq=list(self.moe_layers),
            activation_func_clamp_value=t.get("swiglu_limit"),
            enable_hyper_connections=True,
            num_residual_streams=t["hc_mult"],
            mhc_sinkhorn_iterations=t["hc_sinkhorn_iters"],
            use_fused_mhc=False,
            mtp_num_layers=None,
        )


def validate_glm5_next_runtime(config: Any) -> None:
    """Reject unsupported modes before allocating the model or entering collectives."""
    dtype = getattr(config, "params_dtype", None)
    if dtype is not None and str(dtype) not in ("torch.bfloat16", "bfloat16"):
        raise ValueError("GLM-5.3 sparse kernels require BF16 parameters")
    if config.context_parallel_size != 1:
        raise ValueError(
            "GLM-5.3 kpool attention currently requires context_parallel_size=1"
        )
    if getattr(config, "mtp_num_layers", None):
        raise ValueError("GLM-5.3 MTP training is not supported")
    if getattr(config, "fp8", None):
        raise ValueError("GLM-5.3 bring-up requires BF16 (FP8 is not supported)")
    if "mhc" in (getattr(config, "recompute_modules", None) or []):
        raise ValueError("GLM-5.3 supports full recompute, not the mHC recompute arena")
    if getattr(config, "pipeline_model_parallel_layout", None) is not None:
        raise ValueError("GLM-5.3 requires the standard pipeline layer layout")
    if getattr(config, "virtual_pipeline_model_parallel_size", None) not in (None, 1):
        raise ValueError("GLM-5.3 virtual pipeline parallelism is not supported yet")
