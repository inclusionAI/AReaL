# Adapted from torchtitan: torchtitan/models/qwen3/model/args.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from areal.experimental.models.archon.base import BaseModelArgs

if TYPE_CHECKING:
    from transformers import PretrainedConfig


@dataclass
class Qwen3ModelArgs(BaseModelArgs):
    """Model arguments for Qwen3. Default values are for Qwen3-0.6B."""

    dim: int = 1024
    n_layers: int = 28
    n_heads: int = 16
    n_kv_heads: int = 8
    vocab_size: int = 151936
    head_dim: int = 128
    hidden_dim: int = 3072

    norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    qk_norm: bool = True
    max_seq_len: int = 4096
    depth_init: bool = True

    eos_id: int = 151645
    enable_weight_tying: bool = False
    is_critic: bool = False
    num_labels: int = 1
    moe_enabled: bool = False
    moe_inter_dim: int = 768
    moe_args: dict = field(default_factory=dict)

    @classmethod
    def from_hf_config(
        cls,
        hf_config: PretrainedConfig,
        is_critic: bool = False,
        **kwargs,
    ) -> Qwen3ModelArgs:
        moe_enabled = hasattr(hf_config, "num_experts") and hf_config.num_experts > 1

        return cls(
            dim=hf_config.hidden_size,
            n_layers=hf_config.num_hidden_layers,
            n_heads=hf_config.num_attention_heads,
            n_kv_heads=getattr(
                hf_config, "num_key_value_heads", hf_config.num_attention_heads
            ),
            vocab_size=hf_config.vocab_size,
            head_dim=getattr(
                hf_config,
                "head_dim",
                hf_config.hidden_size // hf_config.num_attention_heads,
            ),
            hidden_dim=hf_config.intermediate_size,
            norm_eps=hf_config.rms_norm_eps,
            rope_theta=getattr(hf_config, "rope_theta", 1000000.0),
            qk_norm=getattr(hf_config, "qk_norm", True),
            max_seq_len=getattr(hf_config, "max_position_embeddings", 4096),
            eos_id=getattr(hf_config, "eos_token_id", 151645),
            enable_weight_tying=getattr(hf_config, "tie_word_embeddings", False),
            is_critic=is_critic,
            moe_enabled=moe_enabled,
            moe_inter_dim=getattr(hf_config, "moe_intermediate_size", 768),
            attn_type=kwargs.get("attn_type", BaseModelArgs.attn_type),
        )


__all__ = ["Qwen3ModelArgs"]
