# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint

import realhf.base.constants as constants
import realhf.base.logging as logging
from realhf.impl.model.parallelism.tensor_parallel.modules import RowParallelLinear
from realhf.impl.model.utils.functional import (
    apply_rotary_varlen,
    compute_varlen_position_indices,
    torch_attn_func,
)

from .mlp import GemmaRMSNorm, LayerNormQKVLinear, LlamaRMSNorm
from .rotary import RotaryEmbedding

try:
    from flash_attn import (
        flash_attn_func,
        flash_attn_varlen_func,
        flash_attn_with_kvcache,
    )
except ModuleNotFoundError:
    pass

logger = logging.getLogger("Attention")


class CausalSelfAttentionLayer(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        n_kv_heads: int,
        n_q_heads: int,
        head_dim: int,
        resid_pdrop: float,
        attn_pdrop: float,
        layer_index: int,
        layer_norm_epsilon: float,
        scale_attn_by_inverse_layer_idx: bool,
        scale_attn_weights: bool,
        # llama does not require attention bias
        use_attention_bias: bool,
        use_attn_proj_bias: bool,
        # layer norm type is special for llama
        layer_norm_type: Optional[str] = None,
        # opt applies layer norm after attn
        do_layernorm_before: bool = True,
        # qk layer norm (Qwen3)
        qk_layernorm: bool = False,
        # rotary embedding
        apply_rotary: bool = False,
        rotary_base: float = 10000.0,
        rotary_interleaved: bool = False,  # False for LLaMA, GPT-neoX; True for GPT-J
        rotary_scaling: Optional[float] = None,
        rotary_scaling_type: Optional[str] = None,
        rotary_special_impl: Optional[str] = None,
        # device and dtype
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        if dtype is None:
            dtype = torch.float16
        assert hidden_dim % head_dim == 0, (hidden_dim, head_dim)
        self.c_attn = LayerNormQKVLinear(
            input_dim=hidden_dim,
            head_dim=head_dim,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            layer_norm_epsilon=layer_norm_epsilon,
            layer_norm_type=layer_norm_type,
            use_attention_bias=use_attention_bias,
            do_layernorm_before=do_layernorm_before,
            dtype=dtype,
            device=device,
            layer_index=layer_index,
        )

        if constants.tensor_parallel_world_size() > 1:
            self.c_proj = RowParallelLinear(
                n_q_heads * head_dim,
                hidden_dim,
                bias=use_attn_proj_bias,
                gradient_accumulation_fusion=constants.gradient_accumulation_fusion(),
                dtype=dtype,
                device=device,
            )
        else:
            self.c_proj = nn.Linear(
                n_q_heads * head_dim,
                hidden_dim,
                bias=use_attn_proj_bias,
                dtype=dtype,
                device=device,
            )

        self.qk_layernorm = qk_layernorm
        if qk_layernorm:
            if layer_norm_type is None:
                layer_norm_fn = nn.LayerNorm
            elif layer_norm_type == "rms":
                layer_norm_fn = LlamaRMSNorm
            elif layer_norm_type == "gemma":
                layer_norm_fn = GemmaRMSNorm
            self.q_ln = layer_norm_fn(
                head_dim, eps=layer_norm_epsilon, dtype=dtype, device=device
            )
            self.k_ln = layer_norm_fn(
                head_dim, eps=layer_norm_epsilon, dtype=dtype, device=device
            )

        self.resid_dropout = nn.Dropout(resid_pdrop)

        self.attn_pdrop = attn_pdrop

        self.applied_attn_pdrop = attn_pdrop

        self.apply_rotary = apply_rotary
        self.rotary_interleaved = rotary_interleaved
        if self.apply_rotary:
            # Will layzily update the cache sequence length of cache.,
            # so we don't need to pass in max_positions.
            self.rotary_emb = RotaryEmbedding(
                head_dim,
                base=rotary_base,
                scale_factor=rotary_scaling,
                scale_type=rotary_scaling_type,
                interleaved=rotary_interleaved,
                device=device,
                special_impl=rotary_special_impl,
            )
            self.rotary_special_impl = rotary_special_impl

        # constant
        self.nq = n_q_heads
        self.nkv = n_kv_heads
        if self.nq % self.nkv != 0:
            raise ValueError(
                f"n_kv_heads ({self.nkv}) must divide n_q_heads ({self.nq})."
            )
        self.d = head_dim

        self.layer_index = layer_index

        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        self.scale_attn_weights = scale_attn_weights

    def train(self, mode: bool):
        if not mode:
            self.applied_attn_pdrop = 0.0
        else:
            self.applied_attn_pdrop = self.attn_pdrop
        super().train(mode)
        return self

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        k_cache: Optional[torch.Tensor] = None,
        v_cache: Optional[torch.Tensor] = None,
        cache_seqlens: Optional[Union[int, torch.Tensor]] = None,
        max_seqlen: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # input shape: [bs, seq, hidden_dim]

        # NOTE: we must ensure the passed-in argument is an interger
        # if we convert the argument to implicitly when calling rotary embedding or flash-attn,
        # aten::item will be called, which will cause a device-host sync and slow down performance.
        assert max_seqlen is None or isinstance(max_seqlen, int), type(max_seqlen)
        assert cu_seqlens is None or cu_seqlens.dtype == torch.int32

        # default upcast, scale
        if self.scale_attn_by_inverse_layer_idx:
            unscale = self.layer_index + 1
            scale_factor = unscale**-1
        else:
            unscale = 1.0
            scale_factor = 1
        if self.scale_attn_weights:
            scale_factor /= self.d**0.5

        q, k, v = self.c_attn(hidden_states)

        if self.qk_layernorm:
            q = self.q_ln(q)
            k = self.k_ln(k)

        if self.apply_rotary and (k_cache is None or str(q.device) == "cpu"):
            # otherwise, we input rotary cos/sin directly into flash_attn_with_kvcache
            rotary_cache_len = max_seqlen
            if k_cache is not None and str(q.device) == "cpu":
                rotary_cache_len = k_cache.shape[1]
            self.rotary_emb._update_cos_sin_cache(rotary_cache_len, q.device, q.dtype)
            rotary_indices = compute_varlen_position_indices(q.shape[0], cu_seqlens)
            qk = apply_rotary_varlen(
                torch.cat([q, k], dim=-2),
                cos=self.rotary_emb._cos_cached,
                sin=self.rotary_emb._sin_cached,
                cu_seqlens=cu_seqlens,
                interleaved=self.rotary_emb.interleaved,
                rotary_indices=rotary_indices,
                seqlen_offsets=cache_seqlens,
                special_impl=self.rotary_special_impl,
            )
            q, k = qk.split((q.shape[-2], k.shape[-2]), dim=-2)
        elif self.apply_rotary:
            self.rotary_emb._update_cos_sin_cache(
                k_cache.shape[1], device=q.device, dtype=q.dtype
            )
            # Rotary cos/sin will be automatically offset by cache_seqlens in flash_attn.
            rotary_cos, rotary_sin = (
                self.rotary_emb._cos_cached,
                self.rotary_emb._sin_cached,
            )
        else:
            rotary_cos = rotary_sin = None

        if str(q.device) == "cpu":
            cu_seqlens_k = cu_seqlens
            max_seqlen_k = max_seqlen
            if k_cache is not None:
                new_k, new_v = [], []
                for i, cache_len in enumerate(cache_seqlens):
                    assert k.shape[0] == cu_seqlens.shape[0] - 1, (k.shape, cu_seqlens)
                    k_cache[i, cache_len] = k[i]
                    new_k.append(k_cache[i, : cache_len + 1])
                    v_cache[i, cache_len] = v[i]
                    new_v.append(v_cache[i, : cache_len + 1])
                k = torch.cat(new_k, dim=0)
                v = torch.cat(new_v, dim=0)
                cu_seqlens_k = torch.nn.functional.pad(
                    (cache_seqlens + 1).cumsum(0), (1, 0)
                )
                max_seqlen_k = max(cache_seqlens) + 1
                cu_seqlens = torch.arange(
                    cu_seqlens_k.shape[0], device=k.device, dtype=k.dtype
                )
                max_seqlen = 1
            # Use vanilla pytorch attention, for debugging.
            hidden_states = torch_attn_func(
                q,
                k,
                v,
                causal=True,
                cu_seqlens_q=cu_seqlens,
                max_seqlen_q=max_seqlen,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_k=max_seqlen_k,
                dropout_p=self.applied_attn_pdrop,
                softmax_scale=scale_factor,
                upcast_unscale=unscale,
            )
        elif k_cache is not None:
            # k_cache/v_cache shape: [bs, max_seq, n_kv_heads, head_dim]
            if cache_seqlens is None:
                raise RuntimeError(
                    "cache_seqlens must be provided if kv_cache is not None."
                )
            q = q.unsqueeze(1)
            k = k.unsqueeze(1)
            v = v.unsqueeze(1)
            # k_cache and v_cache will be modified in-place.
            hidden_states = flash_attn_with_kvcache(
                q,
                k_cache,
                v_cache,
                k=k,
                v=v,
                cache_seqlens=cache_seqlens,
                softmax_scale=scale_factor,
                causal=False,  # True or False doesn't matter because seqlen=1
                rotary_cos=rotary_cos,
                rotary_sin=rotary_sin,
                rotary_interleaved=self.rotary_interleaved,
            )
            hidden_states = hidden_states.squeeze(1)
        elif cu_seqlens is not None:
            assert max_seqlen is not None
            assert len(q.shape) == 3
            hidden_states = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen,
                dropout_p=self.applied_attn_pdrop,
                softmax_scale=scale_factor,
                causal=True,
            )
        else:
            raise NotImplementedError(
                "Don't know which attention implementation to use."
            )
        hidden_states = self.c_proj(hidden_states.flatten(start_dim=-2))
        hidden_states = self.resid_dropout(hidden_states)
        return hidden_states, k, v
