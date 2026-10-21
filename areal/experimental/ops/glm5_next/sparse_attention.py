# SPDX-License-Identifier: Apache-2.0
# Adapted from radixark/miles (Apache-2.0), commit ee36bb2.
# Source: miles/kernels/attention/dsa/sparse_attention.py

"""Sparse attention over top-k selected KV rows (DeepSeek Sparse Attention core).

q [B, S, H, d_v + d_tail] bf16, kv [B, S_kv, G, d_v + d_tail] bf16, indices [B, S, G, topk] int32 with
-1 as padding, attn_sink [H] fp32 or None. GLM-5 / DeepSeek-V3.2 pass a RoPE tail (d_tail > 0) and
no sink; DeepSeek-V4 passes a single latent (d_tail = 0, G = 1) and a learnable per-head sink.

The forward runs FlashMLA's sparse prefill kernel (single latent group, d_v = 512; query heads are
zero-padded up to 64 or 128) or the TileLang kernel; the backward is always the TileLang kernel. It takes
the log-sum-exp in log2 space with the sink folded in, which is what the TileLang forward emits and what
FlashMLA's is converted to. Which forward runs and how both kernels are tiled is a SparseAttentionConfig:
measured rows live in _TUNED, keyed by (arch major, RoPE tail, TP-local heads); callers may pass their own.
"""

from dataclasses import dataclass, replace
from typing import Literal

import torch

try:
    from flash_mla import flash_mla_sparse_fwd
except ImportError:
    flash_mla_sparse_fwd = None

_LOG2_E = 1.4426950408889634
_TILELANG_TOPK_MULTIPLE = 64
_FLASH_MLA_TOPK_MULTIPLE = 128
_FLASH_MLA_HEADS = (64, 128)
_FLASH_MLA_HEAD_DIMS = (512, 576)
_FLASH_MLA_ARCH_MAJORS = (9, 10)


@dataclass(frozen=True)
class SparseAttentionConfig:
    forward_backend: Literal["flash_mla", "tilelang"] = "flash_mla"
    forward_threads: int = 256
    backward_stages: int = 0
    backward_split_store: int = 2


_TILELANG_SMALL_HEADS = SparseAttentionConfig(
    forward_backend="tilelang", forward_threads=128
)

# Measured per layer at 16k / 64k tokens on H200 (9) and B300 / GB300 (10, identical winners) for GLM-5
# (RoPE tail 64, topk 2048) and DeepSeek-V4.1 (no tail, topk 640). FlashMLA pads heads to 64.
_TUNED = {
    (10, True, 8): SparseAttentionConfig(backward_stages=2, backward_split_store=1),
    (10, True, 16): SparseAttentionConfig(backward_stages=2, backward_split_store=1),
    (10, True, 32): SparseAttentionConfig(backward_stages=2, backward_split_store=4),
    (10, True, 64): SparseAttentionConfig(backward_stages=1, backward_split_store=4),
    (10, False, 8): SparseAttentionConfig(backward_stages=1),
    (10, False, 16): SparseAttentionConfig(backward_stages=1),
    (10, False, 32): SparseAttentionConfig(backward_stages=2, backward_split_store=4),
    (10, False, 64): SparseAttentionConfig(backward_stages=1, backward_split_store=4),
    (9, True, 8): SparseAttentionConfig(backward_stages=1),
    (9, True, 16): SparseAttentionConfig(backward_stages=1),
    (9, True, 32): SparseAttentionConfig(backward_stages=1),
    (9, True, 64): SparseAttentionConfig(backward_split_store=4),
    (9, False, 8): replace(
        _TILELANG_SMALL_HEADS, backward_stages=1, backward_split_store=4
    ),
    (9, False, 16): replace(
        _TILELANG_SMALL_HEADS, backward_stages=1, backward_split_store=4
    ),
    (9, False, 32): SparseAttentionConfig(backward_stages=1, backward_split_store=4),
    (9, False, 64): SparseAttentionConfig(backward_split_store=4),
}


def _pad_topk_block(indices, multiple: int):
    topk = indices.shape[-1]
    padded = (topk + multiple - 1) // multiple * multiple
    if padded == topk:
        return indices.contiguous()
    return torch.nn.functional.pad(indices, (0, padded - topk), value=-1).contiguous()


def _default_config(q, kv, d_v: int) -> SparseAttentionConfig:
    arch_major = torch.cuda.get_device_capability(q.device)[0]
    heads_per_group = q.shape[2] // kv.shape[2]
    heads_bucket = min(max(8, 1 << (heads_per_group - 1).bit_length()), 64)
    config = _TUNED.get(
        (arch_major, q.shape[-1] > d_v, heads_bucket), SparseAttentionConfig()
    )
    flash_mla_supported = (
        flash_mla_sparse_fwd is not None
        and arch_major in _FLASH_MLA_ARCH_MAJORS
        and d_v == 512
        and q.shape[-1] in _FLASH_MLA_HEAD_DIMS
        and kv.shape[2] == 1
        and q.shape[2] <= _FLASH_MLA_HEADS[-1]
    )
    if config.forward_backend == "flash_mla" and not flash_mla_supported:
        config = replace(
            config,
            forward_backend="tilelang",
            forward_threads=128 if heads_bucket < 64 else 256,
        )
    return config


def _flash_mla_forward(q, kv, indices, attn_sink, sm_scale):
    batch, seq_len, heads, _ = q.shape
    seq_len_kv = kv.shape[1]
    padded_heads = next(h for h in _FLASH_MLA_HEADS if h >= heads)
    q = q.reshape(batch * seq_len, heads, -1)
    sink = attn_sink
    if padded_heads != heads:
        q = torch.nn.functional.pad(q, (0, 0, 0, padded_heads - heads))
        sink = (
            None
            if attn_sink is None
            else torch.nn.functional.pad(attn_sink, (0, padded_heads - heads))
        )
    indices = _pad_topk_block(indices, _FLASH_MLA_TOPK_MULTIPLE)
    if batch == 1:
        flat_indices = indices.view(seq_len, 1, -1)
    else:
        offsets = (
            torch.arange(batch, device=indices.device, dtype=indices.dtype).view(
                batch, 1, 1, 1
            )
            * seq_len_kv
        )
        flat_indices = torch.where(indices >= 0, indices + offsets, -1).view(
            batch * seq_len, 1, -1
        )
    out, _, lse = flash_mla_sparse_fwd(
        q,
        kv.reshape(batch * seq_len_kv, 1, -1),
        flat_indices,
        sm_scale,
        d_v=512,
        attn_sink=sink,
    )
    out, lse = out[:, :heads].contiguous(), lse[:, :heads] * _LOG2_E
    if attn_sink is not None:
        lse = torch.logaddexp2(lse, attn_sink.float().view(1, heads) * _LOG2_E)
    return out.reshape(batch, seq_len, heads, -1), lse.reshape(
        batch, seq_len, heads
    ).contiguous()


class _SparseAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, indices, attn_sink, sm_scale, d_v, config):
        q, kv, indices = (
            q.contiguous(),
            kv.contiguous(),
            _pad_topk_block(indices, _TILELANG_TOPK_MULTIPLE),
        )
        kv_indices = indices.clamp(min=0)
        if config.forward_backend == "flash_mla":
            out, lse = _flash_mla_forward(q, kv, indices, attn_sink, sm_scale)
        else:
            # tilelang is GPU-only; importing it lazily keeps the package importable on CPU
            from areal.experimental.ops.glm5_next.tilelang.sparse_attention_fwd import (
                sparse_attention_fwd,
            )

            out, lse = sparse_attention_fwd(
                q,
                kv,
                indices,
                kv_indices,
                attn_sink,
                d_v,
                sm_scale=sm_scale,
                threads=config.forward_threads,
            )
        ctx.save_for_backward(q, kv, indices, kv_indices, attn_sink, out, lse)
        ctx.sm_scale = sm_scale
        ctx.d_v = d_v
        ctx.config = config
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, kv, indices, kv_indices, attn_sink, out, lse = ctx.saved_tensors
        from areal.experimental.ops.glm5_next.tilelang.sparse_attention_bwd import (
            sparse_attention_bwd,
        )

        dq, dkv, delta = sparse_attention_bwd(
            q,
            kv,
            out,
            grad_out.contiguous(),
            indices,
            kv_indices,
            lse,
            ctx.d_v,
            sm_scale=ctx.sm_scale,
            num_stages=ctx.config.backward_stages,
            split_store=ctx.config.backward_split_store,
        )
        d_sink = None
        if attn_sink is not None:
            p_sink = torch.exp2(attn_sink.float() * _LOG2_E - lse)
            d_sink = -(delta * p_sink).sum(dim=(0, 1))
        return dq, dkv, None, d_sink, None, None, None


def sparse_attention(
    q,
    kv,
    indices,
    sm_scale: float,
    d_v: int | None = None,
    attn_sink=None,
    config: SparseAttentionConfig | None = None,
) -> torch.Tensor:
    """Returns out [B, S, H, d_v] bf16. ``d_v`` defaults to the full head dim (no RoPE tail).
    ``config`` defaults to the measured row for this arch and shape."""
    if d_v is None:
        d_v = q.shape[-1]
    if config is None:
        config = _default_config(q, kv, d_v)
    return _SparseAttention.apply(q, kv, indices, attn_sink, sm_scale, d_v, config)
