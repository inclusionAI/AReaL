# SPDX-License-Identifier: Apache-2.0
# Adapted from radixark/miles (Apache-2.0), commit ee36bb2.
# Source: miles/kernels/attention/dsa/kpool.py

"""Key-pooled lightning-indexer selection: pooled keys, pool top-k expanded to token indices, causal tail."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

SPARSE_MLA_BLOCK = 64
_SELECT_BLOCK = 256


def pool_boundaries(cu_seqlens: torch.Tensor, kpool: int) -> torch.Tensor:
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    pool_counts = torch.div(seq_lens, kpool, rounding_mode="floor")
    pool_cu_seqlens = torch.zeros_like(cu_seqlens)
    pool_cu_seqlens[1:] = torch.cumsum(pool_counts, dim=0)
    return pool_cu_seqlens


@triton.jit
def _pooled_keys_kernel(
    index_k_ptr,
    gate_ptr,
    ape_ptr,
    cu_seqlens_ptr,
    pool_cu_seqlens_ptr,
    num_seqs,
    out_ptr,
    D: tl.constexpr,
    KPOOL: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    dmask = offs_d < D
    out_offs = pid.to(tl.int64) * D + offs_d
    num_pools = tl.load(pool_cu_seqlens_ptr + num_seqs)
    if pid >= num_pools:
        zeros = tl.zeros((BLOCK_D,), dtype=tl.float32)
        tl.store(out_ptr + out_offs, zeros.to(out_ptr.dtype.element_ty), mask=dmask)
        return
    lo = 0
    hi = num_seqs
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if tl.load(pool_cu_seqlens_ptr + mid) <= pid:
            lo = mid
        else:
            hi = mid - 1
    seq_start = tl.load(cu_seqlens_ptr + lo)
    pool_base = tl.load(pool_cu_seqlens_ptr + lo)
    start = (seq_start + (pid - pool_base) * KPOOL).to(tl.int64)
    peak = tl.full((BLOCK_D,), float("-inf"), dtype=tl.float32)
    for r in tl.static_range(KPOOL):
        gate = tl.load(gate_ptr + (start + r) * D + offs_d, mask=dmask, other=0.0).to(
            tl.float32
        )
        ape = tl.load(ape_ptr + r * D + offs_d, mask=dmask, other=0.0)
        peak = tl.maximum(peak, gate + ape)
    denom = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for r in tl.static_range(KPOOL):
        gate = tl.load(gate_ptr + (start + r) * D + offs_d, mask=dmask, other=0.0).to(
            tl.float32
        )
        ape = tl.load(ape_ptr + r * D + offs_d, mask=dmask, other=0.0)
        denom = denom + libdevice.exp(gate + ape - peak)
    pooled = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for r in tl.static_range(KPOOL):
        gate = tl.load(gate_ptr + (start + r) * D + offs_d, mask=dmask, other=0.0).to(
            tl.float32
        )
        ape = tl.load(ape_ptr + r * D + offs_d, mask=dmask, other=0.0)
        key = tl.load(index_k_ptr + (start + r) * D + offs_d, mask=dmask, other=0.0).to(
            tl.float32
        )
        weight = libdevice.div_rn(libdevice.exp(gate + ape - peak), denom)
        pooled = pooled + libdevice.mul_rn(weight, key)
    tl.store(out_ptr + out_offs, pooled.to(out_ptr.dtype.element_ty), mask=dmask)


def build_pooled_keys(
    index_k: torch.Tensor,
    gate_score: torch.Tensor,
    ape: torch.Tensor,
    cu_seqlens: torch.Tensor,
    kpool: int,
) -> torch.Tensor:
    total_tokens, head_dim = index_k.shape
    max_pools = total_tokens // kpool
    if max_pools == 0:
        return index_k.new_zeros((0, head_dim))
    pool_cu_seqlens = pool_boundaries(cu_seqlens, kpool)
    pooled = torch.empty(
        (max_pools, head_dim), dtype=index_k.dtype, device=index_k.device
    )
    _pooled_keys_kernel[(max_pools,)](
        index_k.contiguous(),
        gate_score.contiguous(),
        ape.float().contiguous(),
        cu_seqlens.contiguous(),
        pool_cu_seqlens.contiguous(),
        cu_seqlens.shape[0] - 1,
        pooled,
        D=head_dim,
        KPOOL=kpool,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )
    return pooled


@triton.jit
def _expand_topk_kernel(
    pools_ptr,
    scores_ptr,
    seq_base_ptr,
    local_pos_ptr,
    pool_base_ptr,
    out_ptr,
    topk,
    group_topk,
    out_width,
    KPOOL: tl.constexpr,
    WITH_TAIL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    col0 = tl.program_id(1) * BLOCK
    cols = col0 + tl.arange(0, BLOCK)
    in_out = cols < out_width
    base = tl.load(seq_base_ptr + t)
    lpos = tl.load(local_pos_ptr + t)
    if (lpos + 1) <= topk:
        val = tl.where((cols < topk) & (cols <= lpos), base + cols, -1)
    else:
        pbase = tl.load(pool_base_ptr + t)
        groups = col0 // KPOOL + tl.arange(0, BLOCK // KPOOL)
        gmask = groups < group_topk
        pool = tl.load(pools_ptr + t * group_topk + groups, mask=gmask, other=0).to(
            tl.int32
        )
        score = tl.load(
            scores_ptr + t * group_topk + groups, mask=gmask, other=float("-inf")
        )
        finite = (score == score) & (score != float("inf")) & (score != float("-inf"))
        slots = tl.arange(0, KPOOL)
        candidates = (base + (pool - pbase) * KPOOL)[:, None] + slots[None, :]
        val = tl.reshape(tl.where((gmask & finite)[:, None], candidates, -1), (BLOCK,))
        if WITH_TAIL:
            tail_start = base + ((lpos + 1) // KPOOL) * KPOOL
            tail_slot = cols - topk
            tail_ok = (tail_slot >= 0) & (tail_slot < (lpos + 1) % KPOOL)
            val = tl.where(tail_ok, tail_start + tail_slot, val)
    tl.store(out_ptr + t * out_width + cols, val.to(tl.int32), mask=in_out)


@triton.jit
def _append_tail_kernel(
    tokens_ptr,
    seq_base_ptr,
    local_pos_ptr,
    shortcut_ptr,
    out_ptr,
    width,
    out_width,
    KPOOL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    in_out = cols < out_width
    copy = cols < width
    tokens = tl.load(tokens_ptr + t * width + cols, mask=copy, other=-1).to(tl.int32)
    base = tl.load(seq_base_ptr + t)
    lpos = tl.load(local_pos_ptr + t)
    shortcut = tl.load(shortcut_ptr + t)
    tail_start = base + ((lpos + 1) // KPOOL) * KPOOL
    tail_slot = cols - width
    tail_ok = (tail_slot >= 0) & (tail_slot < (lpos + 1) % KPOOL) & (shortcut == 0)
    val = tl.where(copy, tokens, tl.where(tail_ok, tail_start + tail_slot, -1))
    tl.store(out_ptr + t * out_width + cols, val, mask=in_out)


def _pool_topk(pool_logits: torch.Tensor, topk: int, kpool: int):
    _, num_pools = pool_logits.shape
    group_topk = min(topk // kpool, num_pools)
    assert group_topk > 0, (topk, kpool, num_pools)
    assert kpool & (kpool - 1) == 0 and _SELECT_BLOCK % kpool == 0, kpool
    scores, pools = torch.topk(pool_logits.float(), group_topk, dim=-1)
    return scores, pools, group_topk


def pool_topk_to_token_fn(seq_token_base, pool_base, local_positions, kpool):
    def topk_fn(pool_logits: torch.Tensor, topk: int) -> torch.Tensor:
        num_tokens = pool_logits.shape[0]
        scores, pools, group_topk = _pool_topk(pool_logits, topk, kpool)
        tokens = torch.empty(
            (num_tokens, topk), dtype=torch.int32, device=pool_logits.device
        )
        grid = (num_tokens, triton.cdiv(topk, _SELECT_BLOCK))
        _expand_topk_kernel[grid](
            pools,
            scores,
            seq_token_base,
            local_positions,
            pool_base,
            tokens,
            topk,
            group_topk,
            topk,
            KPOOL=kpool,
            WITH_TAIL=False,
            BLOCK=_SELECT_BLOCK,
        )
        return tokens

    return topk_fn


def append_tail_and_pad(
    tokens: torch.Tensor,
    seq_token_base: torch.Tensor,
    local_positions: torch.Tensor,
    shortcut: torch.Tensor,
    kpool: int,
    pad_multiple: int = SPARSE_MLA_BLOCK,
) -> torch.Tensor:
    num_tokens, width = tokens.shape
    out_width = (width + kpool - 1 + pad_multiple - 1) // pad_multiple * pad_multiple
    out = torch.empty((num_tokens, out_width), dtype=torch.int32, device=tokens.device)
    grid = (num_tokens, triton.cdiv(out_width, _SELECT_BLOCK))
    _append_tail_kernel[grid](
        tokens.contiguous(),
        seq_token_base.to(torch.int32).contiguous(),
        local_positions.to(torch.int32).contiguous(),
        shortcut.to(torch.int32).contiguous(),
        out,
        width,
        out_width,
        KPOOL=kpool,
        BLOCK=_SELECT_BLOCK,
    )
    return out


def select_expand_tail(
    pool_logits, seq_token_base, pool_base, local_positions, topk, kpool
):
    num_tokens = pool_logits.shape[0]
    scores, pools, group_topk = _pool_topk(pool_logits, topk, kpool)
    out_width = (
        (topk + kpool - 1 + SPARSE_MLA_BLOCK - 1) // SPARSE_MLA_BLOCK * SPARSE_MLA_BLOCK
    )
    out = torch.empty(
        (num_tokens, out_width), dtype=torch.int32, device=pool_logits.device
    )
    grid = (num_tokens, triton.cdiv(out_width, _SELECT_BLOCK))
    _expand_topk_kernel[grid](
        pools,
        scores,
        seq_token_base,
        local_positions,
        pool_base,
        out,
        topk,
        group_topk,
        out_width,
        KPOOL=kpool,
        WITH_TAIL=True,
        BLOCK=_SELECT_BLOCK,
    )
    return out
