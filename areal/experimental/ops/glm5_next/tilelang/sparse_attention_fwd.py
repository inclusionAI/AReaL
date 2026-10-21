# SPDX-License-Identifier: Apache-2.0
# Adapted from radixark/miles (Apache-2.0), commit ee36bb2.
# Source: miles/kernels/attention/dsa/tilelang/sparse_attention_fwd.py

# ruff: noqa
# Adapted from https://github.com/tile-ai/tilelang/blob/e666d2d3cc483829c57618c9ebf2e4f4ada0819d/examples/deepseek_v32/sparse_mla_fwd.py

import tilelang
import torch
from tilelang import language as T


@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def _sparse_attention_fwd_kernel(
    heads,
    dim,
    tail_dim,
    topk,
    kv_group=1,
    sm_scale=None,
    has_sink=False,
    block_I=64,
    threads=256,
):
    assert dim == tilelang.math.next_power_of_2(dim), (
        f"haven't check padding correctness yet, dim={dim}"
    )
    assert tail_dim == 0 or tail_dim == tilelang.math.next_power_of_2(tail_dim), (
        f"haven't check padding correctness yet, dim={tail_dim}"
    )
    assert topk % block_I == 0, (
        "otherwise will load some index=0 thus causing wrong kv to be loaded"
    )
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5 * 1.44269504  # log2(e)
    else:
        sm_scale = sm_scale * 1.44269504  # log2(e)

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    head_kv = heads // kv_group
    q_shape = [batch, seq_len, heads, dim + tail_dim]
    kv_shape = [batch, seq_len_kv, kv_group, dim + tail_dim]
    o_shape = [batch, seq_len, heads, dim]
    indices_shape = [batch, seq_len, kv_group, topk]
    lse_shape = [batch, seq_len, heads]
    sink_shape = [heads]
    indices_dtype = T.int32
    dtype = T.bfloat16
    accum_dtype = T.float32

    G = kv_group
    H = head_kv
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    if padded_H != H:
        assert kv_group == 1, (
            "here we solve the H padding automatically, other wise you should handle Q copy and Output copy with your mask (when kv_group == 1, use g_i * padded_H:(g_i+1) * padded_H would be handled automatically)"
        )
    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    D = dim
    D_tail = tail_dim

    if head_kv > 64:
        assert head_kv % 64 == 0, "head_kv should be a multiple of 64"
        REPLICATE_H = head_kv // 64
    else:
        REPLICATE_H = 1

    H_per_block = padded_H if REPLICATE_H == 1 else 64

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        KV: T.Tensor(kv_shape, dtype),  # type: ignore
        Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
        KVIndices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
        Sink: T.Tensor(sink_shape, accum_dtype),  # type: ignore
        Output: T.Tensor(o_shape, dtype),  # type: ignore
        Lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
    ):
        with T.Kernel(seq_len * REPLICATE_H, batch, kv_group, threads=threads) as (
            bx,
            by,
            bz,
        ):
            Q_shared = T.alloc_shared([H_per_block, D], dtype)
            KV_shared = T.alloc_shared([BI, D], dtype)
            if D_tail > 0:
                Q_tail_shared = T.alloc_shared([H_per_block, D_tail], dtype)
                K_tail_shared = T.alloc_shared([BI, D_tail], dtype)
            O_shared = T.alloc_shared([H_per_block, D], dtype)
            Lse_shared = T.alloc_shared([H_per_block], accum_dtype)

            acc_o = T.alloc_fragment([H_per_block, D], accum_dtype)
            acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
            S_shared = T.alloc_shared([H_per_block, BI], dtype)
            sumexp = T.alloc_fragment([H_per_block], accum_dtype)
            sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
            alpha = T.alloc_fragment([H_per_block], accum_dtype)
            m_i = T.alloc_fragment([H_per_block], accum_dtype)
            m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)

            T.fill(acc_o, 0)
            T.fill(sumexp, 0)
            T.fill(m_i, -(2**30))  # avoid -inf - inf to cause nan

            b_i, g_i = by, bz
            s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)
            H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64)
            H1 = H0 + H_per_block

            T.copy(Q[b_i, s_i, H0:H1, :D], Q_shared)
            if D_tail > 0:
                T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_shared)

            for i_i in T.Pipelined(NI, num_stages=0):
                for bi_i, d_i in T.Parallel(BI, D):
                    KV_shared[bi_i, d_i] = KV[
                        b_i, KVIndices[b_i, s_i, g_i, i_i * BI + bi_i], g_i, d_i
                    ]
                if D_tail > 0:
                    for bi_i, d_i in T.Parallel(BI, D_tail):
                        K_tail_shared[bi_i, d_i] = KV[
                            b_i, KVIndices[b_i, s_i, g_i, i_i * BI + bi_i], g_i, D + d_i
                        ]

                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.if_then_else(
                        Indices[b_i, s_i, g_i, i_i * BI + bi_i] != -1,
                        0,
                        -T.infinity(acc_s.dtype),
                    )
                T.gemm(
                    Q_shared,
                    KV_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                if D_tail > 0:
                    T.gemm(
                        Q_tail_shared,
                        K_tail_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                T.copy(m_i, m_i_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for h_i in T.Parallel(H_per_block):
                    m_i[h_i] = T.max(m_i[h_i], m_i_prev[h_i])
                for h_i in T.Parallel(H_per_block):
                    alpha[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.exp2(
                        acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale
                    )
                T.reduce_sum(acc_s, sumexp_i, dim=1)  # is this a accumulate operator?
                for h_i in T.Parallel(H_per_block):
                    sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                for h_i, d_i in T.Parallel(H_per_block, D):
                    acc_o[h_i, d_i] = acc_o[h_i, d_i] * alpha[h_i]

                T.copy(acc_s, S_shared)
                T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            # Rescale. A query row whose indices are all -1 has no valid key, so sumexp is 0 and
            # both the divide and the log2 go non-finite -- out becomes 0/0 = NaN and Lse becomes
            # -inf, which the backward then turns into exp2(-inf - -inf) = NaN. Any row with at
            # least one valid key has sumexp >= 1 (the running-max term is exp2(0)), so flooring
            # here is a no-op for real rows and makes an empty row contribute an exact zero.
            if has_sink:
                for h_i in T.Parallel(H_per_block):
                    sumexp[h_i] += T.exp2(
                        Sink[H0 + h_i] * 1.44269504 - m_i[h_i] * sm_scale
                    )
            for h_i in T.Parallel(H_per_block):
                sumexp[h_i] = T.max(sumexp[h_i], 1e-30)
            for h_i, d_i in T.Parallel(H_per_block, D):
                acc_o[h_i, d_i] /= sumexp[h_i]
            for h_i in T.Parallel(H_per_block):
                sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale

            T.copy(acc_o, Output[b_i, s_i, H0:H1, :])
            T.copy(sumexp, Lse[b_i, s_i, H0:H1])

    return main


def sparse_attention_fwd(
    q, kv, indices, kv_indices, sink, d_v, *, sm_scale, threads, block_I=64
):
    """q [B, S, H, d_v + d_tail], kv [B, S_kv, G, d_v + d_tail], indices [B, S, G, topk] (-1 = padding),
    kv_indices = indices.clamp(min=0) (the rows actually read), sink [H] fp32 or None.
    Returns out [B, S, H, d_v] and lse [B, S, H] in log2 space."""
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    batch, seq_len, heads, dim_plus_tail_dim = q.shape
    _, seq_len_kv, kv_group, _ = kv.shape
    assert kv.shape[-1] == dim_plus_tail_dim
    assert kv.shape[0] == batch
    topk = indices.shape[-1]
    assert indices.shape == (batch, seq_len, kv_group, topk)
    tail_dim = dim_plus_tail_dim - d_v

    kernel = _sparse_attention_fwd_kernel(
        heads,
        d_v,
        tail_dim,
        topk,
        kv_group,
        sm_scale,
        has_sink=sink is not None,
        block_I=block_I,
        threads=threads,
    )
    if sink is None:
        sink = torch.zeros(heads, device=q.device, dtype=torch.float32)
    return kernel(q, kv, indices, kv_indices, sink)
