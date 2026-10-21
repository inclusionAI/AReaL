# SPDX-License-Identifier: Apache-2.0
# Adapted from radixark/miles (Apache-2.0), commit ee36bb2.

"""Packed kpool selection; each sequence has its own pool origin and causal tail."""

import torch


@torch.no_grad()
def kpool_select_topk(
    index_q, index_k, head_weights, gate_score, ape, cu_seqlens, topk, kpool
):
    # Import GPU kernels only on use, so configuration/tests can run without CUDA.
    from areal.experimental.ops.glm5_next.kpool import (
        build_pooled_keys,
        pool_boundaries,
        select_expand_tail,
    )
    from areal.experimental.ops.glm5_next.tilelang.indexer_fwd import indexer_fwd

    token_ids = torch.arange(index_q.shape[0], device=index_q.device)
    segments = torch.searchsorted(cu_seqlens, token_ids, right=True) - 1
    token_base = cu_seqlens[segments].to(torch.int32)
    pool_base = pool_boundaries(cu_seqlens, kpool)[segments].to(torch.int32)
    positions = (token_ids - token_base).to(torch.int32)
    eligible = torch.div(positions + 1, kpool, rounding_mode="floor")
    keys = build_pooled_keys(index_k, gate_score, ape, cu_seqlens, kpool)
    if keys.shape[0]:
        logits = indexer_fwd(
            index_q,
            keys,
            head_weights.float(),
            pool_base,
            pool_base + eligible,
            clean_logits=True,
        )
    else:
        logits = torch.full((index_q.shape[0], 1), -torch.inf, device=index_q.device)
    return select_expand_tail(
        logits, token_base, pool_base, positions, topk, kpool
    ).unsqueeze(1)
