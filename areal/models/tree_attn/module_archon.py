"""Tree attention module for Archon engine.

This module provides a TreeAttentionWrapper that has the same interface as
VarlenAttentionWrapper but uses tree attention (flex attention or Triton kernel)
instead of standard varlen attention.

The wrapper is designed to be a drop-in replacement for VarlenAttentionWrapper
when tree training is enabled. Set attn_type="tree" in model_args to use it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from areal.models.tree_attn.constants import USE_TRITON_TREE_ATTN
from areal.models.tree_attn.triton_kernel import TRITON_AVAILABLE, TreeAttentionData
from areal.utils import logging

logger = logging.getLogger("TreeAttnArchon")

__all__ = [
    "TreeAttentionWrapper",
]


class TreeAttentionWrapper(nn.Module):
    """Attention wrapper for tree training in Archon Engine.

    This wrapper has the same interface as VarlenAttentionWrapper but uses
    tree attention (flex attention with BlockMask or Triton tree attention)
    instead of standard flash attention.

    The wrapper expects tree attention metadata (block_mask or triton_attn_data)
    to be passed as keyword arguments during forward.

    For packed sequences in Archon, batch is always 1.
    """

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        scale: float | None = None,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        block_mask: BlockMask | None = None,
        triton_attn_data: TreeAttentionData | None = None,
    ) -> torch.Tensor:
        """Compute tree attention.

        Args:
            q: Query tensor, shape [batch, heads, seq_len, head_dim]
            k: Key tensor, shape [batch, heads, seq_len, head_dim]
            v: Value tensor, shape [batch, heads, seq_len, head_dim]
            scale: Optional scale factor for attention scores.
            cu_seqlens: Cumulative sequence lengths (unused for tree attention,
                kept for API compatibility with VarlenAttentionWrapper).
            max_seqlen: Maximum sequence length (unused for tree attention,
                kept for API compatibility with VarlenAttentionWrapper).
            block_mask: BlockMask for flex attention path.
            triton_attn_data: TreeAttentionData for Triton attention path.

        Returns:
            Attention output, shape [batch, heads, seq_len, head_dim]

        Raises:
            ValueError: If neither block_mask nor triton_attn_data is provided,
                or if no valid backend is available.
        """
        # Input: [batch, heads, seq_len, head_dim]
        batch, n_heads, seq_len, head_dim = q.shape

        # For packed sequences, batch should be 1
        assert batch == 1, (
            f"TreeAttentionWrapper expects batch=1 for packed sequences, "
            f"got batch={batch}"
        )

        # Must have at least one tree attention data source
        if block_mask is None and triton_attn_data is None:
            raise ValueError(
                "TreeAttentionWrapper requires either block_mask or triton_attn_data. "
                "For standard attention, use VarlenAttentionWrapper instead."
            )

        # Triton path (preferred when available)
        if USE_TRITON_TREE_ATTN and triton_attn_data is not None and TRITON_AVAILABLE:
            from areal.models.tree_attn.triton_kernel import tree_attention

            output = tree_attention(
                q,
                k,
                v,
                triton_attn_data.packed_mask,
                triton_attn_data.kv_indices,
                triton_attn_data.kv_offsets,
                triton_attn_data.q_indices,
                triton_attn_data.q_offsets,
                sm_scale=scale,
            )
            return output

        # Flex attention path
        if block_mask is not None:
            enable_gqa = q.shape[1] != k.shape[1]
            output = flex_attention(
                q,
                k,
                v,
                block_mask=block_mask,
                scale=scale,
                enable_gqa=enable_gqa,
            )
            return output

        raise ValueError(
            "Tree attention requested but no valid backend available. "
            f"Triton available: {TRITON_AVAILABLE}, "
            f"USE_TRITON_TREE_ATTN: {USE_TRITON_TREE_ATTN}, "
            f"block_mask provided: {block_mask is not None}, "
            f"triton_attn_data provided: {triton_attn_data is not None}"
        )
