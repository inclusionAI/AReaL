# Adapted from torchtitan: torchtitan/models/archon/attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from areal.experimental.models.archon.varlen_attention import (
    VarlenAttentionWrapper,
    varlen_attn,
)

__all__ = [
    "SDPAWrapper",
    "VarlenAttentionWrapper",
    "create_block_causal_mask_2d",
    "varlen_attn",
]


def create_block_causal_mask_2d(
    cu_seqlens: torch.Tensor,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create a 2D block-diagonal causal attention mask for SDPA.

    For packed sequences, creates a mask where:
    - Within each sequence: standard causal masking (lower triangular)
    - Across sequences: no attention allowed

    Args:
        cu_seqlens: Cumulative sequence lengths, shape [num_seqs + 1].
            For example, [0, 3, 5, 7] means 3 sequences with lengths 3, 2, 2.
        seq_len: Total sequence length (should equal cu_seqlens[-1]).
        device: Target device for the mask tensor.
        dtype: Target dtype (float mask with 0.0 and -inf).

    Returns:
        Attention mask of shape [seq_len, seq_len].
        0.0 = attend, -inf = mask out.

    Example for cu_seqlens=[0, 3, 5, 7]:
        [  0, -inf, -inf, | -inf, -inf, | -inf, -inf]
        [  0,   0 , -inf, | -inf, -inf, | -inf, -inf]
        [  0,   0 ,   0 , | -inf, -inf, | -inf, -inf]
        [-inf, -inf, -inf, |   0, -inf, | -inf, -inf]
        [-inf, -inf, -inf, |   0,   0 , | -inf, -inf]
        [-inf, -inf, -inf, | -inf, -inf, |   0, -inf]
        [-inf, -inf, -inf, | -inf, -inf, |   0,   0 ]
    """
    positions = torch.arange(seq_len, device=device)
    seq_ids = torch.searchsorted(cu_seqlens, positions, side="right") - 1

    # same_seq: query and key must be in the same sequence
    # causal: key position <= query position
    same_seq = seq_ids.unsqueeze(1) == seq_ids.unsqueeze(0)
    causal = positions.unsqueeze(0) <= positions.unsqueeze(1)

    mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
    mask.masked_fill_(same_seq & causal, 0.0)
    return mask


class SDPAWrapper(nn.Module):
    """SDPA wrapper for packed sequences with block-diagonal causal mask.

    This wrapper supports packed sequences by creating a block-diagonal
    attention mask that combines causal masking with document boundary
    masking.

    The mask is cached internally to avoid regeneration when the same
    cu_seqlens structure is used repeatedly.
    """

    # Backend priority order
    DEFAULT_BACKENDS = [
        SDPBackend.CUDNN_ATTENTION,
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.MATH,
    ]

    def __init__(self, sdpa_backends: list[SDPBackend] | None = None):
        super().__init__()
        self.sdpa_backends = (
            sdpa_backends if sdpa_backends is not None else self.DEFAULT_BACKENDS
        )
        # Cache for mask reuse
        self._cached_mask: torch.Tensor | None = None
        self._cached_cu_seqlens: torch.Tensor | None = None
        self._cached_seq_len: int = 0

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        scale: float | None = None,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        """Compute attention with block-diagonal causal mask.

        Args:
            q: Query tensor, shape [batch, heads, seq_len, head_dim]
            k: Key tensor, shape [batch, heads, seq_len, head_dim]
            v: Value tensor, shape [batch, heads, seq_len, head_dim]
            scale: Optional scale factor for attention scores.
            cu_seqlens: Cumulative sequence lengths, shape [num_seqs + 1].
            max_seqlen: Maximum sequence length (unused, for API compatibility).

        Returns:
            Attention output, shape [batch, heads, seq_len, head_dim]
        """
        seq_len = q.shape[2]
        attn_mask = self._get_mask(cu_seqlens, seq_len, q.device, q.dtype)

        with sdpa_kernel(self.sdpa_backends, set_priority=True):
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                scale=scale,
                is_causal=False,
                enable_gqa=q.size(1) != k.size(1),
            )

    def _get_mask(
        self,
        cu_seqlens: torch.Tensor,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Get or create cached block-diagonal causal mask."""
        # Check cache validity
        if self._is_cache_valid(cu_seqlens, seq_len):
            # Return cached mask, ensure correct dtype
            if self._cached_mask.dtype != dtype:
                return self._cached_mask.to(dtype)
            return self._cached_mask

        # Create new mask
        mask = create_block_causal_mask_2d(cu_seqlens, seq_len, device, dtype)

        # Update cache
        self._cached_mask = mask
        self._cached_cu_seqlens = cu_seqlens.clone()
        self._cached_seq_len = seq_len

        return mask

    def _is_cache_valid(self, cu_seqlens: torch.Tensor, seq_len: int) -> bool:
        """Check if the cached mask is still valid."""
        if self._cached_mask is None:
            return False
        if self._cached_seq_len != seq_len:
            return False
        if self._cached_cu_seqlens is None:
            return False
        if self._cached_cu_seqlens.shape != cu_seqlens.shape:
            return False
        return torch.equal(self._cached_cu_seqlens, cu_seqlens)
