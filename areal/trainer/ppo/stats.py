from __future__ import annotations

from typing import Any

import torch


def infer_token_denominator(
    input_data: dict[str, Any],
    fallback: torch.Tensor,
) -> torch.Tensor:
    """Infer the full token mask for stats logging.

    Context parallelism may slice intermediate tensors such as ``loss_mask`` or
    model outputs, while the original micro-batch metadata still describes the
    full logical sequence. Prefer that metadata for ``n_tokens`` so statistics
    stay consistent with and without context parallelism.
    """

    attention_mask = input_data.get("attention_mask")
    if torch.is_tensor(attention_mask):
        return torch.ones_like(
            attention_mask,
            dtype=torch.bool,
            device=fallback.device,
        )

    cu_seqlens = input_data.get("cu_seqlens")
    if torch.is_tensor(cu_seqlens) and cu_seqlens.numel() > 0:
        return torch.ones(
            int(cu_seqlens[-1].item()),
            dtype=torch.bool,
            device=fallback.device,
        )

    input_ids = input_data.get("input_ids")
    if torch.is_tensor(input_ids):
        return torch.ones_like(
            input_ids,
            dtype=torch.bool,
            device=fallback.device,
        )

    return torch.ones_like(fallback, dtype=torch.bool, device=fallback.device)
