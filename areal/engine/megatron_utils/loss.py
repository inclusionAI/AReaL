# SPDX-License-Identifier: Apache-2.0

"""Adapt independently normalized losses to MCore's native token denominator."""

from collections.abc import Iterable, Mapping
from typing import Any

import torch
import torch.distributed as dist

from areal.utils.data import TRANSPORT_DUMMY_KEY


def compute_global_token_count(
    microbatches: Iterable[Mapping[str, Any]],
    dp_group: dist.ProcessGroup,
    device: torch.device | str,
) -> torch.Tensor:
    """Count real loss tokens over DP, excluding replicated CP copies.

    Call on every pipeline stage, using full-sequence microbatches and a DP
    group WITHOUT context parallelism. Keep the count integer end to end.
    """
    counts = []
    for batch in microbatches:
        if batch.get(TRANSPORT_DUMMY_KEY) is True:
            counts.append(torch.zeros((), dtype=torch.int64, device=device))
        elif "loss_mask" not in batch:
            raise ValueError("Native named losses require an explicit loss_mask")
        else:
            counts.append(batch["loss_mask"].count_nonzero().to(device=device))
    count = torch.stack(counts).sum()
    count = count.detach().to(device=device, dtype=torch.int64)
    dist.all_reduce(count, group=dp_group)
    return count


def scale_named_loss_for_native(
    normalized_loss: torch.Tensor,
    token_count: torch.Tensor,
    total_token_count: torch.Tensor,
    cp_size: int,
    cp_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a native numerator/count without replacing per-term normalizers.

    ``normalized_loss`` is sum_k N_k / (CP * global_mass_k), computed from
    the full sequence on each CP rank. MCore's per-token path sums gradients
    over DP+CP and divides by the global token count T. Multiplying by T here
    cancels only that final division; each term retains its own denominator.
    No local-token, DP, microbatch, or optimizer loss-scale factor belongs here.

    Partition real integer counts across CP so MCore still sees exactly T,
    preserving native auxiliary-loss scaling. With T=0 MCore skips division,
    so use a factor of one, not a fabricated nonzero token count.
    """
    count = token_count.detach().to(device=normalized_loss.device, dtype=torch.int64)
    local_count = torch.div(count, cp_size, rounding_mode="floor")
    local_count = local_count + (count.remainder(cp_size) > cp_rank).to(count.dtype)
    scale = (
        total_token_count.detach()
        .clamp_min(1)
        .to(device=normalized_loss.device, dtype=torch.float32)
    )
    return normalized_loss.float() * scale, local_count
