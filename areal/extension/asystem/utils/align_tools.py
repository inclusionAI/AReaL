from __future__ import annotations

from collections.abc import Iterable

import torch


def summarize_rewards(rewards: torch.Tensor | Iterable[float]) -> str:
    """Return a compact summary for reward tensors.

    The summary contains the last 10 flattened values, the count of entries equal to 1,
    and the total number of elements.
    """

    if isinstance(rewards, torch.Tensor):
        flat = rewards.detach().flatten()
    else:
        flat = torch.tensor(list(rewards))

    total = flat.numel()
    last_k = flat[-10:] if total > 10 else flat
    ones = (flat == 1).sum().item()

    return f"last_10={last_k.tolist()}, ones_count={ones}, total_count={total}"
