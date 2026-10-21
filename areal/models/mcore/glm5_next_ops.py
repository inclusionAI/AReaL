# SPDX-License-Identifier: Apache-2.0

"""Portable GLM-5.3 mHC math, also used by CPU numerical tests."""

import torch
import torch.nn.functional as F


def mhc_mappings(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    num_streams: int,
    norm_eps: float,
    sinkhorn_iterations: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep routing and Sinkhorn in FP32, including epsilon inside RMS sqrt."""
    x = hidden_states.float()
    projected = F.linear(x, weight.float())
    projected = projected * torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)
    pre, post, residual = projected.split(
        [num_streams, num_streams, num_streams**2], dim=-1
    )
    b_pre, b_post, b_res = bias.float().split(
        [num_streams, num_streams, num_streams**2], dim=-1
    )
    h_pre = (pre * scale[0] + b_pre).sigmoid() + 1e-6
    h_post = (post * scale[1] + b_post).sigmoid() * 2
    logits = (residual * scale[2] + b_res).unflatten(-1, (num_streams, num_streams))
    h_res = logits.softmax(dim=-1) + 1e-6
    h_res = h_res / (h_res.sum(dim=-2, keepdim=True) + 1e-6)
    for _ in range(sinkhorn_iterations - 1):
        h_res = h_res / (h_res.sum(dim=-1, keepdim=True) + 1e-6)
        h_res = h_res / (h_res.sum(dim=-2, keepdim=True) + 1e-6)
    return h_pre, h_post, h_res


def mhc_aggregate(hidden_states: torch.Tensor, h_pre: torch.Tensor) -> torch.Tensor:
    streams = hidden_states.unflatten(-1, (h_pre.shape[-1], -1))
    return (streams.float() * h_pre.unsqueeze(-1)).sum(-2).to(hidden_states.dtype)


def mhc_combine(
    residual: torch.Tensor,
    output: torch.Tensor,
    h_res: torch.Tensor,
    h_post: torch.Tensor,
) -> torch.Tensor:
    streams = residual.unflatten(-1, (h_post.shape[-1], -1)).float()
    mixed = h_res.transpose(-1, -2) @ streams
    injected = output.float().unsqueeze(-2) * h_post.unsqueeze(-1)
    return (mixed + injected).flatten(-2).to(residual.dtype)
