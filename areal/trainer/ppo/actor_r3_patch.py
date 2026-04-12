# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
R3 metrics and logging helpers for the PPO actor.

When Router Replay (R3) is enabled, these functions compute and log
statistics about the replayed routing decisions. 
The key effectiveness metrics are:

1. **Router Agreement Rate** -- fraction of tokens where training routing
   matches the replayed (inference-time) routing. Measures how effectively
   R3 forces routing alignment.

2. **Per-Layer Routing Entropy** -- Shannon entropy of the expert probability
   distribution per MoE layer. Lower entropy under replay indicates stronger
   routing concentration (expected when replay overrides natural routing).

3. **Expert Utilization Balance** -- standard deviation of per-expert token
   counts normalised by the mean. High balance (low std/mean) indicates
   evenly distributed expert usage; replay may skew this.

4. **Routing Data Coverage** -- fraction of micro-batches that carried valid
   replay data. Should be 1.0 in a healthy R3 run.

All logging uses the ``stats_tracker`` infrastructure so that metrics
appear in the same TensorBoard / WandB dashboards as other PPO stats.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from areal.utils import stats_tracker

logger = logging.getLogger(__name__)


def log_r3_data_stats(
    data: dict[str, Any],
    scope: str = "r3",
) -> None:
    """Log summary statistics about the ``routed_experts`` tensor in a
    training data dict.

    Called once per PPO update step (not per micro-batch) to avoid
    log spam.

    Args:
        data: The training data dict that may contain ``"routed_experts"``.
        scope: Stats-tracker scope prefix.
    """
    re = data.get("routed_experts")
    if re is None:
        return

    with stats_tracker.scope(scope):
        if isinstance(re, torch.Tensor):
            stats_tracker.scalar(
                r3_batch_size=re.shape[0],
                r3_seq_len=re.shape[1],
                r3_num_layers=re.shape[2] if re.dim() >= 3 else 0,
                r3_topk=re.shape[3] if re.dim() >= 4 else 0,
                r3_dtype_bytes=re.element_size(),
                r3_max_expert_id=re.max().item() if re.numel() > 0 else 0,
            )

            # Compute R3 effectiveness metrics
            _log_r3_effectiveness_metrics(re)
        else:
            stats_tracker.scalar(r3_present=0)


def _log_r3_effectiveness_metrics(
    routed_experts: torch.Tensor,
) -> None:
    """Compute and log R3 effectiveness metrics following SkyRL's approach.

    These metrics help assess whether Router Replay is working correctly
    and how it affects the MoE routing distribution.

    Args:
        routed_experts: ``(bs, seq_len, num_moe_layers, topk)`` int tensor
            containing the expert indices from inference.
    """
    if routed_experts.dim() != 4 or routed_experts.numel() == 0:
        return

    bs, seq_len, num_moe_layers, topk = routed_experts.shape

    try:
        # --- Metric 1: Per-Layer Routing Entropy ---
        # Measures the diversity of expert assignments per layer.
        # Lower entropy = more concentrated routing.
        # Under R3, this reflects the inference-time routing distribution.
        _log_per_layer_routing_entropy(routed_experts, num_moe_layers, topk)

        # --- Metric 2: Expert Utilization Balance ---
        # Measures how evenly tokens are distributed across experts.
        # Coefficient of variation (std/mean) -- lower = more balanced.
        _log_expert_utilization_balance(routed_experts, num_moe_layers)

        # --- Metric 3: Routing Data Coverage ---
        # Fraction of (batch, layer) combinations with non-zero routing data.
        _log_routing_data_coverage(routed_experts, bs, num_moe_layers)

        # --- Metric 4: Top-1 Expert Concentration ---
        # How often the most popular expert is selected (per layer).
        _log_top1_expert_concentration(routed_experts, num_moe_layers)

    except Exception:
        logger.warning(
            "[R3] Failed to compute R3 effectiveness metrics.",
            exc_info=True,
        )


def _log_per_layer_routing_entropy(
    routed_experts: torch.Tensor,
    num_moe_layers: int,
    topk: int,
) -> None:
    """Log per-layer Shannon entropy of expert routing distribution.

    For each MoE layer, computes the probability distribution over experts
    (from the replay data) and its Shannon entropy.  Reports mean, min,
    max across layers.
    """
    bs, seq_len = routed_experts.shape[:2]
    # Flatten batch and seq dimensions
    flat = routed_experts.view(-1, num_moe_layers, topk)  # (bs*seq_len, L, K)
    num_tokens = flat.shape[0]

    if num_tokens == 0:
        return

    # Determine number of experts from max index
    num_experts = int(routed_experts.max().item()) + 1
    if num_experts <= 0:
        return

    layer_entropies = []
    for layer_idx in range(num_moe_layers):
        # Count expert occurrences for this layer across all tokens and topk slots
        expert_ids = flat[:, layer_idx, :].reshape(-1).long()
        # Filter out padding (expert_id == 0 might be valid, but -1 or very large is not)
        valid_mask = (expert_ids >= 0) & (expert_ids < num_experts)
        expert_ids = expert_ids[valid_mask]
        if expert_ids.numel() == 0:
            continue

        counts = torch.bincount(expert_ids, minlength=num_experts).float()
        probs = counts / counts.sum()
        # Shannon entropy: -sum(p * log(p)), with 0*log(0) = 0
        log_probs = torch.where(probs > 0, torch.log2(probs), torch.zeros_like(probs))
        entropy = -(probs * log_probs).sum().item()
        layer_entropies.append(entropy)

    if layer_entropies:
        mean_entropy = sum(layer_entropies) / len(layer_entropies)
        min_entropy = min(layer_entropies)
        max_entropy = max(layer_entropies)
        # Maximum possible entropy for reference
        max_possible = torch.log2(torch.tensor(float(num_experts))).item()

        stats_tracker.scalar(
            r3_routing_entropy_mean=mean_entropy,
            r3_routing_entropy_min=min_entropy,
            r3_routing_entropy_max=max_entropy,
            r3_routing_entropy_normalised=mean_entropy / max_possible if max_possible > 0 else 0,
            r3_num_experts=num_experts,
        )


def _log_expert_utilization_balance(
    routed_experts: torch.Tensor,
    num_moe_layers: int,
) -> None:
    """Log expert utilization balance (coefficient of variation per layer).

    For each layer, compute the standard deviation of per-expert token
    counts divided by the mean.  Aggregate across layers.
    """
    flat = routed_experts.view(-1, num_moe_layers, routed_experts.shape[-1])
    num_experts = int(routed_experts.max().item()) + 1
    if num_experts <= 1:
        return

    layer_cv_values = []
    for layer_idx in range(num_moe_layers):
        expert_ids = flat[:, layer_idx, :].reshape(-1).long()
        valid_mask = (expert_ids >= 0) & (expert_ids < num_experts)
        expert_ids = expert_ids[valid_mask]
        if expert_ids.numel() == 0:
            continue

        counts = torch.bincount(expert_ids, minlength=num_experts).float()
        mean_count = counts.mean()
        if mean_count > 0:
            cv = counts.std() / mean_count
            layer_cv_values.append(cv.item())

    if layer_cv_values:
        stats_tracker.scalar(
            r3_expert_util_cv_mean=sum(layer_cv_values) / len(layer_cv_values),
            r3_expert_util_cv_max=max(layer_cv_values),
            r3_expert_util_cv_min=min(layer_cv_values),
        )


def _log_routing_data_coverage(
    routed_experts: torch.Tensor,
    bs: int,
    num_moe_layers: int,
) -> None:
    """Log fraction of (sample, layer) with non-zero routing data."""
    # Check each sample x layer has at least one non-zero expert id
    # routed_experts: (bs, seq_len, num_moe_layers, topk)
    # Sum over seq_len and topk dimensions
    has_data = (routed_experts.sum(dim=(1, 3)) > 0).float()  # (bs, num_moe_layers)
    coverage = has_data.mean().item()
    stats_tracker.scalar(r3_routing_data_coverage=coverage)


def _log_top1_expert_concentration(
    routed_experts: torch.Tensor,
    num_moe_layers: int,
) -> None:
    """Log how concentrated routing is on the most popular expert per layer.

    For each layer, the concentration ratio = count(most_popular_expert) / total_count.
    High concentration suggests the replay data has strong routing preferences.
    """
    flat = routed_experts.view(-1, num_moe_layers, routed_experts.shape[-1])
    num_experts = int(routed_experts.max().item()) + 1
    if num_experts <= 0:
        return

    layer_concentrations = []
    for layer_idx in range(num_moe_layers):
        expert_ids = flat[:, layer_idx, :].reshape(-1).long()
        valid_mask = (expert_ids >= 0) & (expert_ids < num_experts)
        expert_ids = expert_ids[valid_mask]
        if expert_ids.numel() == 0:
            continue

        counts = torch.bincount(expert_ids, minlength=num_experts)
        max_count = counts.max().item()
        total = counts.sum().item()
        if total > 0:
            layer_concentrations.append(max_count / total)

    if layer_concentrations:
        stats_tracker.scalar(
            r3_top1_expert_concentration_mean=sum(layer_concentrations) / len(layer_concentrations),
            r3_top1_expert_concentration_max=max(layer_concentrations),
        )


def compute_router_agreement_rate(
    replay_indices: torch.Tensor,
    actual_indices: torch.Tensor,
) -> float:
    """Compute the fraction of tokens where actual routing matches replay target.

    This is the KEY R3 effectiveness metric: if R3 is working correctly,
    agreement should be very close to 1.0 (training router produces the same
    assignments as the replayed inference routing).

    Args:
        replay_indices: ``(num_tokens, topk)`` target expert indices from replay.
        actual_indices: ``(num_tokens, topk)`` actual expert indices from training.

    Returns:
        Agreement rate in [0, 1].  Returns -1.0 if inputs are invalid.
    """
    if replay_indices is None or actual_indices is None:
        return -1.0
    if replay_indices.shape != actual_indices.shape:
        logger.warning(
            "[R3] Agreement rate: shape mismatch replay=%s vs actual=%s.",
            replay_indices.shape, actual_indices.shape,
        )
        return -1.0

    # Sort topk indices per token to handle different ordering
    replay_sorted = replay_indices.sort(dim=-1).values
    actual_sorted = actual_indices.sort(dim=-1).values
    matches = (replay_sorted == actual_sorted).all(dim=-1).float()
    return matches.mean().item()


def strip_routed_experts_before_loss(
    data: dict[str, Any],
) -> dict[str, Any]:
    """Remove ``routed_experts`` from the data dict before the loss function.

    The ``routed_experts`` tensor is consumed by the R3 engine patch
    during ``forward_backward_batch``, so by the time we reach the loss
    function it has already been popped.  This function is a safety net.

    Returns the data dict (modified in-place).
    """
    data.pop("routed_experts", None)
    return data
