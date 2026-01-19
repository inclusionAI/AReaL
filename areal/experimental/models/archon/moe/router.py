# Adapted from torchtitan: torchtitan/models/moe/moe.py

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn


class TokenChoiceTopKRouter(nn.Module):
    """Token-choice routing with top-k expert selection.

    Each token is routed to the top-k experts based on learned gate scores.
    Supports optional node-limited routing where experts are divided into groups,
    and only a subset of groups is considered before selecting top-k experts.

    Args:
        dim: Dimension of input tokens.
        num_experts: Total number of experts.
        top_k: Number of experts each token is routed to.
        score_func: Scoring function, either "softmax" or "sigmoid".
        route_norm: Whether to normalize routing scores after top-k selection.
        route_scale: Scale factor applied to routing scores.
        num_expert_groups: Number of expert groups for node-limited routing.
            If None, standard top-k routing is used.
        num_limited_groups: Number of groups to select in node-limited routing.
            Required when num_expert_groups is set.
        _debug_force_load_balance: Force balanced round-robin routing for debugging.

    Attributes:
        gate: Linear projection to compute routing scores.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int,
        score_func: Literal["softmax", "sigmoid"] = "sigmoid",
        route_norm: bool = False,
        route_scale: float = 1.0,
        num_expert_groups: int | None = None,
        num_limited_groups: int | None = None,
        _debug_force_load_balance: bool = False,
    ):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.num_experts = num_experts
        self.top_k = top_k
        self.score_func = score_func
        self.route_norm = route_norm
        self.route_scale = route_scale
        self.num_expert_groups = num_expert_groups
        self.num_limited_groups = num_limited_groups
        self._debug_force_load_balance = _debug_force_load_balance

    def _debug_force_load_balance_routing(
        self, scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Force balanced round-robin expert assignment for debugging.

        Args:
            scores: Original routing scores, shape (num_tokens, num_experts).

        Returns:
            tuple of (selected_experts_indices, top_scores).
        """
        n_tokens = scores.size(0)
        # Round-robin indices with exact balance
        selected_experts_indices = (
            torch.arange(
                n_tokens * self.top_k, device=scores.device, dtype=torch.int64
            ).reshape(n_tokens, self.top_k)
            % self.num_experts
        )
        top_scores = scores.gather(dim=1, index=selected_experts_indices)
        return selected_experts_indices, top_scores

    def _get_node_limited_routing_scores(
        self,
        scores_for_choice: torch.Tensor,
    ) -> torch.Tensor:
        """Apply node-limited routing by masking non-selected expert groups.

        Args:
            scores_for_choice: Router scores with expert_bias, shape (num_tokens, num_experts).

        Returns:
            Modified scores with non-selected groups masked to -inf.
        """
        if self.num_limited_groups is None:
            raise ValueError(
                "num_limited_groups must be set when num_expert_groups is set"
            )
        assert self.num_expert_groups is not None

        if self.num_experts % self.num_expert_groups != 0:
            raise ValueError(
                f"num_experts ({self.num_experts}) must be divisible by "
                f"num_expert_groups ({self.num_expert_groups})"
            )

        experts_per_group = self.num_experts // self.num_expert_groups
        if experts_per_group < 2:
            raise ValueError(f"experts_per_group ({experts_per_group}) must be >= 2")

        # Reshape to (num_tokens, num_groups, experts_per_group)
        scores_grouped = scores_for_choice.view(
            -1, self.num_expert_groups, experts_per_group
        )

        # Score each group by sum of top-2 expert scores within group
        top2_scores_in_group, _ = scores_grouped.topk(2, dim=-1)
        group_scores = top2_scores_in_group.sum(dim=-1)

        # Select top groups
        _, group_idx = torch.topk(
            group_scores, k=self.num_limited_groups, dim=-1, sorted=False
        )

        # Create mask: False for selected groups (keep), True for others (mask)
        group_mask = torch.ones_like(group_scores, dtype=torch.bool)
        group_mask.scatter_(1, group_idx, False)

        # Mask out experts from non-selected groups
        scores_for_choice = scores_grouped.masked_fill(
            group_mask.unsqueeze(-1), float("-inf")
        ).view(-1, self.num_experts)

        return scores_for_choice

    def forward(
        self, x: torch.Tensor, expert_bias: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route tokens to experts.

        Args:
            x: Input tensor, shape (num_tokens, dim) or (batch_size * seq_len, dim).
            expert_bias: Optional bias tensor for load balancing, shape (num_experts,).

        Returns:
            tuple containing:
                - top_scores: Routing scores for selected experts, shape (num_tokens, top_k).
                - selected_experts_indices: Expert indices for each token, shape (num_tokens, top_k).
                - num_tokens_per_expert: Token count per expert, shape (num_experts,).
        """
        # Compute gate scores: (num_tokens, num_experts)
        scores = self.gate(x)

        # Apply scoring function in float32 to avoid loss explosion
        if self.score_func == "sigmoid":
            scores = torch.sigmoid(scores.to(torch.float32))
        elif self.score_func == "softmax":
            scores = F.softmax(scores.to(torch.float32), dim=1)
        else:
            raise NotImplementedError(f"Unknown score function: {self.score_func}")

        # Apply expert bias for routing (not for scoring)
        scores_for_choice = scores if expert_bias is None else scores + expert_bias

        # Apply node-limited routing if configured
        if self.num_expert_groups is not None:
            scores_for_choice = self._get_node_limited_routing_scores(scores_for_choice)

        # Select top-k experts
        _, selected_experts_indices = torch.topk(
            scores_for_choice, k=self.top_k, dim=-1, sorted=False
        )

        # Get scores for selected experts (without bias)
        top_scores = scores.gather(dim=1, index=selected_experts_indices)

        # Debug override: balanced round-robin routing
        if self._debug_force_load_balance:
            selected_experts_indices, top_scores = (
                self._debug_force_load_balance_routing(scores)
            )

        # Normalize scores if requested
        if self.route_norm:
            denominator = top_scores.sum(dim=-1, keepdim=True) + 1e-20
            top_scores = top_scores / denominator

        # Apply scale factor
        top_scores = top_scores * self.route_scale

        # Count tokens per expert
        num_tokens_per_expert = torch.histc(
            selected_experts_indices.view(-1),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        ).to(torch.int64)

        return top_scores, selected_experts_indices, num_tokens_per_expert

    def init_weights(self, init_std: float = 0.02):
        """Initialize gate weights.

        Args:
            init_std: Standard deviation for truncated normal initialization.
        """
        nn.init.trunc_normal_(self.gate.weight, mean=0.0, std=init_std)


__all__ = ["TokenChoiceTopKRouter"]
