# Adapted from torchtitan: torchtitan/distributed/expert_parallel.py

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.distributed._functional_collectives import (
    all_to_all_single,
    all_to_all_single_autograd,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Shard, distribute_module, distribute_tensor
from torch.distributed.tensor.parallel.style import ParallelStyle

from areal.experimental.models.archon.moe.utils import _permute, _unpermute


class BaseExpertParallel(ParallelStyle, ABC):
    """Abstract base class for Expert Parallelism styles.

    Subclasses implement specific communication patterns for
    dispatching tokens to experts and combining results.

    Subclasses must implement:
    - _partition_fn: Shard expert weights across devices
    - _token_dispatch: Dispatch tokens to devices holding their assigned experts
    - _token_combine: Combine expert outputs back to original token locations
    - _apply: Apply parallelism using distribute_module with hooks
    """

    @abstractmethod
    def _partition_fn(
        self,
        name: str,
        module: nn.Module,
        device_mesh: DeviceMesh,
    ) -> None:
        """Partition expert weights across devices."""

    @abstractmethod
    def _token_dispatch(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, torch.Tensor],
        device_mesh: DeviceMesh,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dispatch tokens to devices holding their assigned experts."""

    @abstractmethod
    def _token_combine(
        self,
        module: nn.Module,
        output: torch.Tensor,
        device_mesh: DeviceMesh,
    ) -> torch.Tensor:
        """Combine expert outputs back to original token locations."""


class ExpertParallel(BaseExpertParallel):
    """Expert Parallelism with ETP=1 (no tensor parallelism within experts).

    This class implements EP by:
    1. Sharding expert weights on dim 0 (expert dimension) across EP ranks
    2. Using all-to-all to dispatch tokens to devices holding their experts
    3. Using all-to-all to combine results back

    Each EP rank holds num_experts // ep_size local experts.
    """

    def __init__(self) -> None:
        super().__init__()
        # State saved during dispatch for use in combine
        self.input_splits: list[int] | None = None
        self.output_splits: list[int] | None = None
        self.input_shape: tuple[int, ...] | None = None
        self.permuted_indices: torch.Tensor | None = None

    def _partition_fn(
        self,
        name: str,
        module: nn.Module,
        device_mesh: DeviceMesh,
    ) -> None:
        """Shard expert weights on expert dimension (dim 0).

        Args:
            name: Module name (unused).
            module: GroupedExperts module to partition.
            device_mesh: 1D mesh for EP.
        """
        for param_name, param in module.named_parameters(recurse=False):
            dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
            module.register_parameter(param_name, dist_param)

    def _token_dispatch(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, torch.Tensor],
        device_mesh: DeviceMesh,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dispatch tokens to EP ranks via all-to-all.

        Args:
            module: GroupedExperts module.
            inputs: Tuple of (routed_input, num_tokens_per_expert).
                routed_input: Shape (total_tokens, dim), tokens sorted by expert.
                num_tokens_per_expert: Shape (num_experts,), token counts per expert.
            device_mesh: 1D mesh for EP.

        Returns:
            Tuple of (dispatched_tokens, local_num_tokens_per_expert).
        """
        routed_input, num_tokens_per_expert = inputs
        group = device_mesh.get_group()
        ep_degree = device_mesh.size()
        num_local_experts = num_tokens_per_expert.shape[0] // ep_degree

        # Step 1: Exchange token counts via all-to-all
        with torch.no_grad():
            num_tokens_per_expert_received = all_to_all_single(
                num_tokens_per_expert,
                None,
                None,
                group=group,
            )
            # Need to wait explicitly because it is used by downstream operations
            # which don't realize that AsyncCollectiveTensor needs unwrapping.
            # This is required for torch.compile compatibility.
            num_tokens_per_expert_received = torch.ops._c10d_functional.wait_tensor(
                num_tokens_per_expert_received
            )

            # Compute splits for variable-size all-to-all
            # input_splits: how many tokens to send to each rank
            # output_splits: how many tokens to receive from each rank
            counts_view = num_tokens_per_expert.view(ep_degree, num_local_experts)
            received_view = num_tokens_per_expert_received.view(
                ep_degree, num_local_experts
            )

            # NOTE: tolist() requires sync to CPU
            self.input_splits = counts_view.sum(dim=1).to(
                torch.device("cpu"), non_blocking=True
            )
            self.output_splits = received_view.sum(dim=1).to(
                torch.device("cpu"), non_blocking=False
            )
            self.input_splits = self.input_splits.tolist()
            self.output_splits = self.output_splits.tolist()

        # Step 2: All-to-all to dispatch tokens
        routed_input = all_to_all_single_autograd(
            routed_input,
            self.output_splits,
            self.input_splits,
            group,
        )

        # Step 3: Permute for grouped_mm alignment
        (
            self.input_shape,
            routed_input,
            self.permuted_indices,
            aligned_num_tokens,
        ) = _permute(
            routed_input,
            num_tokens_per_expert_received,
            ep_degree,
            num_local_experts,
        )

        return routed_input, aligned_num_tokens

    def _token_combine(
        self,
        module: nn.Module,
        output: torch.Tensor,
        device_mesh: DeviceMesh,
    ) -> torch.Tensor:
        """Combine expert outputs via all-to-all back to original locations.

        Args:
            module: GroupedExperts module.
            output: Expert computation output.
            device_mesh: 1D mesh for EP.

        Returns:
            Combined output in original token order.
        """
        group = device_mesh.get_group()

        # Step 1: Unpermute to pre-permutation order
        output = _unpermute(output, self.input_shape, self.permuted_indices)

        # Step 2: All-to-all to combine (reverse direction)
        output = all_to_all_single_autograd(
            output,
            self.input_splits,  # Swap for reverse
            self.output_splits,
            group,
        )

        return output

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        """Apply expert parallelism to the module.

        Uses distribute_module to:
        1. Shard expert weights via partition_fn
        2. Register _token_dispatch as forward pre-hook (input_fn)
        3. Register _token_combine as forward hook (output_fn)

        This way EP communication happens automatically during forward pass.

        Args:
            module: GroupedExperts module to parallelize.
            device_mesh: 1D mesh for EP.

        Returns:
            The parallelized module.
        """
        return distribute_module(
            module,
            device_mesh,
            partition_fn=self._partition_fn,
            input_fn=self._token_dispatch,
            output_fn=self._token_combine,
        )


def apply_expert_parallel(
    experts_module: nn.Module,
    ep_mesh: DeviceMesh,
) -> None:
    """Apply Expert Parallelism to a GroupedExperts module.

    This is a convenience function that applies EP with ETP=1 to the
    given experts module. It uses distribute_module to register hooks
    for automatic token dispatch/combine during forward pass.

    Args:
        experts_module: GroupedExperts module to parallelize.
        ep_mesh: 1D device mesh for EP.

    Example:
        >>> from areal.experimental.models.archon import apply_expert_parallel
        >>> apply_expert_parallel(moe.experts, ep_mesh)
    """
    ep_style = ExpertParallel()
    ep_style._apply(experts_module, ep_mesh)


__all__ = [
    "ExpertParallel",
    "apply_expert_parallel",
]
