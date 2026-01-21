# Modified from Megatron-LM.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from typing import List, Optional, Tuple, Union

import torch

import realhf.base.constants as constants
from realhf.api.core.model_api import ReaLModelConfig
from realhf.impl.model.parallelism.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
)
from realhf.impl.model.utils.moe import custom_histc, permute, unpermute


class MoETokenDispatcher:
    """AlltoAll Based Token dispatcher."""

    def __init__(
        self,
        config: ReaLModelConfig,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Initialize the AlltoAll token dispatcher. Currently does not support
        expert parallel.

        Args:
            num_local_experts (int): Number of local experts on the current device.
            config (TransformerConfig): Configuration for the transformer model.
        """
        self.config = config
        self.dtype = dtype
        self.device = device

        self.hidden_shape = None
        self.num_experts = self.config.moe.num_experts

        assert self.num_experts > 0, "Expected at least one expert"

        self.router_topk = self.config.moe.top_k
        self.probs = None

        # Token drop and padding.
        # We need to keep track of the token num if we drop tokens without padding them.
        self.num_out_tokens = None
        # Drop and pad the input to capacity.
        self.capacity_factor = self.config.moe.capacity_factor
        self.drop_and_pad = self.config.moe.pad_to_capacity
        if self.drop_and_pad:
            assert self.capacity_factor is not None
        self.capacity = torch.tensor(0, dtype=torch.long, device=self.device)

        self.num_tokens_per_expert = torch.zeros(
            (self.num_experts,), dtype=torch.long, device=self.device
        )
        self.num_out_tokens = torch.tensor(0, dtype=torch.long, device=self.device)

    def preprocess(self, indices: torch.Tensor) -> torch.Tensor:
        """Preprocess token indices for AlltoAll communication and token
        permutation. This method computes the number of tokens assigned to each
        expert based on the input indices. It also initializes the necessary
        data structures for AlltoAll communication, such as input and output
        splits, and the mapping between global tokens and local experts.

        Args:
            indices (torch.Tensor): Tensor of indices mapping tokens to experts.

        Returns:
            torch.Tensor: Tensor containing the number of tokens assigned to local expert.
        """
        self.num_tokens_per_expert = custom_histc(
            indices,
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )
        # num_local_tokens_per_expert: [num_experts]

        if self.drop_and_pad:
            # probs: [num_experts, capacity]
            self.capacity = self.probs.size(1)
            self.num_tokens_per_expert.fill_(self.capacity)
            return self.num_tokens_per_expert

        self.num_out_tokens = self.num_tokens_per_expert.sum()
        return self.num_tokens_per_expert

    def token_permutation(
        self,
        hidden_states: torch.Tensor,
        probs: torch.Tensor,
        indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dispatch tokens to local experts using AlltoAll communication.

        Args:
            hidden_states (torch.Tensor): Input token embeddings.
            probs (torch.Tensor): Probs of tokens assigned to experts.
            indices (torch.Tensor): Indices of tokens assigned to experts.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Permuted token embeddings for local experts.
                - Number of tokens per expert.
        """

        # Preprocess: Get the metadata for communication, permutation and computation operations.
        self.hidden_shape = hidden_states.shape
        self.probs = probs
        assert probs.dim() == 2, "Expected 2D tensor for probs"
        assert indices.dim() == 2, "Expected 2D tensor for indices"
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
        tokens_per_expert = self.preprocess(indices)

        if constants.sequence_parallel():
            hidden_states = gather_from_sequence_parallel_region(hidden_states)

        # Permutation
        self.hiddden_shape_before_permute = hidden_states.shape

        permutated_input_tokens, self.reversed_input_permutation_mapping = permute(
            hidden_states,
            indices,
            num_out_tokens=self.num_out_tokens,
            padded_mode=self.drop_and_pad,
        )

        return permutated_input_tokens, tokens_per_expert

    def token_unpermutation(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Reverse the token permutation to restore the original order.

        Args:
            hidden_states (torch.Tensor): Output from local experts.
            bias (torch.Tensor, optional): Bias tensor (not supported).

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Unpermuted token embeddings in the original order.
                - None (bias is not supported).
        """

        # Unpermutation
        output = unpermute(
            hidden_states,
            self.reversed_input_permutation_mapping,
            probs=self.probs,
            padded_mode=self.drop_and_pad,
            restore_shape=self.hiddden_shape_before_permute,
        )

        if constants.sequence_parallel():
            output = scatter_to_sequence_parallel_region(output)

        # Reshape the output tensor
        output = output.view(self.hidden_shape)
        return output
