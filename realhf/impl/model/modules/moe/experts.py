# Modified from Megatron-LM.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from typing import Callable, List, Optional, Union

import torch
import torch.nn.init as init
from torch.nn.parameter import Parameter

import realhf.base.constants as constants
from realhf.api.core.model_api import ReaLModelConfig
from realhf.impl.model.modules.mlp import LlamaLayerNormMLP, get_activation_fn
from realhf.impl.model.parallelism.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
)
from realhf.impl.model.parallelism.tensor_parallel.utils import divide
from realhf.impl.model.utils.random import _initialize_affine_weight_gpu

try:
    import grouped_gemm
except ImportError:
    grouped_gemm = None


class SequentialMLP(torch.nn.Module):
    """An implementation of the Experts layer using a sequence of MLP layers.

    This class executes each expert sequentially.
    """

    def __init__(
        self,
        config: ReaLModelConfig,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        self.config = config

        self.num_experts = self.config.moe.num_experts
        self.local_experts = torch.nn.ModuleList()

        for _ in range(self.num_experts):
            expert = LlamaLayerNormMLP(
                hidden_dim=config.hidden_dim,
                intermediate_dim=config.intermediate_dim,
                activation_function=config.activation_function,
                use_bias=config.use_mlp_bias,
                is_expert=True,
                dtype=dtype,
                device=device,
            )
            self.local_experts.append(expert)

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
    ):
        output_local = torch.zeros_like(permuted_local_hidden_states)
        cumsum_num_tokens = torch.cumsum(tokens_per_expert, dim=0)
        # Insert zero at the begining for offset index's convenience
        zero_tensor = torch.zeros(1, dtype=torch.long, device=cumsum_num_tokens.device)
        cumsum_num_tokens = torch.cat((zero_tensor, cumsum_num_tokens))

        for expert_num, expert in enumerate(self.local_experts):
            start = cumsum_num_tokens[expert_num]
            end = cumsum_num_tokens[expert_num + 1]
            hidden = permuted_local_hidden_states[start:end]
            output = expert(hidden)
            output_local[start:end] = output

        return output_local


class ExpertParam(torch.nn.Module):
    """A dummy class that maps weight tensors in GroupedMLP to pytorch
    parameters for compatibility of weight saving/loading."""

    def __init__(
        self,
        gate_proj: torch.Tensor,
        up_proj: torch.Tensor,
        down_proj: torch.Tensor,
    ):
        class LinearParam(torch.nn.Module):
            def __init__(self, param: torch.Tensor):
                super(LinearParam, self).__init__()
                self.weight = Parameter(param)

        super(ExpertParam, self).__init__()

        self.gate_proj = LinearParam(gate_proj)
        self.up_proj = LinearParam(up_proj)
        self.down_proj = LinearParam(down_proj)


class GroupedMLP(torch.nn.Module):
    """An efficient implementation of the Experts layer using CUTLASS GroupedGEMM.
    See https://github.com/tgale96/grouped_gemm for details.

    This class is designed to execute multiple experts in parallel, thereby maximizing computational efficiency.
    """

    def __init__(
        self,
        config: ReaLModelConfig,
        init_method: Callable = init.xavier_normal_,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        assert (
            not constants.sequence_parallel()
        ), "Grouped GEMM does not support sequence parallel"

        self.config = config
        self.dtype = dtype
        self.device = device
        self.num_experts = config.moe.num_experts

        assert grouped_gemm is not None, "Grouped GEMM is not available."

        self.activation_func = get_activation_fn(self.config.activation_function)

        # How many feature each rank holds for fc1 and fc2, respectively.
        tp_size = constants.tensor_parallel_world_size()
        intermediate_dim_per_partition = divide(self.config.intermediate_dim, tp_size)

        # Note: The current kernel implementations of grouped_gemm
        # does not support transposition with CUTLASS grouped GEMM
        # and as a result we avoid allocate the transpose of weights.
        self.grouped_gate_proj = torch.empty(
            self.num_experts,
            self.config.hidden_dim,
            intermediate_dim_per_partition,
            device=self.device,
            dtype=self.dtype,
        )
        self.grouped_up_proj = torch.empty(
            self.num_experts,
            self.config.hidden_dim,
            intermediate_dim_per_partition,
            device=self.device,
            dtype=self.dtype,
        )
        self.grouped_down_proj = torch.empty(
            self.num_experts,
            intermediate_dim_per_partition,
            self.config.hidden_dim,
            device=self.device,
            dtype=self.dtype,
        )
        # Initialize weight.
        _initialize_affine_weight_gpu(
            self.grouped_gate_proj,
            init_method,
            partition_dim=1,
        )
        _initialize_affine_weight_gpu(
            self.grouped_up_proj,
            init_method,
            partition_dim=0,
        )
        _initialize_affine_weight_gpu(
            self.grouped_down_proj,
            init_method,
            partition_dim=0,
        )

        # Parameters for weight loading
        self.local_experts = torch.nn.ModuleList()
        for i in range(self.num_experts):
            expert = ExpertParam(
                self.grouped_gate_proj[i, :].transpose_(0, 1),
                self.grouped_up_proj[i, :].transpose_(0, 1),
                self.grouped_down_proj[i, :].transpose_(0, 1),
            )
            self.local_experts.append(expert)

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
    ):
        tokens_per_expert = tokens_per_expert.cpu()
        if permuted_local_hidden_states.nelement() != 0:
            if constants.tensor_parallel_world_size() > 1:
                permuted_local_hidden_states = copy_to_tensor_model_parallel_region(
                    permuted_local_hidden_states
                )

            # Reshape the weights for the grouped GEMMs.
            o1 = grouped_gemm.ops.gmm(
                permuted_local_hidden_states,
                self.grouped_gate_proj,
                tokens_per_expert,
                trans_b=False,
            )
            o2 = grouped_gemm.ops.gmm(
                permuted_local_hidden_states,
                self.grouped_up_proj,
                tokens_per_expert,
                trans_b=False,
            )
            inter = self.activation_func(o1) * o2
            output = grouped_gemm.ops.gmm(
                inter, self.grouped_down_proj, tokens_per_expert, trans_b=False
            )
            if constants.tensor_parallel_world_size() > 1:
                output = reduce_from_tensor_model_parallel_region(output)
        else:
            # No token is allocated for local experts.
            assert torch.count_nonzero(tokens_per_expert) == 0

            # Make sure parameters still have gradients when no tokens are routed to this set of experts.
            gate_proj = self.grouped_gate_proj.view(self.config.hidden_dim, -1)
            up_proj = self.grouped_up_proj.view(self.config.hidden_dim, -1)
            down_proj = self.grouped_down_proj.view(-1, self.config.hidden_dim)

            o1 = torch.matmul(permuted_local_hidden_states, gate_proj)
            o2 = torch.matmul(permuted_local_hidden_states, up_proj)
            inter = self.activation_func(o1 * o2)
            output = torch.matmul(inter, down_proj)
        return output
