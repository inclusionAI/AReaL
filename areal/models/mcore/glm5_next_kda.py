# SPDX-License-Identifier: Apache-2.0
# Adapted from radixark/miles (Apache-2.0), commit ee36bb2,
# miles_plugins/models/glm5_next/kda.py. CP is intentionally restricted to one.

import torch
import torch.nn as nn
from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32


class Glm5NextKDA(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        conv_kernel_size: int,
        gate_lower_bound: float,
        rms_norm_eps: float,
    ):
        super().__init__()
        from fla.modules import FusedRMSNormGated, ShortConvolution

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.projection_size = num_heads * head_dim
        self.conv_kernel_size = conv_kernel_size
        self.gate_lower_bound = gate_lower_bound

        self.q_proj = nn.Linear(hidden_size, self.projection_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.projection_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.projection_size, bias=False)
        self.conv1d = ShortConvolution(
            hidden_size=3 * self.projection_size,
            kernel_size=conv_kernel_size,
            bias=False,
            activation="silu",
        )
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)
        self.f_a_proj = nn.Linear(hidden_size, head_dim, bias=False)
        self.f_b_proj = nn.Linear(head_dim, self.projection_size, bias=False)
        self.g_a_proj = nn.Linear(hidden_size, head_dim, bias=False)
        self.g_b_proj = nn.Linear(head_dim, self.projection_size, bias=False)
        self.A_log = mark_keep_in_fp32(
            nn.Parameter(torch.zeros(num_heads, dtype=torch.float32))
        )
        self.dt_bias = mark_keep_in_fp32(
            nn.Parameter(torch.zeros(self.projection_size, dtype=torch.float32))
        )
        self.o_norm = FusedRMSNormGated(
            head_dim, eps=rms_norm_eps, activation="sigmoid"
        )
        self.o_proj = nn.Linear(self.projection_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor):
        from fla.ops.kda import chunk_kda
        from fla.ops.kda.gate import fused_kda_gate

        mixed_qkv = torch.cat(
            (
                self.q_proj(hidden_states),
                self.k_proj(hidden_states),
                self.v_proj(hidden_states),
            ),
            dim=-1,
        )
        mixed_qkv, _ = self.conv1d(
            x=mixed_qkv,
            cu_seqlens=cu_seqlens,
        )
        query, key, value = torch.split(mixed_qkv, [self.projection_size] * 3, dim=-1)
        query = query.unflatten(-1, (self.num_heads, self.head_dim))
        key = key.unflatten(-1, (self.num_heads, self.head_dim))
        value = value.unflatten(-1, (self.num_heads, self.head_dim))

        beta = torch.sigmoid(self.b_proj(hidden_states).float())
        forget = self.f_b_proj(self.f_a_proj(hidden_states))
        g = fused_kda_gate(
            forget.unflatten(-1, (self.num_heads, self.head_dim)),
            self.A_log,
            self.dt_bias,
            lower_bound=self.gate_lower_bound,
        )

        core_attn_out, _ = chunk_kda(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )

        norm_gate = self.g_b_proj(self.g_a_proj(hidden_states))
        out_shape = core_attn_out.shape
        core_attn_out = self.o_norm(
            core_attn_out.reshape(-1, self.head_dim),
            norm_gate.reshape(-1, self.head_dim),
        )
        core_attn_out = core_attn_out.reshape(out_shape[0], out_shape[1], -1)
        return self.o_proj(core_attn_out)


class Glm5NextKDAAttention(MegatronModule):
    """Replicated KDA with SP gather/scatter, matching the Miles GLM recipe."""

    def __init__(
        self, config, layer_number: int, glm_config, pg_collection=None, **kwargs
    ):
        super().__init__(config)
        self.layer_number = layer_number
        if config.context_parallel_size != 1:
            raise ValueError("GLM-5.3 KDA currently requires CP=1")
        self.tp_group = (
            pg_collection.tp
            if pg_collection is not None
            else parallel_state.get_tensor_model_parallel_group()
        )
        self.kda = Glm5NextKDA(
            hidden_size=glm_config.text["hidden_size"],
            num_heads=glm_config.linear_num_heads,
            head_dim=glm_config.linear_head_dim,
            conv_kernel_size=glm_config.linear_conv_kernel_dim,
            gate_lower_bound=glm_config.linear_lower_bound,
            rms_norm_eps=glm_config.text["rms_norm_eps"],
        )
        # Full-sequence replicated computation already produces identical complete
        # parameter gradients. Do not tag these for SP's additional all-reduce.
        for param in self.kda.parameters():
            param.sequence_parallel = False

    def forward(
        self, hidden_states, attention_mask=None, packed_seq_params=None, **kwargs
    ):
        if packed_seq_params is None or hidden_states.shape[1] != 1:
            raise ValueError("GLM-5.3 KDA requires a packed THD batch")
        if (
            kwargs.get("inference_context") is not None
            or kwargs.get("inference_params") is not None
        ):
            raise ValueError(
                "GLM-5.3 actor attention does not support cached inference"
            )
        if self.config.sequence_parallel:
            hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
                hidden_states,
                tensor_parallel_output_grad=False,
                group=self.tp_group,
            )
        output = self.kda(hidden_states.transpose(0, 1), packed_seq_params.cu_seqlens_q)
        output = output.transpose(0, 1).contiguous()
        if self.config.sequence_parallel:
            output = tensor_parallel.scatter_to_sequence_parallel_region(
                output, group=self.tp_group
            )
        return output, None
