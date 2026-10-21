# SPDX-License-Identifier: Apache-2.0

"""GLM mHC slots for Megatron's HyperConnectionTransformerLayer."""

import torch
import torch.nn.functional as F
from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32

from areal.models.mcore.glm5_next_ops import mhc_aggregate, mhc_combine, mhc_mappings


class Glm5NextHyperConnection(MegatronModule):
    def __init__(self, config, layer_number: int):
        super().__init__(config)
        self.layer_number = layer_number
        self.n = config.num_residual_streams
        self.mapping_proj = torch.nn.Linear(
            self.n * config.hidden_size,
            self.n**2 + 2 * self.n,
            bias=False,
            dtype=torch.float32,
        )
        # Keep the checkpoint's single scale tensor. Splitting into three Parameters
        # prevents streaming HF export from reconstructing hc_{attn,ffn}_scale.
        self.scale = torch.nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.bias = torch.nn.Parameter(
            torch.zeros(self.n**2 + 2 * self.n, dtype=torch.float32)
        )
        for param in self.parameters():
            mark_keep_in_fp32(param)
            param.sequence_parallel = config.sequence_parallel

    def forward(self, hidden_states, mhc_recompute_manager=None, output_slot=None):
        if mhc_recompute_manager is not None or output_slot is not None:
            raise ValueError(
                "GLM-5.3 supports full recompute, not the mHC recompute arena"
            )
        pre, post, residual = mhc_mappings(
            hidden_states,
            self.mapping_proj.weight,
            self.scale,
            self.bias,
            self.n,
            self.config.layernorm_epsilon,
            self.config.mhc_sinkhorn_iterations,
        )
        return mhc_aggregate(hidden_states, pre), residual, post, hidden_states

    def fused_h_res_h_post_bda(
        self,
        h_res,
        original_residual,
        h_post,
        layer_output_with_bias,
        dropout_prob,
        training,
        fused,
        manager=None,
    ):
        if manager is not None:
            raise ValueError("GLM-5.3 does not support the mHC recompute arena")
        output, bias = layer_output_with_bias
        output = output.float()
        if bias is not None:
            output = output + bias.float()
        if dropout_prob:
            output = F.dropout(output, p=dropout_prob, training=training)
        return mhc_combine(original_residual, output, h_res, h_post)


class Glm5NextMeanOutputContraction(MegatronModule):
    def forward(self, hidden_states):
        return hidden_states.unflatten(
            -1, (self.config.num_residual_streams, self.config.hidden_size)
        ).mean(-2)
