# Modified from Megatron-LM.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from typing import Optional, Union

import torch
import torch.nn as nn

import realhf.base.constants as constants
import realhf.base.logging as logging
from realhf.api.core.model_api import ReaLModelConfig
from realhf.impl.model.modules.mlp import GemmaRMSNorm, LlamaRMSNorm
from realhf.impl.model.modules.moe.experts import GroupedMLP, SequentialMLP
from realhf.impl.model.modules.moe.router import TopKRouter
from realhf.impl.model.modules.moe.token_dispatcher import MoETokenDispatcher

logger = logging.getLogger(__name__)


class LayerNormMoELayer(torch.nn.Module):

    def __init__(
        self,
        config: ReaLModelConfig,
        layer_idx: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super(LayerNormMoELayer, self).__init__()

        self.config = config
        self.dtype = dtype
        self.device = device
        self.num_experts = self.config.moe.num_experts

        if config.layer_norm_type is None:
            layer_norm_fn = nn.LayerNorm
        elif config.layer_norm_type == "rms":
            layer_norm_fn = LlamaRMSNorm
        elif config.layer_norm_type == "gemma":
            layer_norm_fn = GemmaRMSNorm
        self.ln = layer_norm_fn(
            config.hidden_dim, eps=config.layer_norm_epsilon, dtype=dtype, device=device
        )

        self.router = TopKRouter(config=self.config, layer_idx=layer_idx)
        self.token_dispatcher = MoETokenDispatcher(config=self.config)
        if config.moe.use_grouped_gemm and dtype == torch.bfloat16:
            self.experts = GroupedMLP(self.config, dtype=dtype, device=device)
        else:
            if config.moe.use_grouped_gemm:
                logger.warning(
                    "GroupedGemm only supports bfloat16. Fallback to SequentialMLP."
                )
            self.experts = SequentialMLP(self.config, dtype=dtype, device=device)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.ln(hidden_states)
        probs, indices = self.router(hidden_states)
        (dispatched_input, tokens_per_expert) = self.token_dispatcher.token_permutation(
            hidden_states, probs, indices
        )
        expert_output = self.experts(dispatched_input, tokens_per_expert)
        output = self.token_dispatcher.token_unpermutation(
            expert_output,
        )
        return output
