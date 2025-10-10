from __future__ import annotations  # noqa

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from openai.types.chat import ChatCompletion

from areal.api.io_struct import ModelResponse
from areal.utils import logging

logger = logging.getLogger("CompletionWithTokenLogpReward")


@dataclass
class CompletionWithTokenLogpReward:
    """Internal structure to store completion with its reward."""

    completion: ChatCompletion
    response: ModelResponse
    messages: List[dict] = field(default_factory=list)
    reward: Optional[float] = None
    # Optional precomputed multimodal payload (organized in dict form), e.g.:
    # {"multi_modal_input": [{"pixel_values": Tensor, ...}]}
    multimodal_data: Optional[Dict[str, Any]] = None

    def to_tensor_dict(self) -> Dict[str, Any]:
        resp = self.response
        seq = resp.input_tokens + resp.output_tokens
        logprobs = [0.0] * resp.input_len + resp.output_logprobs
        loss_mask = [0] * resp.input_len + [1] * resp.output_len
        versions = [-1] * resp.input_len + resp.output_versions
        reward = self.reward
        assert reward is not None
        result: Dict[str, Any] = dict(
            # unsqueeze to add an additional batch dimension
            input_ids=torch.tensor(seq).unsqueeze(0),
            loss_mask=torch.tensor(loss_mask).unsqueeze(0),
            logprobs=torch.tensor(logprobs).unsqueeze(0),
            versions=torch.tensor(versions).unsqueeze(0),
            attention_mask=torch.ones(len(seq), dtype=torch.bool).unsqueeze(0),
            # reward
            rewards=torch.tensor([float(reward)]),
        )
        # Attach multimodal tensors if available
        if self.multimodal_data is not None:
            result.update(self.multimodal_data)
        return result
