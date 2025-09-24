from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from openai.types.chat import ChatCompletion

from areal.api.io_struct import ModelResponse


@dataclass
class CompletionWithTokenLogpReward:
    """Internal structure to store completion with its reward."""

    completion: ChatCompletion
    response: ModelResponse
    messages: List[dict] = field(default_factory=list)
    reward: Optional[float] = None
    parent: "CompletionWithTokenLogpReward" = None

    def to_tensor_dict(self) -> Dict[str, torch.Tensor]:
        resp = self.response
        seq = resp.input_tokens + resp.output_tokens
        if self.parent:
            parent_res = self.parent.to_tensor_dict()
            logprobs = (
                parent_res["logprobs"] + [0.0] * resp.input_len + resp.output_logprobs
            )
            loss_mask = (
                parent_res["loss_mask"] + [0] * resp.input_len + [1] * resp.output_len
            )
            versions = (
                parent_res["versions"] + [-1] * resp.input_len + resp.output_versions
            )
        else:
            logprobs = [0.0] * resp.input_len + resp.output_logprobs
            loss_mask = [0] * resp.input_len + [1] * resp.output_len
            versions = [-1] * resp.input_len + resp.output_versions
        reward = self.reward if self.reward is not None else 0.0
        return dict(
            # unsqueeze to add an additional batch dimension
            input_ids=torch.tensor(seq).unsqueeze(0),
            loss_mask=torch.tensor(loss_mask).unsqueeze(0),
            logprobs=torch.tensor(logprobs).unsqueeze(0),
            versions=torch.tensor(versions).unsqueeze(0),
            attention_mask=torch.ones(len(seq), dtype=torch.bool).unsqueeze(0),
            # reward
            rewards=torch.tensor([float(reward)]),
        )
