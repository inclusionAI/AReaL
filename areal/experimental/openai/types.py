from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from openai.types.chat import ChatCompletion
from transformers import PreTrainedTokenizer

from areal.api.io_struct import ModelResponse


@dataclass
class CompletionWithTokenLogpReward:
    """Internal structure to store completion with its reward."""

    completion: ChatCompletion
    response: ModelResponse
    messages: List[dict] = field(default_factory=list)
    reward: float | None = None
    parent: Optional["CompletionWithTokenLogpReward"] | None = None
    use_chat_template: bool = False
    tokenizer: Optional[PreTrainedTokenizer] = None

    def to_tensor_dict(self) -> Dict[str, torch.Tensor]:
        resp = self.response
        self.seq_tokens = seq = resp.input_tokens + resp.output_tokens
        if self.parent:
            assert (
                not self.use_chat_template
            ), "Cannot use parent with completions that use chat template."
            parent_res = self.parent.to_tensor_dict()
            parent_logprobs = parent_res["logprobs"].squeeze(0).tolist()
            parent_loss_mask = parent_res["loss_mask"].squeeze(0).tolist()
            parent_versions = parent_res["versions"].squeeze(0).tolist()
            parent_len = len(parent_logprobs)
            assert parent_len == len(parent_loss_mask) == len(parent_versions)
            try:
                assert resp.input_len >= parent_len, (
                    "The input length of the child completion must be greater than or equal to "
                    "the length of the parent completion."
                )
            except AssertionError as e:
                # for debugging
                parent_input = self.tokenizer.decode(
                    self.parent.response.input_tokens, skip_special_tokens=False
                )
                parent_output = self.tokenizer.decode(
                    self.parent.response.output_tokens, skip_special_tokens=False
                )
                child_input = self.tokenizer.decode(
                    resp.input_tokens, skip_special_tokens=False
                )
                child_output = self.tokenizer.decode(
                    resp.output_tokens, skip_special_tokens=False
                )
                print(
                    f"[Debug] Parent input: {len(self.parent.response.input_tokens)}: **********************************************\n",
                    parent_input,
                )
                print(
                    "[Debug] Parent input end: **********************************************\n"
                )
                print(
                    f"[Debug] Parent output: {len(self.parent.response.output_tokens)}: *********************************************\n",
                    parent_output,
                )
                print(
                    "[Debug] Parent output end: **********************************************\n"
                )
                print(
                    f"[Debug] Child input: {len(resp.input_tokens)}: **********************************************\n",
                    child_input,
                )
                print(
                    "[Debug] Child input end: **********************************************\n"
                )
                print(
                    f"[Debug] Child output: {len(resp.output_tokens)}: **********************************************\n",
                    child_output,
                )
                print(
                    "[Debug] Child output end: **********************************************\n"
                )
                raise e

            logprobs = (
                parent_logprobs
                + [0.0] * (resp.input_len - parent_len)
                + resp.output_logprobs
            )
            loss_mask = (
                parent_loss_mask
                + [0] * (resp.input_len - parent_len)
                + [1] * resp.output_len
            )
            versions = (
                parent_versions
                + [-1] * (resp.input_len - parent_len)
                + resp.output_versions
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
