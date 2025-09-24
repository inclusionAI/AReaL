from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import torch
from openai._types import Body
from openai.types.chat import ChatCompletion, ChatCompletionToolParam

from areal.api.io_struct import ModelResponse


@dataclass
class CompletionWithTokenLogpReward:
    """Internal structure to store completion with its reward."""

    completion: ChatCompletion
    response: ModelResponse
    messages: List[dict] = field(default_factory=list)
    reward: float | None = None
    parent: Optional["CompletionWithTokenLogpReward"] | None = None
    tools: Iterable[ChatCompletionToolParam] | None = None
    extra_body: Body | Dict = field(default_factory=dict)
    seq_tokens: List[int] | None = None

    def to_tensor_dict(self) -> Dict[str, torch.Tensor]:
        resp = self.response
        tokenizer = resp.tokenizer
        if self.parent:
            parent_res = self.parent.to_tensor_dict()
            assert self.parent.seq_tokens is not None, "Parent seq_tokens must be set."
            assert (
                tokenizer is not None
            ), "Tokenizer must be provided in ModelResponse if completion is exported with concat style."
            parent_messages = self.parent.messages
            parent_messages_with_output = self.messages[: len(parent_messages) + 1]
            print(
                f"[Debug] parent_messages_with_output = {parent_messages_with_output}"
            )
            print(f"[Debug] parent.messages = {parent_messages}")
            print(f"[Debug] self.messages = {self.messages}")
            debug_parent_messages_tokens = tokenizer.apply_chat_template(
                parent_messages,
                tools=self.tools,
                add_generation_prompt=True,
                tokenize=True,
                **self.extra_body.get("chat_template_kwargs", {}),
            )
            print(
                f"[Debug] parent.messages tokens = {len(debug_parent_messages_tokens)} {debug_parent_messages_tokens}"
            )
            debug_self_messages_tokens = tokenizer.apply_chat_template(
                self.messages,
                tools=self.tools,
                add_generation_prompt=True,
                tokenize=True,
                **self.extra_body.get("chat_template_kwargs", {}),
            )
            print(
                f"[Debug] self.messages tokens = {len(debug_self_messages_tokens)} {debug_self_messages_tokens}"
            )

            parent_remaining_tokens = tokenizer.apply_chat_template(
                parent_messages_with_output,
                tools=self.tools,
                add_generation_prompt=True,
                tokenize=True,
                **self.extra_body.get("chat_template_kwargs", {}),
            )
            print(
                f"[Debug] parent_remaining_tokens = {len(parent_remaining_tokens)} {parent_remaining_tokens}"
            )
            print(
                f"[Debug] resp.input_tokens = {resp.input_tokens}, resp.output_tokens = {resp.output_tokens}"
            )
            new_input_tokens_length = resp.input_len - len(parent_remaining_tokens)
            assert new_input_tokens_length >= 0, (
                f"New input tokens length must be non-negative if a parent is present, got {new_input_tokens_length}."
                "This usually indicates an unexpected behavior when tokenizer applying chat template."
                "Expected behaviors include removing thinking outputs and adding delimiter tokens."
            )
            # Complete the entire sequence including parent's output tokens
            # removed by tokenizer.apply_chat_template
            self.seq_tokens = seq = (
                self.parent.seq_tokens
                + resp.input_tokens[-new_input_tokens_length:]
                + resp.output_tokens
            )
            parent_logprobs = parent_res["logprobs"].squeeze(0).tolist()
            parent_loss_mask = parent_res["loss_mask"].squeeze(0).tolist()
            parent_versions = parent_res["versions"].squeeze(0).tolist()
            parent_len = len(parent_logprobs)
            assert parent_len == len(parent_loss_mask) == len(parent_versions)

            logprobs = (
                parent_logprobs + [0.0] * new_input_tokens_length + resp.output_logprobs
            )
            loss_mask = (
                parent_loss_mask + [0] * new_input_tokens_length + [1] * resp.output_len
            )
            versions = (
                parent_versions + [-1] * new_input_tokens_length + resp.output_versions
            )
        else:
            self.seq_tokens = seq = resp.input_tokens + resp.output_tokens
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
