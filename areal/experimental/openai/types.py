from __future__ import annotations  # noqa

from dataclasses import dataclass, field

import torch
from openai.types.chat import ChatCompletion
from openai.types.responses.response import Response
from openai.types.responses.response_input_param import ResponseInputParam

from areal.api.io_struct import LLMResponse, ModelResponse
from areal.utils import logging

logger = logging.getLogger("InteractionWithTokenLogpReward")


@dataclass
class InteractionWithTokenLogpReward:
    """Internal structure to store completions/responses with their rewards."""

    # Common
    model_response: ModelResponse | LLMResponse
    reward: float | None = None
    parent: InteractionWithTokenLogpReward | None = None
    chat_template_type: str = "hf"
    _cache: dict[str, torch.Tensor] | None = None

    # Completion fields (optional for response)
    completion: ChatCompletion | None = None
    messages: list[dict] = field(default_factory=list)

    # Response fields (optional for completion)
    response: Response | None = None
    input_data: str | ResponseInputParam = field(default_factory=lambda: "")

    # Optional fields for compatibility with rlvr workflow
    task_id: int = 4  # Task ID for multi-task training
    eos_token_id: int | None = None  # EOS token ID for seq_no_eos_mask calculation
    pad_token_id: int | None = None  # PAD token ID for seq_no_eos_mask calculation

    @property
    def is_completion(self) -> bool:
        return self.completion is not None

    @property
    def api_type(self) -> str:
        """API type (completion/response)."""
        return "completion" if self.is_completion else "response"

    @property
    def input_name_for_logging(self) -> str:
        return "messages" if self.is_completion else "input_data"

    def get_parent_data_for_logging(self) -> str:
        if self.parent is None:
            return ""
        if self.is_completion:
            return str(self.parent.messages)
        else:
            return str(self.parent.input_data)

    def get_current_data_for_logging(self) -> str:
        if self.is_completion:
            return str(self.messages)
        else:
            return str(self.input_data)

    def to_tensor_dict(self) -> dict[str, torch.Tensor]:
        if self._cache is not None:
            return self._cache
        resp = self.model_response
        self.seq_tokens = seq = resp.input_tokens + resp.output_tokens
        if self.parent:
            assert self.chat_template_type == "concat"
            parent_res = self.parent.to_tensor_dict()
            parent_logprobs = parent_res["logprobs"].squeeze(0).tolist()
            parent_loss_mask = parent_res["loss_mask"].squeeze(0).tolist()
            parent_versions = parent_res["versions"].squeeze(0).tolist()
            parent_len = len(parent_logprobs)
            assert parent_len == len(parent_loss_mask) == len(parent_versions), (
                f"parent_len: {parent_len}, parent_loss_mask len: {len(parent_loss_mask)}, parent_versions len: {len(parent_versions)}"
            )
            if resp.input_len > parent_len:
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
                # FIXME: Find out why this happens occasionally
                api_type = self.api_type
                input_name = self.input_name_for_logging
                logger.warning(
                    f"The input length of the child {api_type} ({resp.input_len}) is less than or "
                    f"equal to the length of the parent {api_type} {parent_len}. "
                    f"This should not happen if the {input_name}s are constructed properly."
                    f"Ignoring the parent {api_type} by masking them out. \n"
                    f"Parent input token ids: {self.parent.model_response.input_tokens}\n"
                    f"Parent output token ids: {self.parent.model_response.output_tokens}\n"
                    f"Child input token ids: {resp.input_tokens}\n"
                    f"Parent input {input_name}: {self.get_parent_data_for_logging()}\n"
                    f"Child input {input_name}: {self.get_current_data_for_logging()}",
                )
                logprobs = [0.0] * resp.input_len + resp.output_logprobs
                loss_mask = [0] * resp.input_len + [1] * resp.output_len
                versions = [-1] * resp.input_len + resp.output_versions
        else:
            logprobs = [0.0] * resp.input_len + resp.output_logprobs
            loss_mask = [0] * resp.input_len + [1] * resp.output_len
            versions = [-1] * resp.input_len + resp.output_versions
        reward = self.reward if self.reward is not None else 0.0

        # Calculate seq_no_eos_mask: True if sequence doesn't end with EOS or PAD
        seq_no_eos_mask = True
        if seq and (self.eos_token_id is not None or self.pad_token_id is not None):
            last_token = seq[-1]
            ends_with_eos = (
                self.eos_token_id is not None and last_token == self.eos_token_id
            )
            ends_with_pad = (
                self.pad_token_id is not None and last_token == self.pad_token_id
            )
            seq_no_eos_mask = not (ends_with_eos or ends_with_pad)

        # Calculate prompt_mask: 1 for prompt tokens, 0 for output tokens
        # This is the inverse of loss_mask
        prompt_mask = [1 - m for m in loss_mask]

        result = dict(
            # unsqueeze to add an additional batch dimension
            input_ids=torch.tensor(seq).unsqueeze(0),
            loss_mask=torch.tensor(loss_mask).unsqueeze(0),
            prompt_mask=torch.tensor(prompt_mask).unsqueeze(0),
            logprobs=torch.tensor(logprobs).unsqueeze(0),
            versions=torch.tensor(versions).unsqueeze(0),
            attention_mask=torch.ones(len(seq), dtype=torch.bool).unsqueeze(0),
            # reward
            rewards=torch.tensor([float(reward)]),
            # seqlen
            seqlen=torch.tensor([len(seq)]),
            # task_ids
            task_ids=torch.tensor([self.task_id]),
            # seq_no_eos_mask
            seq_no_eos_mask=torch.tensor([seq_no_eos_mask]),
        )
        self._cache = result
        return result
