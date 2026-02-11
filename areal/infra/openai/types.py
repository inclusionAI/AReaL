from __future__ import annotations  # noqa

from dataclasses import dataclass, field

import torch
from openai.types.chat import ChatCompletion
from openai.types.responses.response import Response
from openai.types.responses.response_input_param import ResponseInputParam

from areal.api.io_struct import ModelResponse
from areal.utils import logging

logger = logging.getLogger("TokenLogpReward")


@dataclass
class InteractionWithTokenLogpReward:
    """Internal structure to store completions/responses with their rewards."""

    # Common
    model_response: ModelResponse | None = None
    reward: float | None = None
    parent: InteractionWithTokenLogpReward | None = None
    chat_template_type: str = "hf"
    _cache: dict[str, torch.Tensor] | None = None

    # Fields used for parent-child relationship resolving
    messages: list[dict] = field(default_factory=list)
    output_message_list: list[dict] | None = None

    # Completion fields (optional for response)
    completion: ChatCompletion | None = None

    # Response fields (optional for completion)
    response: Response | None = None
    input_data: str | ResponseInputParam = field(default_factory=lambda: "")

    @property
    def is_completion(self) -> bool:
        return self.completion is not None

    @property
    def is_response(self) -> bool:
        return self.response is not None

    @property
    def api_type(self) -> str:
        # TODO: replace api_type value with enum
        """API type (completion/response)."""
        if self.is_completion:
            return "completion"
        elif self.is_response:
            return "response"
        else:
            return "none"

    @property
    def input_name_for_logging(self) -> str:
        # TODO: replace input_name value with enum
        if self.is_completion:
            return "messages"
        elif self.is_response:
            return "input_data"
        else:
            return "none"

    @property
    def current_data(self) -> list[dict] | str | ResponseInputParam | None:
        if self.is_completion:
            return self.messages
        elif self.is_response:
            return self.input_data
        else:
            return None

    @property
    def parent_data(self) -> list[dict] | str | ResponseInputParam | None:
        if self.parent is None:
            return None
        return self.parent.current_data

    @property
    def interaction_id(self) -> str | None:
        if self.is_completion:
            return self.completion.id
        elif self.is_response:
            return self.response.id
        else:
            return None

    @property
    def created_at(self) -> float | None:
        if self.is_completion:
            return float(self.completion.created)
        elif self.is_response:
            return float(self.response.created_at)
        else:
            return None

    @property
    def remaining_messages(self) -> list[dict]:
        if self.parent is None:
            return self.messages
        assert self.parent.output_message_list is not None, (
            "Parent output message is not set."
        )
        parent_len = len(self.parent.messages + self.parent.output_message_list)
        return self.messages[parent_len:]

    def to_tensor_dict(self) -> dict[str, torch.Tensor]:
        if self._cache is not None:
            return self._cache
        resp = self.model_response
        assert resp is not None, "Model response is not set."
        self.seq_tokens = seq = resp.input_tokens + resp.output_tokens
        if self.chat_template_type == "concat" and self.parent is not None:
            parent_res = self.parent.to_tensor_dict()
            parent_logprobs = parent_res["logprobs"].squeeze(0).tolist()
            parent_loss_mask = parent_res["loss_mask"].squeeze(0).tolist()
            parent_versions = parent_res["versions"].squeeze(0).tolist()
            parent_len = len(parent_logprobs)
            assert parent_len == len(parent_loss_mask) == len(parent_versions)
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
                    f"Parent input {input_name}: {self.parent_data}\n"
                    f"Child input {input_name}: {self.current_data}",
                )
                logprobs = [0.0] * resp.input_len + resp.output_logprobs
                loss_mask = [0] * resp.input_len + [1] * resp.output_len
                versions = [-1] * resp.input_len + resp.output_versions
        else:
            logprobs = [0.0] * resp.input_len + resp.output_logprobs
            loss_mask = [0] * resp.input_len + [1] * resp.output_len
            versions = [-1] * resp.input_len + resp.output_versions
        reward = self.reward if self.reward is not None else 0.0
        result = dict(
            # unsqueeze to add an additional batch dimension
            input_ids=torch.tensor(seq).unsqueeze(0),
            loss_mask=torch.tensor(loss_mask).unsqueeze(0),
            logprobs=torch.tensor(logprobs).unsqueeze(0),
            versions=torch.tensor(versions).unsqueeze(0),
            attention_mask=torch.ones(len(seq), dtype=torch.bool).unsqueeze(0),
            # reward
            rewards=torch.tensor([float(reward)]),
        )
        self._cache = result
        return result
