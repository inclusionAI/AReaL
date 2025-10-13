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
    parent: "CompletionWithTokenLogpReward" | None = None
    chat_template_type: str = "hf"
    _cache: Dict[str, torch.Tensor] | None = None
    # Optional precomputed multimodal payload (organized in dict form), e.g.:
    # {"multi_modal_input": [{"pixel_values": Tensor, ...}]}
    # Preferred naming aligned with engine/utils expectations
    multi_modal_input: Optional[Dict[str, Any]] = None

    def to_tensor_dict(self) -> Dict[str, torch.Tensor]:
        if self._cache is not None:
            return self._cache
        resp = self.response
        self.seq_tokens = seq = resp.input_tokens + resp.output_tokens
        if self.parent:
            assert self.chat_template_type == "concat"
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
                logger.warning(
                    f"The input length of the child completion ({resp.input_len}) is less than or "
                    f"equal to the length of the parent completion {parent_len}. "
                    "This should not happen if the messages are constructed properly."
                    "Ignoring the parent completion by masking them out. \n"
                    f"Parent input token ids: {self.parent.response.input_tokens}\n"
                    f"Parent output token ids: {self.parent.response.output_tokens}\n"
                    f"Child input token ids: {resp.input_tokens}\n"
                    f"Parent input messages: {self.parent.messages}\n"
                    f"Child input messages: {self.messages}",
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

        # Attach multimodal tensors if available. Prefer key "multi_modal_input".
        if self.multi_modal_input is not None:
            # Enforce image token count equals number of pixel_values entries
            resp_tokens = seq  # use full sequence as requested
            img_token_count = None

            image_token_id = 151655

            if image_token_id is not None and isinstance(resp_tokens, list):
                img_token_count = resp_tokens.count(image_token_id)

            # Count number of pixel_values provided in multi_modal_input
            mm = self.multi_modal_input
            pixel_count = sum(
                1 for it in mm["multi_modal_input"] if isinstance(it, dict) and ("pixel_values" in it)
            )


            if img_token_count is not None and img_token_count != pixel_count:
                msg = (
                    f"Image token/pixel_values mismatch: tokens={img_token_count}, "
                    f"pixel_values={pixel_count}, input_len={resp.input_len}, output_len={resp.output_len}"
                )
                # Output info then raise error
                try:
                    logger.error(msg)
                finally:
                    raise ValueError(msg)

            result["multi_modal_input"] = self.multi_modal_input
        self._cache = result

        return result
