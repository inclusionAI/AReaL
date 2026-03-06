"""ChatCompletionHandler — chat template → SGLang → InteractionCache → ChatCompletion.

Adapted from areal/experimental/openai/client.py (AsyncCompletionsWithReward).
Replaces engine.agenerate() with SGLangBackendWithResubmit.generate() via HTTP.
No imports from areal.experimental.openai.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncGenerator, Iterable
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from openai._types import NOT_GIVEN, Body, NotGiven
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_tool_choice_option_param import (
    ChatCompletionToolChoiceOptionParam,
)
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.completion_usage import CompletionUsage
from openai.types.shared_params.metadata import Metadata

from areal.experimental.gateway.data_proxy.backend import SGLangBackendWithResubmit
from areal.experimental.gateway.data_proxy.tokenizer_proxy import TokenizerProxy
from areal.experimental.gateway.data_proxy.types import InteractionWithTokenLogpReward

if TYPE_CHECKING:
    from areal.api.io_struct import ModelResponse
    from areal.experimental.gateway.data_proxy.cache import InteractionCache

logger = logging.getLogger("DataProxyChat")


def is_omitted(value) -> bool:
    """Check if a value is NOT_GIVEN or Omit type or None."""
    if value is NOT_GIVEN or value is None:
        return True
    try:
        from openai import Omit

        if isinstance(value, Omit):
            return True
    except ImportError:
        pass
    if hasattr(value, "__class__"):
        return value.__class__.__name__ in ("NotGiven", "Omit")
    return False


def _ensure_message_dict_list(
    name: str,
    messages: list,
) -> list[dict]:
    """Convert a list of messages (dicts or BaseModel) to list[dict]."""
    result = []
    for msg in messages:
        if isinstance(msg, dict):
            result.append(msg)
        elif hasattr(msg, "model_dump"):
            result.append(msg.model_dump(exclude_none=True))
        else:
            raise TypeError(
                f"Each element in {name} must be a dict or BaseModel, got {type(msg)}"
            )
    return result


class ChatCompletionHandler:
    """Handles chat completion requests: chat template → SGLang → cache → ChatCompletion.

    This replaces the ArealOpenAI + AsyncCompletionsWithReward chain for the data proxy.
    """

    def __init__(self, backend: SGLangBackendWithResubmit, tok: TokenizerProxy):
        self.backend = backend
        self.tok = tok

    async def create(
        self,
        *,
        messages: Iterable[ChatCompletionMessageParam],
        stream: bool | NotGiven = NOT_GIVEN,
        temperature: float | None | NotGiven = NOT_GIVEN,
        top_p: float | None | NotGiven = NOT_GIVEN,
        max_tokens: int | None | NotGiven = NOT_GIVEN,
        max_completion_tokens: int | None | NotGiven = NOT_GIVEN,
        max_total_tokens: int | None | NotGiven = NOT_GIVEN,
        n: int | None | NotGiven = NOT_GIVEN,
        stop: str | None | list[str] | NotGiven = NOT_GIVEN,
        store: bool | None | NotGiven = NOT_GIVEN,
        frequency_penalty: float | None | NotGiven = NOT_GIVEN,
        tools: Iterable[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN,
        metadata: Metadata | None | NotGiven = NOT_GIVEN,
        extra_body: Body | None = None,
        areal_cache: InteractionCache | None = None,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncGenerator[ChatCompletionChunk, None]:
        """Create a chat completion (OpenAI-compatible signature).

        Matches AsyncCompletionsWithReward.create() from client.py, but uses
        SGLangBackendWithResubmit.generate() instead of engine.agenerate().
        """
        is_streaming = not is_omitted(stream) and stream is True

        # Extract and validate supported parameters
        cache: InteractionCache | None = None
        interaction: InteractionWithTokenLogpReward | None = None
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"

        if not isinstance(messages, Iterable):
            raise TypeError(
                "messages must be provided as an iterable of dictionaries or BaseModel instances."
            )
        if not is_omitted(n) and n != 1:
            raise NotImplementedError("n != 1 is not supported yet")
        n = 1

        messages_list_raw = list(messages)
        if not messages_list_raw:
            raise ValueError("messages cannot be empty")
        messages_list = _ensure_message_dict_list("messages", messages_list_raw)

        if extra_body is None:
            extra_body = {}

        current_time = int(time.time())

        # Add interaction to cache, resolve parent relationship
        if is_omitted(store) or store:
            cache = areal_cache
            if cache is not None:
                if completion_id in cache:
                    raise ValueError(
                        f"Completion {completion_id} already exists in cache"
                    )
                interaction = InteractionWithTokenLogpReward(
                    messages=deepcopy(messages_list),
                )
                cache[completion_id] = interaction

        # Resolve parent interaction for multi-turn
        parent = self._find_parent(areal_cache) if areal_cache is not None else None

        # Compute prompt token IDs
        if parent is not None and parent.model_response is not None:
            prompt_token_ids = self._concat_with_parent(messages_list, parent)
        else:
            prompt_token_ids = await self.tok.apply_chat_template(messages_list)

        # --- Parameter handling (matching AsyncCompletionsWithReward exactly) ---
        temp = 1.0 if is_omitted(temperature) else (temperature or 0.0)

        if not is_omitted(max_tokens):
            # Support deprecated max_tokens usage
            if not is_omitted(max_completion_tokens):
                if (
                    interaction is not None
                    and cache is not None
                    and completion_id in cache
                ):
                    del cache[completion_id]
                raise ValueError(
                    "max_tokens and max_completion_tokens cannot be set at the same time. "
                    "max_tokens has been deprecated. Please use max_completion_tokens instead. "
                    "To set the total max tokens, please use max_total_tokens instead."
                )
            max_completion_tokens = max_tokens

        max_total_tokens_final = None
        if not is_omitted(max_total_tokens):
            max_total_tokens_final = max_total_tokens

        max_new_tokens = None
        if max_total_tokens_final is not None:
            max_new_tokens = max_total_tokens_final - len(prompt_token_ids)
            if max_new_tokens <= 0:
                if (
                    interaction is not None
                    and cache is not None
                    and completion_id in cache
                ):
                    del cache[completion_id]
                raise ValueError(
                    f"len of prompt tokens {len(prompt_token_ids)} exceeds max_total_tokens {max_total_tokens_final}"
                )
        if not is_omitted(max_completion_tokens):
            if max_new_tokens is None:
                max_new_tokens = max_completion_tokens
            else:
                max_new_tokens = min(max_new_tokens, max_completion_tokens)
        if max_new_tokens is None:
            max_new_tokens = 512  # Default value
            logger.warning(
                "Neither max_tokens nor max_completion_tokens is set; "
                "defaulting max_new_tokens to 512."
            )

        top_p_val = 1.0 if is_omitted(top_p) else (top_p or 1.0)
        stop_tokens = None if is_omitted(stop) else stop

        if stop_tokens is not None and not isinstance(stop_tokens, list):
            stop_tokens = [stop_tokens]

        if is_omitted(frequency_penalty):
            frequency_penalty = 0.0

        # --- Build sampling params dict for SGLang backend ---
        sampling_params_dict: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "temperature": temp,
            "top_p": top_p_val,
            "stop_token_ids": list(set([self.tok.eos_token_id, self.tok.pad_token_id])),
        }
        if stop_tokens:
            sampling_params_dict["stop"] = stop_tokens
        if frequency_penalty:
            sampling_params_dict["frequency_penalty"] = frequency_penalty

        # Call SGLang backend
        result = await self.backend.generate(prompt_token_ids, sampling_params_dict)

        # Build ModelResponse for output_tokens_without_stop (lazy import)
        from areal.api.io_struct import ModelResponse  # noqa: F811

        model_resp = ModelResponse(
            input_tokens=prompt_token_ids,
            output_tokens=result.output_tokens,
            output_logprobs=result.output_logprobs,
            stop_reason=result.stop_reason,
            tokenizer=self.tok._tok,
        )

        # Decode output
        try:
            output_tokens_clean = model_resp.output_tokens_without_stop
        except ValueError:
            output_tokens_clean = result.output_tokens
        output_text = self.tok.decode_tokens(output_tokens_clean)

        # Build ChatCompletion
        output_message = ChatCompletionMessage(
            content=output_text,
            role="assistant",
        )
        chat_completion = ChatCompletion(
            id=completion_id,
            choices=[
                Choice(
                    finish_reason=result.stop_reason,
                    index=0,
                    logprobs=None,
                    message=output_message,
                )
            ],
            created=current_time,
            model="sglang",
            object="chat.completion",
            usage=CompletionUsage(
                completion_tokens=len(result.output_tokens),
                prompt_tokens=len(prompt_token_ids),
                total_tokens=len(prompt_token_ids) + len(result.output_tokens),
            ),
        )

        # Update cache with results
        if cache is not None and completion_id in cache:
            cache[completion_id].completion = chat_completion
            cache[completion_id].model_response = model_resp
            cache[completion_id].output_message_list = [
                output_message.model_dump(exclude_none=True)
            ]

        if is_streaming:
            return self._create_stream(
                completion_id=completion_id,
                current_time=current_time,
                output_text=output_text,
                stop_reason=result.stop_reason,
                prompt_tokens=len(prompt_token_ids),
                completion_tokens=len(result.output_tokens),
            )

        return chat_completion

    def _find_parent(self, cache: Any) -> InteractionWithTokenLogpReward | None:
        """Find the last interaction in the cache to use as parent for multi-turn."""
        if len(cache) == 0:
            return None
        last_key = next(reversed(cache))
        parent = cache[last_key]
        # Only use as parent if it has model_response and output_message_list
        if parent.model_response is not None and parent.output_message_list is not None:
            return parent
        return None

    def _concat_with_parent(
        self,
        messages: list[dict],
        parent: InteractionWithTokenLogpReward,
    ) -> list[int]:
        """Concatenate parent tokens with new message tokens for multi-turn.

        Adapted from client.py:concat_prompt_token_ids_with_parent.
        Uses EOS-count alignment to find where new tokens start.
        """
        assert parent.model_response is not None

        parent_tokens = (
            parent.model_response.input_tokens
            + parent.model_response.output_tokens_without_stop
        )
        eos_token_id = self.tok.eos_token_id

        # Add EOS to align with chat template expectations
        parent_tokens = parent_tokens + [eos_token_id]

        # Build full message list: parent messages + parent output + new messages
        all_messages: list[dict] = []
        all_messages += parent.messages if parent.messages else []
        all_messages += parent.output_message_list if parent.output_message_list else []
        all_messages += messages

        # Apply chat template to full conversation
        all_tokens = self.tok._tok.apply_chat_template(
            all_messages,
            add_generation_prompt=True,
            tokenize=True,
        )

        # Find the split point using EOS count alignment
        parent_eos_num = parent_tokens.count(eos_token_id)
        if parent_eos_num > 0:
            child_truncate_idx = _find_kth(all_tokens, eos_token_id, parent_eos_num)
            if child_truncate_idx == -1 or child_truncate_idx + 1 >= len(all_tokens):
                # Fallback: can't align, just use full template output
                return all_tokens
        else:
            child_truncate_idx = -1

        prompt_token_ids = parent_tokens + all_tokens[child_truncate_idx + 1 :]
        return prompt_token_ids

    def _create_stream(
        self,
        completion_id: str,
        current_time: int,
        output_text: str,
        stop_reason: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Generate streaming ChatCompletionChunk objects.

        Since SGLang is called non-streaming, we simulate streaming by
        yielding the complete response as chunks.
        """

        async def _generate():
            # First chunk: role
            yield ChatCompletionChunk(
                id=completion_id,
                choices=[
                    ChunkChoice(
                        delta=ChoiceDelta(role="assistant", content=""),
                        index=0,
                        finish_reason=None,
                    )
                ],
                created=current_time,
                model="sglang",
                object="chat.completion.chunk",
            )

            # Content chunk
            if output_text:
                yield ChatCompletionChunk(
                    id=completion_id,
                    choices=[
                        ChunkChoice(
                            delta=ChoiceDelta(content=output_text),
                            index=0,
                            finish_reason=None,
                        )
                    ],
                    created=current_time,
                    model="sglang",
                    object="chat.completion.chunk",
                )

            # Final chunk with finish_reason and usage
            yield ChatCompletionChunk(
                id=completion_id,
                choices=[
                    ChunkChoice(
                        delta=ChoiceDelta(),
                        index=0,
                        finish_reason=stop_reason,
                    )
                ],
                created=current_time,
                model="sglang",
                object="chat.completion.chunk",
                usage=CompletionUsage(
                    completion_tokens=completion_tokens,
                    prompt_tokens=prompt_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
            )

        return _generate()


def _find_kth(lst: list, target: Any, k: int) -> int:
    """Find the index of the k-th occurrence of target in lst."""
    count = 0
    for i, val in enumerate(lst):
        if val == target:
            count += 1
            if count == k:
                return i
    return -1
