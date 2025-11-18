import datetime
import os
import uuid
from collections.abc import Iterable
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from openai import AsyncOpenAI
from openai._types import NOT_GIVEN, Body, NotGiven
from openai.resources.chat.completions.completions import (
    AsyncCompletions as BaseAsyncCompletions,
)
from openai.resources.responses.responses import AsyncResponses as BaseAsyncResponses
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_tool_choice_option_param import (
    ChatCompletionToolChoiceOptionParam,
)
from openai.types.completion_usage import CompletionUsage
from openai.types.responses import ResponseInputItemParam, response_create_params
from openai.types.responses.response import Response
from openai.types.responses.response_input_param import ResponseInputParam
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_output_text import ResponseOutputText
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
    ResponseUsage,
)
from openai.types.responses.tool_param import ToolParam
from openai.types.shared_params.metadata import Metadata

from areal.api.cli_args import GenerationHyperparameters
from areal.api.io_struct import ModelRequest
from areal.experimental.openai.cache import CompletionCache
from areal.experimental.openai.tool_call_parser import process_tool_calls
from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.utils import logging

if TYPE_CHECKING:
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

    from areal.api.engine_api import InferenceEngine

# reset OpenAI keys when using the wrapped client.
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "none")
os.environ["OPENAI_BASE_URL"] = os.environ.get("OPENAI_BASE_URL", "none")

logger = logging.getLogger("AReaLOpenAI Client")


class AsyncCompletionsWithReward(BaseAsyncCompletions):
    """Extended AsyncCompletions that adds caching and reward functionality."""

    # Class-level set to track which parameters have been warned about
    # (shared across all instances)
    _warned_parameters: set[str] = set()

    def __init__(
        self,
        client,
        engine: "InferenceEngine",
        tokenizer: "PreTrainedTokenizerFast",
        cache: CompletionCache,
        tool_call_parser: str | None = None,
        chat_template_type: str = "hf",
        messages_delimiter_start: str = "<|im_start|>",
        messages_delimiter_end: str = "<|im_end|>",
    ):
        super().__init__(client)
        self.engine = engine
        self.tokenizer = tokenizer
        self.tool_call_parser = tool_call_parser
        self._cache = cache
        self.chat_template_type = chat_template_type
        self.messages_delimiter_start = messages_delimiter_start
        self.messages_delimiter_end = messages_delimiter_end

    async def create(
        self,
        *,
        messages: Iterable[ChatCompletionMessageParam],
        frequency_penalty: float | None | NotGiven = NOT_GIVEN,
        max_completion_tokens: int | None | NotGiven = NOT_GIVEN,
        max_tokens: int | None | NotGiven = NOT_GIVEN,
        metadata: Metadata | None | NotGiven = NOT_GIVEN,
        stop: str | None | list[str] | None | NotGiven = NOT_GIVEN,
        store: bool | None | NotGiven = NOT_GIVEN,
        temperature: float | None | NotGiven = NOT_GIVEN,
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN,
        tools: Iterable[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
        top_p: float | None | NotGiven = NOT_GIVEN,
        extra_body: Body | None = None,
        areal_completion_cache: CompletionCache | None = None,
        **kwargs: Any,
    ) -> ChatCompletion:
        """Override create method to use AReaL engine and cache responses."""
        # Extract and validate supported parameters
        messages_list = list(messages)
        if not messages_list:
            raise ValueError("messages cannot be empty")
        if extra_body is None:
            extra_body = {}
        # Convert messages to prompt format
        tools = tools if not is_omitted(tools) else None
        if self.chat_template_type == "hf":
            prompt_token_ids = self.tokenizer.apply_chat_template(
                messages_list,
                tools=tools,
                add_generation_prompt=True,
                tokenize=True,
                **extra_body.get("chat_template_kwargs", {}),
            )
        elif self.chat_template_type == "concat":
            # By default, follows Qwen3 chat template.
            start, end = self.messages_delimiter_start, self.messages_delimiter_end
            message_strs = []
            for msg in messages_list:
                message_strs.append(f"{start}{msg['role']}\n{msg['content']}{end}\n")
            message_strs.append(f"{start}assistant\n")
            prompt_token_ids = self.tokenizer.encode("".join(message_strs))
        else:
            raise ValueError(
                f"Unsupported chat_template_type {self.chat_template_type}"
            )

        temp = 1.0 if is_omitted(temperature) else (temperature or 0.0)
        max_new_tokens = None
        if not is_omitted(max_tokens):
            max_new_tokens = max_tokens - len(prompt_token_ids)
            if max_new_tokens <= 0:
                raise RuntimeError(
                    "max_tokens must be greater than the number of prompt tokens"
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

        # Create generation config
        gconfig = GenerationHyperparameters(
            n_samples=1,
            temperature=temp,
            max_new_tokens=max_new_tokens,
            top_p=top_p_val,
            stop=stop_tokens,
            greedy=temp == 0,
            frequency_penalty=frequency_penalty,
            stop_token_ids=list(
                set([self.tokenizer.eos_token_id, self.tokenizer.pad_token_id])
            ),
        )

        model_request = ModelRequest(
            input_ids=prompt_token_ids,
            gconfig=gconfig,
            rid=str(uuid.uuid4()),
            metadata=metadata if not is_omitted(metadata) else {},
            tokenizer=self.tokenizer,
        )

        # Call inference engine
        response = await self.engine.agenerate(model_request)

        # Convert response to OpenAI format
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        current_time = int(datetime.datetime.now().timestamp())

        output_text = self.tokenizer.decode(response.output_tokens)

        # Parse tool calls.
        tool_calls = None
        if tool_choice != "none" and tools:
            tool_calls, output_text, response.stop_reason = process_tool_calls(
                output_text,
                tools,
                self.tool_call_parser,
                response.stop_reason,
            )

        # Create proper ChatCompletion object with all required fields
        chat_completion = ChatCompletion(
            id=completion_id,
            choices=[
                Choice(
                    finish_reason=response.stop_reason,
                    index=0,
                    logprobs=None,  # For simplicity
                    message=ChatCompletionMessage(
                        content=output_text,
                        role="assistant",
                        tool_calls=tool_calls,
                    ),
                )
            ],
            created=current_time,
            model="None",
            object="chat.completion",
            service_tier=None,
            system_fingerprint=None,
            usage=CompletionUsage(
                completion_tokens=len(response.output_tokens),
                prompt_tokens=len(response.input_tokens),
                total_tokens=len(response.input_tokens) + len(response.output_tokens),
            ),
        )

        if is_omitted(store) or store:
            # Cache the completion with its input messages
            cache = (
                areal_completion_cache
                if areal_completion_cache is not None
                else self._cache
            )
            if completion_id in cache:
                raise ValueError(f"Completion {completion_id} already exists in cache")

            cache[completion_id] = InteractionWithTokenLogpReward(
                completion=deepcopy(chat_completion),
                model_response=response,  # Should not deepcopy response because of tokenizer
                messages=deepcopy(messages_list),  # Store a copy of the input messages
                chat_template_type=self.chat_template_type,
            )
        return chat_completion


class AsyncResponsesWithReward(BaseAsyncResponses):
    """Extended AsyncResponses that adds caching and reward functionality."""

    def __init__(
        self,
        client,
        engine: "InferenceEngine",
        tokenizer: "PreTrainedTokenizerFast",
        cache: CompletionCache,
        tool_call_parser: str | None = None,
        chat_template_type: str = "hf",
        messages_delimiter_start: str = "<|im_start|>",
        messages_delimiter_end: str = "<|im_end|>",
    ):
        super().__init__(client)
        self.engine = engine
        self.tokenizer = tokenizer
        self.tool_call_parser = tool_call_parser
        self._cache = cache
        self.chat_template_type = chat_template_type
        self.messages_delimiter_start = messages_delimiter_start
        self.messages_delimiter_end = messages_delimiter_end

    async def create(
        self,
        *,
        input: str | ResponseInputParam | NotGiven = NOT_GIVEN,
        instructions: str | None | NotGiven = NOT_GIVEN,
        max_output_tokens: int | None | NotGiven = NOT_GIVEN,
        metadata: Metadata | None | NotGiven = NOT_GIVEN,
        tool_choice: response_create_params.ToolChoice | NotGiven = NOT_GIVEN,
        tools: Iterable[ToolParam] | NotGiven = NOT_GIVEN,
        temperature: float | None | NotGiven = NOT_GIVEN,
        top_p: float | None | NotGiven = NOT_GIVEN,
        extra_body: Body | None = None,
        areal_response_cache: dict[str, InteractionWithTokenLogpReward] | None = None,
        **kwargs: Any,
    ) -> Response:
        """Override create method to use AReaL engine"""
        if extra_body is None:
            extra_body = {}

        # Build a simple messages list compatible with tokenizer chat template
        messages_list: list[dict] = []
        if not is_omitted(instructions):
            messages_list = [
                {"role": "system", "content": instructions},
            ]
        if is_omitted(input):
            raise ValueError("input is required for Responses.create")

        def _convert_tool_output_format(
            item: dict,
        ) -> ChatCompletionToolMessageParam | dict:
            """Convert custom tool output format to standard chat template format.

            Converts openai.types.responses.response_input_item_param.FunctionCallOutput
            to openai.types.chat.ChatCompletionToolMessageParam.

            Args:
                item: Input dict, could be FunctionCallOutput from openai-agents SDK
                      with format: {'call_id': str, 'output': str, 'type': 'function_call_output'}

            Returns:
                ChatCompletionToolMessageParam (TypedDict) with format:
                {'role': 'tool', 'content': str, 'tool_call_id': str}
                or the original dict if conversion is not needed.
            """
            if (
                isinstance(item, dict)
                and "output" in item
                and item.get("type") == "function_call_output"
            ):
                converted = {
                    "role": "tool",
                    "content": item["output"],
                }
                # Add tool_call_id if present
                if "call_id" in item:
                    converted["tool_call_id"] = item["call_id"]
                return converted
            return item

        def _build_messages_list(item: ResponseInputItemParam) -> list[dict]:
            messages_list = []
            if "content" in item:
                if isinstance(item["content"], str):
                    messages_list.append(
                        {"role": item["role"], "content": item["content"]},
                    )
                elif isinstance(item["content"], Iterable):
                    for content in item["content"]:
                        if (
                            isinstance(content, dict)
                            and content.get("type") == "output_text"
                            and "text" in content
                        ):
                            messages_list.append(
                                {"role": item["role"], "content": content["text"]},
                            )
                        else:
                            raise ValueError("Unsupported content format")
                else:
                    raise ValueError("Unsupported input item format")
            else:
                # Convert tool output format if needed
                converted = _convert_tool_output_format(item)
                messages_list.append(deepcopy(converted))
            return messages_list

        if isinstance(input, str):
            messages_list += [
                {"role": "user", "content": input},
            ]
        elif isinstance(input, list):
            for item in input:
                messages_list += _build_messages_list(item)
        else:
            raise ValueError(
                "Unsupported Responses input format: "
                "expected str or list of message items with input_text."
            )

        # Apply chat template
        tools = list(tools) if not is_omitted(tools) else None
        if self.chat_template_type == "hf":
            prompt_token_ids = self.tokenizer.apply_chat_template(
                messages_list,
                tools=tools,
                add_generation_prompt=True,
                tokenize=True,
                **extra_body.get("chat_template_kwargs", {}),
            )
        elif self.chat_template_type == "concat":
            # By default, follows Qwen3 chat template.
            start, end = self.messages_delimiter_start, self.messages_delimiter_end
            message_strs: list[str] = []
            for msg in messages_list:
                message_strs.append(f"{start}{msg['role']}\n{msg['content']}{end}\n")
            message_strs.append(f"{start}assistant\n")
            prompt_token_ids = self.tokenizer.encode("".join(message_strs))
        else:
            raise ValueError(
                f"Unsupported chat_template_type {self.chat_template_type}"
            )

        # Map sampling params
        temp = 1.0 if is_omitted(temperature) else (temperature or 0.0)
        top_p_val = 1.0 if is_omitted(top_p) else (top_p or 1.0)
        max_new_tokens = None
        if not is_omitted(max_output_tokens):
            max_new_tokens = max_output_tokens
        if max_new_tokens is None:
            max_new_tokens = 512  # Default value
            logger.warning("max_output_tokens not specified, defaulting to 512.")

        stop = kwargs.get("stop", None)
        frequency_penalty = kwargs.get("frequency_penalty", 0.0)

        # Create generation config and request
        gconfig = GenerationHyperparameters(
            n_samples=1,
            temperature=temp,
            max_new_tokens=max_new_tokens,
            top_p=top_p_val,
            stop=stop,
            greedy=temp == 0,
            frequency_penalty=frequency_penalty,
            stop_token_ids=list(
                set([self.tokenizer.eos_token_id, self.tokenizer.pad_token_id])
            ),
        )

        model_request = ModelRequest(
            input_ids=prompt_token_ids,
            gconfig=gconfig,
            rid=str(uuid.uuid4()),
            metadata=metadata if not is_omitted(metadata) else {},
            tokenizer=self.tokenizer,
        )

        # Call inference engine
        engine_resp = await self.engine.agenerate(model_request)
        output_text = self.tokenizer.decode(engine_resp.output_tokens)

        # Parse tool calls.
        tool_calls = None
        if not is_omitted(tool_choice) and tool_choice != "none" and tools:
            tool_calls, output_text, engine_resp.stop_reason = process_tool_calls(
                output_text,
                tools,
                self.tool_call_parser,
                engine_resp.stop_reason,
                use_responses=True,
            )

        # Extract reasoning tokens from output
        reasoning_token_count = self._count_reasoning_tokens(output_text)

        # Build Responses API objects
        resp_id = f"resp-{uuid.uuid4().hex[:29]}"
        msg_id = f"msg-{uuid.uuid4().hex[:29]}"
        current_time = float(int(datetime.datetime.now().timestamp()))

        output_message = ResponseOutputMessage(
            id=msg_id,
            role="assistant",
            status="completed",
            type="message",
            content=[
                ResponseOutputText(
                    annotations=[],
                    text=output_text,
                    type="output_text",
                )
            ],
        )

        if tool_calls:
            resp_output = tool_calls
        else:
            resp_output = [output_message]

        usage = ResponseUsage(
            input_tokens=len(engine_resp.input_tokens),
            input_tokens_details=InputTokensDetails(cached_tokens=0),
            output_tokens=len(engine_resp.output_tokens),
            output_tokens_details=OutputTokensDetails(
                reasoning_tokens=reasoning_token_count
            ),
            total_tokens=len(engine_resp.input_tokens) + len(engine_resp.output_tokens),
        )

        response = Response(
            id=resp_id,
            created_at=current_time,
            error=None,
            incomplete_details=None,
            instructions=None if is_omitted(instructions) else instructions,
            metadata=None if is_omitted(metadata) else metadata,
            model="None",
            object="response",
            output=resp_output,
            parallel_tool_calls=False,
            temperature=temp,
            tool_choice=tool_choice if not is_omitted(tool_choice) else "none",
            tools=tools,
            top_p=top_p_val,
            background=None,
            conversation=None,
            max_output_tokens=max_new_tokens,
            max_tool_calls=None,
            previous_response_id=None,
            prompt=None,
            prompt_cache_key=None,
            reasoning=None,
            safety_identifier=None,
            service_tier=None,
            status="completed",
            text=None,
            top_logprobs=None,
            truncation=None,
            usage=usage,
            user=None,
        )

        # Cache the response with its input data
        cache = (
            areal_response_cache if areal_response_cache is not None else self._cache
        )

        if resp_id in cache:
            raise ValueError(f"Response {resp_id} already exists in cache")

        cache[resp_id] = InteractionWithTokenLogpReward(
            response=deepcopy(response),
            model_response=engine_resp,  # Should not deepcopy because of tokenizer
            input_data=(
                deepcopy(input) if not is_omitted(input) else ""
            ),  # Store a copy of the input data
            chat_template_type=self.chat_template_type,
        )

        return response

    def _count_reasoning_tokens(
        self,
        output_text: str,
        thinking_start_token: str = "<think>",
        thinking_end_token: str = "</think>",
    ) -> int:
        """
        Count reasoning tokens from output text by extracting content within thinking start and end tokens.
        """

        if thinking_start_token not in output_text:
            return 0
        processed_text = output_text.split(thinking_start_token, maxsplit=1)[1]
        if thinking_end_token in processed_text:
            processed_text = processed_text.split(thinking_end_token, maxsplit=1)[0]
        return len(self.tokenizer.encode(processed_text, add_special_tokens=False))


class ArealOpenAI(AsyncOpenAI):
    """
    Extended AsyncOpenAI client that uses AReaL's inference engine
    and supports reward setting.
    """

    def __init__(
        self,
        engine: "InferenceEngine",
        tokenizer: "PreTrainedTokenizerFast",
        tool_call_parser: str | None = None,
        chat_template_type: str = "hf",
        messages_delimiter_start: str = "<|im_start|>",
        messages_delimiter_end: str = "<|im_end|>",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.engine = engine
        self.tokenizer = tokenizer
        self.tool_call_parser = tool_call_parser

        # Use an ordered dict to maintain insertion order of completions/responses
        self._cache: CompletionCache = CompletionCache()

        # Override responses with our extended implementation
        self.responses = AsyncResponsesWithReward(
            self,
            engine,
            tokenizer,
            self._cache,
            tool_call_parser=self.tool_call_parser,
            chat_template_type=chat_template_type,
            messages_delimiter_start=messages_delimiter_start,
            messages_delimiter_end=messages_delimiter_end,
        )

        # Override chat.completions with our extended implementation
        self.chat.completions = AsyncCompletionsWithReward(
            self,
            engine,
            tokenizer,
            self._cache,
            tool_call_parser=self.tool_call_parser,
            chat_template_type=chat_template_type,
            messages_delimiter_start=messages_delimiter_start,
            messages_delimiter_end=messages_delimiter_end,
        )

    def get_interaction(self, id: str) -> InteractionWithTokenLogpReward | None:
        """Get completion/response with its reward from cache."""
        return self._cache.get(id)

    def get_completions(self, id: str) -> InteractionWithTokenLogpReward | None:
        logger.warning(
            "get_completions is deprecated. Please use get_interaction instead."
        )
        return self.get_interaction(id)

    def get_responses(self, id: str) -> InteractionWithTokenLogpReward | None:
        logger.warning(
            "get_responses is deprecated. Please use get_interaction instead."
        )
        return self.get_interaction(id)

    def set_reward(self, id: str, reward: float) -> None:
        """Set reward for a specific completion/response by its ID."""
        if id not in self._cache:
            raise KeyError(f"Interaction with ID {id} not found in cache")
        return self._cache.set_reward(id, reward)

    def set_final_reward(self, reward: float) -> None:
        """Set reward for the most recent completion/response."""
        if not self._cache:
            raise RuntimeError("No interaction in cache to set reward for")
        return self._cache.set_final_reward(reward)

    def apply_reward_discount(self, turn_discount: float = 1.0) -> None:
        """Apply backward discounted rewards across cached completions/responses.

        This method iterates over the cached completions/responses in reverse creation
        (insertion) order and applies a geometric discount to propagate reward
        signal backward in time. The most recent completion/response is treated as the
        starting point. If it does not have an explicit reward, a warning is
        logged and a default reward of ``0.0`` is used. For each earlier
        completion/response, its reward is initialized to ``0.0`` if unset, then the
        discounted reward from the next later completion/response is added:

        ``reward[i] += reward[i+1] * turn_discount``.

        Typically called before exporting completions/responses in 'individual' style
        to each completion/response is assigned with a valid reward value.

        Parameters
        ----------
        turn_discount : float, optional
            The per-turn discount factor applied when propagating reward
            backward from a later completion/response to an earlier one, by default 1.0.

        Returns
        -------
        Dict[str, InteractionWithTokenLogpReward]
            A shallow copy of the completion/response cache after rewards have been
            updated in-place.
        """
        return self._cache.apply_reward_discount(turn_discount)

    def export_interactions(
        self, style: str
    ) -> dict[str, InteractionWithTokenLogpReward]:
        """Export cached completions/responses in different formats.

        When ``style='concat'``, this method constructs a conversation tree by
        linking completions/responses whose input message lists form a strict-prefix
        relationship. The longest-prefix rule is used to determine each node's
        parent. It then returns only leaf-node completions/responses (those without
        children). No reward propagation is performed here.

        When ``style='individual'``, all cached completions/responses are returned as-is
        without constructing the tree.

        Parameters
        ----------
        style : str, optional
            The export style, either ``'concat'`` (build tree and return leaves)
            or ``'individual'`` (return all), by default 'concat'.

        Returns
        -------
        Dict[str, InteractionWithTokenLogpReward]
            A mapping from completion/response ID to completion/response objects. For
            ``'concat'``, this contains only leaf nodes. For ``'individual'``,
            this contains all cached completions/responses.

        Raises
        ------
        ValueError
            If an unsupported ``style`` is provided.
        """
        return self._cache.export_interactions(style)

    def export_completions(
        self, style: str = "concat"
    ) -> dict[str, InteractionWithTokenLogpReward]:
        logger.warning(
            "export_completions is deprecated. Please use export_interactions instead."
        )
        return self.export_interactions(style)

    def export_responses(
        self, style: str = "concat"
    ) -> dict[str, InteractionWithTokenLogpReward]:
        logger.warning(
            "export_responses is deprecated. Please use export_interactions instead."
        )
        return self.export_interactions(style)


def is_omitted(value) -> bool:
    """Check if a value is NOT_GIVEN or Omit type or None."""
    if value is NOT_GIVEN or value is None:
        return True
    # Use isinstance for type safety and robustness
    # Check for common omitted types from OpenAI SDK
    try:
        from openai import Omit

        if isinstance(value, Omit):
            return True
    except ImportError:
        pass

    # Fallback for other omit types
    if hasattr(value, "__class__"):
        return value.__class__.__name__ in ("NotGiven", "Omit")
    return False
