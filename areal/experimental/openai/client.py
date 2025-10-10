import datetime
import os
import uuid
from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Set, Union

from openai import AsyncOpenAI
from openai._types import NOT_GIVEN, Body, NotGiven
from openai.resources.chat.completions.completions import (
    AsyncCompletions as BaseAsyncCompletions,
)
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_tool_choice_option_param import (
    ChatCompletionToolChoiceOptionParam,
)
from openai.types.completion_usage import CompletionUsage
from openai.types.shared_params.metadata import Metadata

from areal.api.cli_args import GenerationHyperparameters
from areal.api.io_struct import ModelRequest
from areal.experimental.openai.tool_call_parser import process_tool_calls
from areal.experimental.openai.types import CompletionWithTokenLogpReward
from areal.utils import logging

if TYPE_CHECKING:
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

    from areal.api.engine_api import InferenceEngine

# reset OpenAI keys when using the wrapped client.
os.environ["OPENAI_API_KEY"] = "none"
os.environ["OPENAI_BASE_URL"] = "none"

logger = logging.getLogger("AReaLOpenAI Client")


class AsyncCompletionsWithReward(BaseAsyncCompletions):
    """Extended AsyncCompletions that adds caching and reward functionality."""

    # Class-level set to track which parameters have been warned about (shared across all instances)
    _warned_parameters: Set[str] = set()

    def __init__(
        self,
        client,
        engine: "InferenceEngine",
        tokenizer: "PreTrainedTokenizerFast",
        cache: Dict[str, CompletionWithTokenLogpReward],
        tool_call_parser: Optional[str] = None,
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
        frequency_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        max_completion_tokens: Optional[int] | NotGiven = NOT_GIVEN,
        max_tokens: Optional[int] | NotGiven = NOT_GIVEN,
        metadata: Optional[Metadata] | NotGiven = NOT_GIVEN,
        stop: Union[Optional[str], List[str], None] | NotGiven = NOT_GIVEN,
        store: Optional[bool] | NotGiven = NOT_GIVEN,
        temperature: Optional[float] | NotGiven = NOT_GIVEN,
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN,
        tools: Iterable[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
        top_p: Optional[float] | NotGiven = NOT_GIVEN,
        extra_body: Body | None = None,
    ) -> ChatCompletion:
        """Override create method to use AReaL engine and cache responses."""
        # Extract and validate supported parameters
        messages_list = list(messages)
        if not messages_list:
            raise ValueError("messages cannot be empty")
        if extra_body is None:
            extra_body = {}
        # Convert messages to prompt format
        tools = tools if tools is not NOT_GIVEN else None
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

        temp = 1.0 if temperature is NOT_GIVEN else (temperature or 0.0)
        max_new_tokens = 512
        if max_tokens is not NOT_GIVEN and max_tokens is not None:
            max_new_tokens = max_tokens - len(prompt_token_ids)
            if max_new_tokens <= 0:
                raise RuntimeError(
                    "max_tokens must be greater than the number of prompt tokens"
                )
        if max_completion_tokens is not NOT_GIVEN and max_completion_tokens is not None:
            max_new_tokens = min(max_new_tokens, max_completion_tokens)

        top_p_val = 1.0 if top_p is NOT_GIVEN else (top_p or 1.0)
        stop_tokens = None if stop is NOT_GIVEN else stop

        if frequency_penalty is NOT_GIVEN or frequency_penalty is None:
            frequency_penalty = 0.0

        # Create generation config
        gconfig = GenerationHyperparameters(
            n_samples=1,
            temperature=temp,
            max_new_tokens=max_new_tokens,
            top_p=top_p_val,
            stop=(
                stop_tokens
                if isinstance(stop_tokens, list)
                else [stop_tokens] if stop_tokens else None
            ),
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
            metadata=metadata,
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

        if store is NOT_GIVEN or store:
            # Cache the completion with its input messages
            self._cache[completion_id] = CompletionWithTokenLogpReward(
                completion=deepcopy(chat_completion),
                response=response,  # Should not deepcopy response because of tokenizer
                messages=deepcopy(messages_list),  # Store a copy of the input messages
                chat_template_type=self.chat_template_type,
            )
        return chat_completion


class ArealOpenAI(AsyncOpenAI):
    """Extended AsyncOpenAI client that uses AReaL's inference engine and supports reward setting."""

    def __init__(
        self,
        engine: "InferenceEngine",
        tokenizer: "PreTrainedTokenizerFast",
        tool_call_parser: Optional[str] = None,
        chat_template_type: str = "hf",
        messages_delimiter_start: str = "<|im_start|>",
        messages_delimiter_end: str = "<|im_end|>",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.engine = engine
        self.tokenizer = tokenizer
        self.tool_call_parser = tool_call_parser
        # Use an ordered dict to maintain insertion order of completions
        self._completion_cache: OrderedDict[str, CompletionWithTokenLogpReward] = (
            OrderedDict()
        )

        # Override chat.completions with our extended implementation
        self.chat.completions = AsyncCompletionsWithReward(
            self,
            engine,
            tokenizer,
            self._completion_cache,
            tool_call_parser=self.tool_call_parser,
            chat_template_type=chat_template_type,
            messages_delimiter_start=messages_delimiter_start,
            messages_delimiter_end=messages_delimiter_end,
        )

    def get_completions(
        self, completion_id: str
    ) -> Optional[CompletionWithTokenLogpReward]:
        """Get completion with its reward from cache."""
        return self._completion_cache.get(completion_id)

    def set_reward(self, completion_id: str, reward: float) -> None:
        """Set reward for a specific completion by its ID."""
        if completion_id not in self._completion_cache:
            raise KeyError(f"Completion with ID {completion_id} not found in cache")
        self._completion_cache[completion_id].reward = reward

    def apply_reward_discount(self, turn_discount: float = 1.0) -> None:
        """Apply backward discounted rewards across cached completions.

        This method iterates over the cached completions in reverse creation
        (insertion) order and applies a geometric discount to propagate reward
        signal backward in time. The most recent completion is treated as the
        starting point. If it does not have an explicit reward, a warning is
        logged and a default reward of ``0.0`` is used. For each earlier
        completion, its reward is initialized to ``0.0`` if unset, then the
        discounted reward from the next later completion is added:

        ``reward[i] += reward[i+1] * turn_discount``.

        Typically called before exporting completions in 'individual' style
        to each completion is assigned with a valid reward value.

        Parameters
        ----------
        turn_discount : float, optional
            The per-turn discount factor applied when propagating reward
            backward from a later completion to an earlier one, by default 1.0.

        Returns
        -------
        Dict[str, CompletionWithTokenLogpReward]
            A shallow copy of the completion cache after rewards have been
            updated in-place.
        """
        # Assign rewards to completions in cache based on their created time
        comp_time_sequence = list(
            reversed([comp for _, comp in self._completion_cache.items()])
        )
        # Check if the last-created completion has a reward set
        if comp_time_sequence:
            if comp_time_sequence[0].reward is None:
                logger.warning(
                    "The most recent completion does not have a reward set. "
                    "All completions will have None reward."
                )
                comp_time_sequence[0].reward = 0.0
            # Propagate rewards backwards with discounting if reward is not set
            for i in range(1, len(comp_time_sequence)):
                if comp_time_sequence[i].reward is None:
                    comp_time_sequence[i].reward = 0.0
                comp_time_sequence[i].reward += (
                    comp_time_sequence[i - 1].reward * turn_discount
                )
        return dict(**self._completion_cache)

    def export_completions(
        self, style: str = "concat"
    ) -> Dict[str, CompletionWithTokenLogpReward]:
        """Export cached completions in different formats.

        When ``style='concat'``, this method constructs a conversation tree by
        linking completions whose input message lists form a strict-prefix
        relationship. The longest-prefix rule is used to determine each node's
        parent. It then returns only leaf-node completions (those without
        children). No reward propagation is performed here.

        When ``style='individual'``, all cached completions are returned as-is
        without constructing the tree.

        Parameters
        ----------
        style : str, optional
            The export style, either ``'concat'`` (build tree and return leaves)
            or ``'individual'`` (return all), by default 'concat'.

        Returns
        -------
        Dict[str, CompletionWithTokenLogpReward]
            A mapping from completion ID to completion objects. For
            ``'concat'``, this contains only leaf nodes. For ``'individual'``,
            this contains all cached completions.

        Raises
        ------
        ValueError
            If an unsupported ``style`` is provided.
        """
        if len(self._completion_cache) == 0:
            return {}

        if style == "concat":
            for comp in self._completion_cache.values():
                if comp.chat_template_type != "concat":
                    raise ValueError(
                        "Cannot export completions in 'concat' style when "
                        'comp.chat_template_type != "concat" for any completion. '
                        "This is because when applying chat template using some tokenizers, "
                        "there might be some tokens added or removed (e.g. think tokens), "
                        "making it impossible to construct the conversation tree. "
                        "Please use 'individual' style instead."
                    )

            def _is_prefix(a: List[Dict], b: List[Dict]) -> bool:
                # True if a is a strict prefix of b
                if len(a) >= len(b):
                    return False
                for i in range(len(a)):
                    if a[i] != b[i]:
                        return False
                return True

            # Precompute normalized messages
            meta = {}
            for cid, comp in self._completion_cache.items():
                meta[cid] = {
                    "norm_msgs": comp.messages or [],
                    "obj": comp,
                }

            # 1) Construct parent-child relationships using longest prefix rule
            # Sort potential children by (message length asc, created asc) so parents are available
            ordered = sorted(
                meta.items(),
                key=lambda kv: (
                    len(kv[1]["norm_msgs"]),
                    kv[1]["obj"].completion.created,
                ),
            )

            # Reset parents before rebuilding
            for _, info in ordered:
                info["obj"].parent = None

            for child_id, child_info in ordered:
                child_msgs = child_info["norm_msgs"]
                best_parent = None
                best_len = -1
                for parent_id, parent_info in ordered:
                    if parent_id == child_id:
                        continue
                    parent_msgs = parent_info["norm_msgs"]
                    if _is_prefix(parent_msgs, child_msgs):
                        plen = len(parent_msgs)
                        # choose the longest prefix
                        if plen > best_len:
                            best_parent = parent_info["obj"]
                            best_len = plen
                child_info["obj"].parent = best_parent

            # Build children mapping to find leaf nodes.
            children_map: Dict[str, List[CompletionWithTokenLogpReward]] = defaultdict(
                list
            )
            for _, info in meta.items():
                obj = info["obj"]
                if obj.parent is not None:
                    children_map[obj.parent.completion.id].append(obj)

            # Return only leaf nodes (nodes without children)
            parents_with_children = set(children_map.keys())
            leaf_only: Dict[str, CompletionWithTokenLogpReward] = {}
            for cid, info in meta.items():
                obj = info["obj"]
                if obj.completion.id not in parents_with_children:
                    leaf_only[cid] = obj
            return leaf_only
        elif style == "individual":
            return dict(**self._completion_cache)
        else:
            raise ValueError(f"Invalid export completions style {style}")
