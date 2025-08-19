import os
import time
import uuid
from collections import defaultdict
from copy import deepcopy
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
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

if TYPE_CHECKING:
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

    from areal.api.engine_api import InferenceEngine

# reset OpenAI keys when using the wrapped client.
os.environ["OPENAI_API_KEY"] = "none"
os.environ["OPENAI_BASE_URL"] = "none"


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
    ):
        super().__init__(client)
        self.engine = engine
        self.tokenizer = tokenizer
        self.tool_call_parser = tool_call_parser
        self._cache = cache

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
        prompt_token_ids = self.tokenizer.apply_chat_template(
            messages_list,
            tools=tools,
            add_generation_prompt=True,
            tokenize=True,
            **extra_body.get("chat_template_kwargs", {}),
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
        current_time = int(time.time())

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
            )
        return chat_completion


class ArealOpenAI(AsyncOpenAI):
    """Extended AsyncOpenAI client that uses AReaL's inference engine and supports reward setting."""

    def __init__(
        self,
        engine: "InferenceEngine",
        tokenizer: "PreTrainedTokenizerFast",
        tool_call_parser: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.engine = engine
        self.tokenizer = tokenizer
        self.tool_call_parser = tool_call_parser
        self._completion_cache: Dict[str, CompletionWithTokenLogpReward] = {}

        # Override chat.completions with our extended implementation
        self.chat.completions = AsyncCompletionsWithReward(
            self,
            engine,
            tokenizer,
            self._completion_cache,
            tool_call_parser=self.tool_call_parser,
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

    def export_completions(
        self, turn_discount: float = 1.0
    ) -> Dict[str, CompletionWithTokenLogpReward]:
        """Export all completions with rewards after backpropagation."""

        # Step 1: Build tree structure based on role prefix relationships
        children = defaultdict(list)  # parent_id -> [child_ids]
        parents = {}  # child_id -> parent_id
        roots = set()  # completion_ids with no parents

        # Convert messages to role sequence for easier comparison
        def get_role_sequence(messages: List[dict]) -> Tuple[str, ...]:
            return tuple(msg.get("role", "user") for msg in messages)

        # Build role sequences for all completions
        completion_roles = {}
        for completion_id, completion_data in self._completion_cache.items():
            completion_roles[completion_id] = get_role_sequence(
                completion_data.messages
            )

        # Find parent-child relationships based on role prefix matching
        completion_ids = list(self._completion_cache.keys())
        for i, child_id in enumerate(completion_ids):
            child_roles = completion_roles[child_id]
            parent_found = False

            for j, potential_parent_id in enumerate(completion_ids):
                if i == j:
                    continue
                parent_roles = completion_roles[potential_parent_id]

                # Check if parent_roles is a prefix of child_roles
                if (
                    len(parent_roles) < len(child_roles)
                    and child_roles[: len(parent_roles)] == parent_roles
                ):

                    # Find the best parent (longest prefix)
                    if child_id not in parents or len(parent_roles) >= len(
                        completion_roles[parents[child_id]]
                    ):
                        # Remove from previous parent if exists
                        if child_id in parents and len(parent_roles) > len(
                            completion_roles[parents[child_id]]
                        ):
                            old_parent = parents[child_id]
                            children[old_parent].remove(child_id)

                        parents[child_id] = potential_parent_id
                        children[potential_parent_id].append(child_id)
                        parent_found = True

            if not parent_found:
                roots.add(child_id)

        # Step 2: Perform topological sorting on each tree
        def topological_sort_tree(root_id: str) -> List[str]:
            """Perform DFS-based topological sort starting from root."""
            visited = set()
            stack = []

            def dfs(node_id: str):
                if node_id in visited:
                    return
                visited.add(node_id)

                # Visit all children first (post-order)
                for child_id in children[node_id]:
                    dfs(child_id)

                # Add current node to stack after visiting children
                stack.append(node_id)

            dfs(root_id)
            return stack

        # Get topological order for all trees
        topo_order = []
        for root_id in roots:
            topo_order.extend(topological_sort_tree(root_id))

        # Step 3: Backpropagate rewards from leaves to roots
        # Process nodes in topological order (leaves first, since topological_sort_tree returns post-order)
        for completion_id in topo_order:
            completion_data = self._completion_cache[completion_id]

            # If this is a leaf node with a reward, keep it
            if len(children[completion_id]) == 0:
                # This is a leaf - its reward should already be set
                if completion_data.reward is None:
                    completion_data.reward = 0
                continue

            # Calculate reward as discounted sum from children
            child_rewards = [
                self._completion_cache[child_id].reward
                for child_id in children[completion_id]
            ]
            if completion_data.reward is None:
                completion_data.reward = 0.0
            # Find the highest level/the minimum reward
            # Do not overwrite the existing reward if explicitly set
            child_non_none_rewards = list(
                filter(lambda x: x is not None, child_rewards)
            )
            if len(child_non_none_rewards) > 0:
                completion_data.reward += turn_discount * np.mean(
                    child_non_none_rewards
                )

        return self._completion_cache.copy()
