import datetime
import socket
import uuid
from collections import OrderedDict, defaultdict
from copy import deepcopy
from threading import Lock
from typing import Dict, List, Optional, Tuple

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
)
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import ModelRequest
from areal.experimental.openai.tool_call_parser import process_tool_calls
from areal.experimental.openai.types import CompletionWithTokenLogpReward
from areal.utils import logging

logger = logging.getLogger("Agent Proxy Server")


class ProxyServer:
    def __init__(
        self,
        engine: InferenceEngine,
        tokenizer: PreTrainedTokenizerFast,
        config_max_tokens: int,
        tool_call_parser: Optional[str] = None,
        chat_template_type: str = "hf",
        messages_delimiter_start: str = "<|im_start|>",
        messages_delimiter_end: str = "<|im_end|>",
    ):
        self.engine = engine
        self.tokenizer = tokenizer
        self.cache: OrderedDict[
            str, Tuple[Lock, OrderedDict[str, CompletionWithTokenLogpReward]]
        ] = OrderedDict()  # task_id -> completion_id -> CompletionWithTokenLogpReward
        self.tool_call_parser = tool_call_parser
        self.chat_template_type = chat_template_type
        self.messages_delimiter_start = messages_delimiter_start
        self.messages_delimiter_end = messages_delimiter_end
        self.config_max_tokens = config_max_tokens

        self.port: int = None
        self.sock: socket.socket = None
        self.server: uvicorn.Server = None
        self.cache_lock = Lock()

    def run(self, sock: socket.socket):
        self.port = sock.getsockname()[1]
        self.sock = sock

        app = FastAPI()

        @app.post("/v1/{task_id}/chat/completions")
        async def chat_completions_proxy(request: Request, task_id: str):
            """
            代理 /v1/chat/completions 接口
            """
            request_data = await request.json()

            # 判断是否是流式请求
            is_stream = request_data.get("stream", False)
            if is_stream:
                return JSONResponse(
                    status_code=500, content={"error": "stream is not support"}
                )

            messages_list = request_data.get("messages", [])
            if not messages_list:
                return JSONResponse(
                    status_code=400, content={"error": "messages is required"}
                )

            extra_body = request_data.get("extra_body", {})
            tools = request_data.get("tools", None)
            temp = request_data.get("temperature", 1.0)
            top_p_val = request_data.get("top_p", 1.0)
            stop_sequences = request_data.get("stop", None)
            frequency_penalty = request_data.get("frequency_penalty", 0.0)
            metadata = request_data.get("metadata", None)
            tool_choice = request_data.get("tool_choice", None)

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
                    message_strs.append(
                        f"{start}{msg['role']}\n{msg['content']}{end}\n"
                    )
                message_strs.append(f"{start}assistant\n")
                prompt_token_ids = self.tokenizer.encode("".join(message_strs))
            else:
                raise ValueError(
                    f"Unsupported chat_template_type {self.chat_template_type}"
                )

            max_tokens = request_data.get("max_tokens", None)
            if max_tokens is None:
                max_tokens = self.config_max_tokens
            max_tokens = min(max_tokens, self.config_max_tokens)

            new_tokens_limit_by_max_tokens = max_tokens - len(prompt_token_ids)
            max_new_tokens = request_data.get("max_completion_tokens", None)
            if max_new_tokens is None:
                max_new_tokens = new_tokens_limit_by_max_tokens
            max_new_tokens = min(max_new_tokens, new_tokens_limit_by_max_tokens)

            # Create generation config
            gconfig = GenerationHyperparameters(
                n_samples=1,
                temperature=temp,
                max_new_tokens=max_new_tokens,
                top_p=top_p_val,
                stop=stop_sequences,
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
            if tool_choice and tool_choice != "none" and tools:
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
                    total_tokens=len(response.input_tokens)
                    + len(response.output_tokens),
                ),
            )

            with self.cache_lock:
                if task_id not in self.cache:
                    self.cache[task_id] = (Lock(), OrderedDict())
                task_lock, task_cache = self.cache[task_id]

            with task_lock:
                task_cache[completion_id] = CompletionWithTokenLogpReward(
                    completion=deepcopy(chat_completion),
                    response=response,  # Should not deepcopy response because of tokenizer
                    messages=deepcopy(
                        messages_list
                    ),  # Store a copy of the input messages
                    chat_template_type=self.chat_template_type,
                )

            return JSONResponse(status_code=200, content=chat_completion.model_dump())

        @app.post("/v1/{task_id}/final_reward")
        async def final_reward(request: Request, task_id: str):
            request_data = await request.json()
            reward = request_data.get("final_reward", 0.0)
            self.set_final_reward(task_id, reward)
            return JSONResponse(status_code=200, content={})

        @app.get("/health")
        @app.post("/health")
        async def health_check():
            return JSONResponse(status_code=200, content={"status": "ok"})

        config = uvicorn.Config(app=app, log_level="warning")
        self.server = uvicorn.Server(config=config)
        self.server.run(sockets=[self.sock])

    # Many of this code are copied from areal/experimental/openai/client.py
    # I only add lock for thread safety

    def get_completions(
        self, task_id: str, completion_id: str
    ) -> Optional[CompletionWithTokenLogpReward]:
        """Get completion with its reward from cache."""
        with self.cache_lock:
            lock_and_task_cache = self.cache.get(task_id, None)
            if not lock_and_task_cache:
                return None
            task_lock, task_cache = lock_and_task_cache

        with task_lock:
            return task_cache.get(completion_id, None)

    def set_reward(self, task_id: str, completion_id: str, reward: float) -> None:
        """Set reward for a specific completion by its ID."""
        with self.cache_lock:
            if task_id not in self.cache:
                raise KeyError(f"Task with ID {task_id} not found in cache")
            task_lock, task_cache = self.cache[task_id]
        with task_lock:
            if completion_id not in task_cache:
                raise KeyError(f"Completion with ID {completion_id} not found in cache")
            task_cache[completion_id].reward = reward

    def set_final_reward(self, task_id: str, reward: float) -> None:
        """Set reward for the most recent completion."""
        with self.cache_lock:
            if task_id not in self.cache:
                raise KeyError(f"Task with ID {task_id} not found in cache")
            task_lock, task_cache = self.cache[task_id]
        with task_lock:
            last_comp_id = next(reversed(task_cache))
            task_cache[last_comp_id].reward = reward

    def get_final_reward(self, task_id: str) -> Optional[float]:
        """Get reward for the most recent completion."""
        with self.cache_lock:
            if task_id not in self.cache:
                raise KeyError(f"Task with ID {task_id} not found in cache")
            task_lock, task_cache = self.cache[task_id]
        with task_lock:
            last_comp_id = next(reversed(task_cache))
            return task_cache[last_comp_id].reward

    def apply_reward_discount(self, task_id: str, turn_discount: float = 1.0) -> None:
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
        with self.cache_lock:
            if task_id not in self.cache:
                logger.warning(f"Task with ID {task_id} not found in cache")
                return {}
            task_lock, task_completion_cache = self.cache.get(task_id)

        with task_lock:
            # Assign rewards to completions in cache based on their created time
            comp_time_sequence = list(
                reversed([comp for _, comp in task_completion_cache.items()])
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
            return dict(**task_completion_cache)

    def export_completions(
        self, task_id: str, style: str = "concat"
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
        with self.cache_lock:
            if task_id not in self.cache:
                logger.warning(f"Task with ID {task_id} not found in cache")
                return {}
            task_lock, task_completion_cache = self.cache.pop(task_id, {})

        with task_lock:

            if len(task_completion_cache) == 0:
                logger.warning(f"No completions found for task ID {task_id}")
                return {}

            if style == "concat":
                for comp in task_completion_cache.values():
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
                for cid, comp in task_completion_cache.items():
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
                children_map: Dict[str, List[CompletionWithTokenLogpReward]] = (
                    defaultdict(list)
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
                return dict(**task_completion_cache)
            else:
                raise ValueError(f"Invalid export completions style {style}")

    def __del__(self):
        """Ensure socket is closed on deletion."""
        if self.server:
            self.server.should_exit = True
            self.server = None
        if self.sock:
            self.sock.close()
            self.sock = None
