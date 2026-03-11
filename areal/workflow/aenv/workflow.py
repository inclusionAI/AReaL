"""AEnvironment-integrated rollout workflow."""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from transformers import PreTrainedTokenizerFast

from areal import workflow_context
from areal.api import AsyncRewardWrapper, RolloutWorkflow
from areal.api.cli_args import GenerationHyperparameters
from areal.experimental.openai import ArealOpenAI
from areal.infra.aenv import (
    AenvConfig,
    AenvEnvironmentAdapter,
    normalize_openai_tools,
    parse_tool_arguments,
)
from areal.utils import logging, stats_tracker
from areal.utils.dynamic_import import import_from_string

logger = logging.getLogger("AenvWorkflow")


class AenvWorkflow(RolloutWorkflow):
    """RolloutWorkflow that executes model tool calls through AEnvironment."""

    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        aenv_config: AenvConfig | None = None,
        reward_fn: Callable[..., float]
        | Callable[..., Awaitable[float]]
        | str
        | None = None,
        max_turns: int = 8,
        export_style: Literal["individual", "concat"] = "individual",
        tool_choice: Literal["auto", "none"] = "auto",
        tool_call_parser: str = "qwen25",
        system_prompt: str | None = None,
    ):
        if isinstance(tokenizer, str):
            from areal.utils.hf_utils import load_hf_tokenizer

            tokenizer = load_hf_tokenizer(tokenizer)

        if isinstance(reward_fn, str):
            reward_fn = import_from_string(reward_fn)

        if max_turns <= 0:
            raise ValueError(f"max_turns must be positive, got {max_turns}")
        if export_style not in {"individual", "concat"}:
            raise ValueError(f"Invalid export_style: {export_style}")

        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(tokenizer)
        self.tokenizer = tokenizer
        self.aenv_config = aenv_config or AenvConfig()
        self.max_turns = max_turns
        self.export_style = export_style
        self.chat_template_type = "concat" if export_style == "concat" else "hf"
        self.tool_choice = tool_choice
        self.tool_call_parser = tool_call_parser
        self.system_prompt = system_prompt

        self._reward_fn_async: Callable[..., Awaitable[float]] | None = None
        self._reward_wrapper: AsyncRewardWrapper | None = None
        if reward_fn is not None:
            if self._is_async_callable(reward_fn):
                self._reward_fn_async = reward_fn
            else:
                self._reward_wrapper = AsyncRewardWrapper(reward_fn)

    async def arun_episode(self, engine, data: dict[str, Any]):
        client = ArealOpenAI(
            engine=engine,
            tokenizer=self.tokenizer,
            tool_call_parser=self.tool_call_parser,
            chat_template_type=self.chat_template_type,
        )
        messages = self._build_messages(data)
        model_turn_count = 0
        tool_round_count = 0

        async with AenvEnvironmentAdapter(self.aenv_config) as env_adapter:
            tools = await self._load_openai_tools(env_adapter)

            while True:
                request_kwargs = {
                    "messages": messages,
                    # Keep outer rollout grouping driven by trainer group_size while
                    # forcing each OpenAI-style generation request to be single-sample.
                    **self.gconfig.to_openai_args_dict(exclude_args=["n_samples"]),
                    "n": 1,
                }
                if tools:
                    request_kwargs["tools"] = tools
                    request_kwargs["tool_choice"] = self.tool_choice

                completion = await client.chat.completions.create(**request_kwargs)
                model_turn_count += 1
                assistant_message = completion.choices[0].message
                messages.append(assistant_message.model_dump(exclude_none=True))

                tool_calls = assistant_message.tool_calls or []
                if not tool_calls:
                    break

                if tool_round_count >= self.max_turns:
                    logger.warning(
                        "Reached max_turns; stop before executing additional tool calls"
                    )
                    break

                await self._execute_tool_calls(
                    env_adapter=env_adapter,
                    tool_calls=tool_calls,
                    messages=messages,
                )
                tool_round_count += 1

        final_reward = await self._compute_reward(messages=messages, data=data)
        client.set_last_reward(final_reward)
        client.apply_reward_discount(turn_discount=self.aenv_config.turn_discount)

        stats_tracker.get(workflow_context.stat_scope()).scalar(
            reward=final_reward,
            turns=model_turn_count,
            tool_rounds=tool_round_count,
        )
        return client.export_interactions(style=self.export_style)

    async def _load_openai_tools(
        self,
        env_adapter: AenvEnvironmentAdapter,
    ) -> list[dict[str, Any]]:
        try:
            raw_tools = await env_adapter.list_tools()
            return normalize_openai_tools(raw_tools)
        except Exception as exc:
            if self.aenv_config.tool_error_policy == "raise":
                raise
            logger.warning(
                f"Failed to list AEnvironment tools, fallback to no-tool run: {exc}"
            )
            return []

    async def _execute_tool_calls(
        self,
        env_adapter: AenvEnvironmentAdapter,
        tool_calls: list[Any],
        messages: list[dict[str, Any]],
    ) -> None:
        for index, tool_call in enumerate(tool_calls):
            tool_call_id = getattr(tool_call, "id", None) or f"tool_call_{index}"
            function = getattr(tool_call, "function", None)
            tool_name = getattr(function, "name", None)
            if not tool_name:
                if self.aenv_config.tool_error_policy == "raise":
                    raise RuntimeError("Tool call is missing function name")
                messages.append(
                    {
                        "role": "tool",
                        "content": "Error: missing tool function name",
                        "tool_call_id": tool_call_id,
                    }
                )
                continue

            try:
                raw_arguments = getattr(function, "arguments", None)
                logger.debug(
                    "Parsing tool arguments",
                    extra={
                        "tool_name": tool_name,
                        "tool_call_id": tool_call_id,
                        "raw_arguments_type": type(raw_arguments).__name__,
                    },
                )
                tool_args = parse_tool_arguments(raw_arguments)
                logger.debug(
                    "Tool arguments parsed successfully",
                    extra={
                        "tool_name": tool_name,
                        "tool_call_id": tool_call_id,
                        "argument_keys": list(tool_args.keys()) if tool_args else [],
                    },
                )
                result = await env_adapter.call_tool(
                    tool_name=tool_name,
                    arguments=tool_args,
                    timeout=self.aenv_config.tool_call_timeout,
                )
                rendered_content = self._render_tool_content(result.content)
                if result.is_error and self.aenv_config.tool_error_policy == "raise":
                    raise RuntimeError(
                        f"Tool call returned error for '{tool_name}': {rendered_content}"
                    )
                messages.append(
                    {
                        "role": "tool",
                        "content": rendered_content,
                        "tool_call_id": tool_call_id,
                    }
                )
            except ValueError as exc:
                # Argument parsing failed - log detailed context before handling
                logger.warning(
                    "Failed to parse tool arguments",
                    extra={
                        "tool_name": tool_name,
                        "tool_call_id": tool_call_id,
                        "raw_arguments": repr(raw_arguments)[
                            :500
                        ],  # Truncate for safety
                        "error": str(exc),
                    },
                )
                if self.aenv_config.tool_error_policy == "raise":
                    raise RuntimeError(
                        f"Failed to parse arguments for tool '{tool_name}': {exc}"
                    ) from exc
                messages.append(
                    {
                        "role": "tool",
                        "content": f"Error: failed to parse arguments - {exc}",
                        "tool_call_id": tool_call_id,
                    }
                )
            except TypeError as exc:
                # Unsupported argument type - log detailed context before handling
                logger.warning(
                    "Unsupported tool argument type",
                    extra={
                        "tool_name": tool_name,
                        "tool_call_id": tool_call_id,
                        "raw_arguments_type": type(raw_arguments).__name__,
                        "error": str(exc),
                    },
                )
                if self.aenv_config.tool_error_policy == "raise":
                    raise RuntimeError(
                        f"Unsupported argument type for tool '{tool_name}': {exc}"
                    ) from exc
                messages.append(
                    {
                        "role": "tool",
                        "content": f"Error: unsupported argument type - {exc}",
                        "tool_call_id": tool_call_id,
                    }
                )
            except Exception as exc:
                # Catch-all for other exceptions (e.g., tool execution errors)
                logger.warning(
                    "Tool execution failed",
                    extra={
                        "tool_name": tool_name,
                        "tool_call_id": tool_call_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                if self.aenv_config.tool_error_policy == "raise":
                    raise
                messages.append(
                    {
                        "role": "tool",
                        "content": f"Error: {exc}",
                        "tool_call_id": tool_call_id,
                    }
                )

    async def _compute_reward(
        self, messages: list[dict[str, Any]], data: dict[str, Any]
    ) -> float:
        if self._reward_fn_async is None and self._reward_wrapper is None:
            return 0.0

        reward_kwargs = {
            "prompt": data.get("prompt"),
            "completions": self._get_last_assistant_content(messages),
            "prompt_ids": data.get("prompt_ids"),
            "completion_ids": data.get("completion_ids"),
            "answer": data.get("answer"),
            "messages": messages,
            **{
                key: value
                for key, value in data.items()
                if key
                not in {
                    "prompt",
                    "completions",
                    "prompt_ids",
                    "completion_ids",
                    "answer",
                    "messages",
                }
            },
        }

        if self._reward_wrapper is not None:
            reward = await self._reward_wrapper(**reward_kwargs)
            return float(reward)

        assert self._reward_fn_async is not None
        reward = await self._reward_fn_async(**reward_kwargs)
        return float(reward)

    def _build_messages(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        user_input = data.get("messages")
        if isinstance(user_input, list) and user_input:
            for item in user_input:
                messages.append(self._message_to_dict(item))
            return messages

        prompt = data.get("prompt", "")
        messages.append({"role": "user", "content": str(prompt)})
        return messages

    @staticmethod
    def _message_to_dict(message: Any) -> dict[str, Any]:
        if isinstance(message, dict):
            return dict(message)
        if hasattr(message, "model_dump"):
            return message.model_dump(exclude_none=True)
        raise TypeError(f"Unsupported message type: {type(message).__name__}")

    @staticmethod
    def _is_async_callable(fn: Callable[..., Any]) -> bool:
        return inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(
            getattr(fn, "__call__", None)
        )

    @staticmethod
    def _get_last_assistant_content(messages: list[dict[str, Any]]) -> str:
        for message in reversed(messages):
            if message.get("role") == "assistant":
                content = message.get("content", "")
                return "" if content is None else str(content)
        return ""

    @staticmethod
    def _render_tool_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text" and "text" in item:
                        chunks.append(str(item["text"]))
                    else:
                        chunks.append(json.dumps(item, sort_keys=True))
                else:
                    chunks.append(str(item))
            return "\n".join(chunks)
        if isinstance(content, dict):
            return json.dumps(content, sort_keys=True)
        return str(content)
