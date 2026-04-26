"""Tau2 agent for the experimental agent service."""

from __future__ import annotations

import inspect
import json
import os
from typing import Any

from areal.experimental.agent_service.types import (
    AgentRequest,
    AgentResponse,
    EventEmitter,
)
from areal.utils import logging

logger = logging.getLogger("Tau2Agent")


def _make_pydantic_tool(tau2_tool: Any):
    fn = tau2_tool._func  # noqa: SLF001
    name = tau2_tool.name
    schema = getattr(tau2_tool, "openai_schema", {}) or {}
    doc = schema.get("function", {}).get("description", name)

    async def _wrapper(**kwargs: Any) -> str:
        try:
            result = fn(**kwargs)
        except Exception as exc:
            result = f"Tool error: {exc}"
        if not isinstance(result, str):
            result = json.dumps(result, default=str)
        return result

    _wrapper.__name__ = name
    _wrapper.__qualname__ = name
    _wrapper.__doc__ = doc
    sig = inspect.signature(fn)
    _wrapper.__signature__ = inspect.Signature(
        [
            inspect.Parameter(
                pname,
                kind=inspect.Parameter.KEYWORD_ONLY,
                default=param.default,
                annotation=param.annotation,
            )
            for pname, param in sig.parameters.items()
        ]
    )
    if hasattr(fn, "__annotations__"):
        _wrapper.__annotations__ = {
            k: v for k, v in fn.__annotations__.items() if k != "return"
        }
    return _wrapper


def _think_tool_fn(thoughts: str) -> str:
    del thoughts
    return "Your thoughts are recorded. Please continue your work."


class Tau2Agent:
    """AgentRunnable that wraps a PydanticAI agent with tau2 tools."""

    def __init__(self, config: dict[str, Any] | None = None, **kwargs: Any) -> None:
        del kwargs
        try:
            from pydantic_ai import Agent
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.providers.openai import OpenAIProvider
            from tau2.environment.tool import Tool as Tau2Tool
            from tau2.registry import registry
        except ImportError as exc:
            raise ImportError(
                "Tau2 agent service example requires 'pydantic-ai' and 'tau2-bench'"
            ) from exc

        config = config or {}
        tau2_cfg = config.get("tau2", {})
        agent_llm_cfg = config.get("agent_llm", {})

        self._domain = tau2_cfg.get("domain") or os.environ.get(
            "TAU2_DOMAIN", "airline"
        )
        add_thinking = tau2_cfg.get("add_thinking_tool", False)

        data_dir = tau2_cfg.get("data_dir") or os.environ.get("TAU2_DATA_DIR")
        if data_dir:
            os.environ["TAU2_DATA_DIR"] = data_dir

        env_constructor = registry.get_env_constructor(self._domain)
        env = env_constructor(solo_mode=False)
        tau2_tools: list[Tau2Tool] = env.get_tools()
        if add_thinking:
            tau2_tools.append(Tau2Tool(_think_tool_fn))

        tools = [_make_pydantic_tool(t) for t in tau2_tools]
        system_prompt = env.get_policy()

        model_name = agent_llm_cfg.get("model", "openai:default")
        base_url = agent_llm_cfg.get("base_url")
        api_key = agent_llm_cfg.get("api_key", "unused")

        if base_url:
            model: Any = OpenAIChatModel(
                model_name.replace("openai:", ""),
                provider=OpenAIProvider(base_url=base_url, api_key=api_key),
            )
        else:
            model = model_name

        self._openai_chat_model = OpenAIChatModel
        self._openai_provider = OpenAIProvider
        self._agent = Agent(model, system_prompt=system_prompt, tools=tools)
        logger.info(
            "Tau2Agent initialized (domain=%s, tools=%d, model=%s)",
            self._domain,
            len(tools),
            model_name,
        )

    def _resolve_model(self, metadata: dict[str, Any]) -> Any:
        base_url = metadata.get("inference_base_url")
        if not base_url:
            return self._agent.model
        model_name = metadata.get("inference_model", "default")
        api_key = metadata.get("inference_api_key", "unused")
        return self._openai_chat_model(
            model_name,
            provider=self._openai_provider(base_url=base_url, api_key=api_key),
        )

    async def run(
        self,
        request: AgentRequest,
        *,
        emitter: EventEmitter,
    ) -> AgentResponse:
        from pydantic_ai.messages import (
            ModelRequest,
            TextPart,
            ToolCallPart,
            ToolReturnPart,
            UserPromptPart,
        )
        from pydantic_ai.messages import ModelResponse as PAModelResponse

        message_history: list[ModelRequest | PAModelResponse] = []
        for msg in request.history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                message_history.append(
                    ModelRequest(parts=[UserPromptPart(content=content or "")])
                )
            elif role == "assistant":
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    parts = []
                    for tc in tool_calls:
                        fn = tc.get("function", tc)
                        parts.append(
                            ToolCallPart(
                                tool_name=fn.get("name", ""),
                                args=fn.get("arguments", ""),
                                tool_call_id=tc.get("id", ""),
                            )
                        )
                    message_history.append(PAModelResponse(parts=parts))
                elif content:
                    message_history.append(
                        PAModelResponse(parts=[TextPart(content=content)])
                    )
            elif role == "tool":
                tool_call_id = msg.get("tool_call_id", "")
                message_history.append(
                    ModelRequest(
                        parts=[
                            ToolReturnPart(
                                tool_name=tool_call_id,
                                content=content or "",
                                tool_call_id=tool_call_id,
                            )
                        ]
                    )
                )

        try:
            result = await self._agent.run(
                request.message,
                message_history=message_history,
                model=self._resolve_model(request.metadata),
            )
        except Exception as exc:
            logger.error("Tau2Agent turn failed: %s", exc)
            await emitter.emit_delta(f"Agent error: {exc}")
            return AgentResponse(
                summary=f"Agent error: {exc}", metadata={"tool_calls": []}
            )

        final_text = str(result.output) if result.output else ""
        tool_calls: list[dict[str, Any]] = []
        for msg in result.new_messages():
            if not hasattr(msg, "parts"):
                continue
            for part in msg.parts:
                kind = getattr(part, "part_kind", "")
                if kind == "tool-call":
                    name = getattr(part, "tool_name", "")
                    args = getattr(part, "args", "")
                    if isinstance(args, dict):
                        args = json.dumps(args)
                    await emitter.emit_tool_call(name=name, args=str(args))
                    tool_calls.append({"name": name, "arguments": args})
                elif kind == "tool-return":
                    name = getattr(part, "tool_name", "")
                    content = str(getattr(part, "content", ""))
                    await emitter.emit_tool_result(name=name, result=content)

        if final_text:
            await emitter.emit_delta(final_text)

        return AgentResponse(
            summary=final_text[:200], metadata={"tool_calls": tool_calls}
        )
