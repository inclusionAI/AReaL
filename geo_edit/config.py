from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from google.genai import types
from geo_edit.environment.action import get_tool_declarations, get_tool_functions


@dataclass(frozen=True, slots=True)
class GoogleAgentConfigs:
    tools: types.Tool
    tool_config: types.ToolConfig
    generate_config: types.GenerateContentConfig
    force_final_generate_config: types.GenerateContentConfig


@dataclass(frozen=True, slots=True)
class APIAgentConfigs:
    """Unified config for OpenAI-compatible APIs (OpenAI, vLLM, SGLang).

    Works with both responses API and chat_completions API.
    """
    tools: Optional[List[Dict[str, Any]]]
    tool_choice: Optional[Union[str, Dict[str, Any]]]
    generate_config: Dict[str, Any]
    force_final_generate_config: Dict[str, Any]
    system_prompt: Optional[str] = None


def build_google_agent_configs(
    *,
    max_output_tokens: Optional[int] = None,
    thinking_level: Optional[str] = "high",
    include_thoughts: Optional[bool] = True,
    temperature: float = 1.0,
    system_prompt: Optional[str | None] = None,
    tool_mode: Optional[str] = None,
) -> GoogleAgentConfigs:
    tool_declarations = get_tool_declarations()
    tool_functions = get_tool_functions()
    tools = types.Tool(function_declarations=tool_declarations)

    if tool_mode == "force":
        tool_config = types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode="ANY", allowed_function_names=tool_functions.keys()),
        )
    elif tool_mode == "auto":
        tool_config = types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode=tool_mode))
    elif tool_mode == "direct":
        tool_config = None
        tools = None
    else:
        raise ValueError(f"Invalid tool_mode: {tool_mode}")

    thinking_config = None
    if thinking_level is not None:
        thinking_config = types.ThinkingConfig(
            thinkingLevel=thinking_level,
            include_thoughts=True if include_thoughts is None else include_thoughts,
        )

    generate_kwargs = dict(
        thinking_config=thinking_config,
        temperature=temperature,
        system_instruction=system_prompt,
        max_output_tokens=max_output_tokens,
        candidate_count=1,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    if tools is not None:
        generate_kwargs["tools"] = [tools]
        generate_kwargs["tool_config"] = tool_config

    generate_config = types.GenerateContentConfig(**generate_kwargs)

    force_final_kwargs = dict(
        thinking_config=thinking_config,
        tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="NONE")),
        temperature=temperature,
        system_instruction=system_prompt,
        max_output_tokens=max_output_tokens,
        candidate_count=1,
    )
    if tools is not None:
        force_final_kwargs["tools"] = [tools]

    force_final_generate_config = types.GenerateContentConfig(**force_final_kwargs)

    return GoogleAgentConfigs(
        tools=tools,
        tool_config=tool_config,
        generate_config=generate_config,
        force_final_generate_config=force_final_generate_config,
    )


def _build_tools_list(api_mode: str) -> List[Dict[str, Any]]:
    """Build tools list in the correct format based on API mode.

    - responses API: flat format {"type": "function", "name": ..., "parameters": ...}
    - chat_completions API: nested format {"type": "function", "function": {"name": ..., "parameters": ...}}
    """
    tool_declarations = get_tool_declarations()
    tools = []

    for tool in tool_declarations:
        if api_mode == "responses":
            # Responses API: flat format
            tools.append({
                "type": "function",
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            })
        elif api_mode == "chat_completions":
            # Chat Completions API: nested format
            tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                },
            })

    return tools


def build_api_agent_configs(
    *,
    api_mode: str = "responses",
    max_output_tokens: Optional[int] = None,
    temperature: float = 1.0,
    tool_mode: Optional[str] = None,
    reasoning_level: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> APIAgentConfigs:
    """Build unified config for OpenAI-compatible APIs."""
    if api_mode not in ("responses", "chat_completions"):
        raise ValueError(f"Invalid api_mode: {api_mode}. Must be 'responses' or 'chat_completions'.")

    # Build tools and normalize tool_choice
    tools = _build_tools_list(api_mode)
    if tool_mode == "direct":
        tool_choice = "none" if api_mode == "chat_completions" else None
        tools = None
    elif tool_mode == "force":
        tool_choice = "required"
    elif tool_mode == "auto":
        tool_choice = "auto"
    else:
        raise ValueError(f"Invalid tool_mode: {tool_mode}")

    # Field name differs by API mode
    max_tokens_key = "max_output_tokens" if api_mode == "responses" else "max_tokens"

    # Build base config
    generate_config: Dict[str, Any] = {"temperature": temperature}
    if max_output_tokens is not None:
        generate_config[max_tokens_key] = max_output_tokens
    if tools is not None:
        generate_config["tools"] = tools
        generate_config["tool_choice"] = tool_choice
    if api_mode == "responses":
        if reasoning_level is not None:
            generate_config["reasoning"] = {"effort": reasoning_level}
        if system_prompt is not None:
            generate_config["instructions"] = system_prompt

    # Build force_final config (copy and override tool_choice)
    force_final_generate_config = generate_config.copy()
    if tools is not None:
        force_final_generate_config["tool_choice"] = "none"

    return APIAgentConfigs(
        tools=tools,
        tool_choice=tool_choice,
        generate_config=generate_config,
        force_final_generate_config=force_final_generate_config,
        system_prompt=system_prompt,
    )


__all__ = [
    "GoogleAgentConfigs",
    "APIAgentConfigs",
    "build_google_agent_configs",
    "build_api_agent_configs",
]
