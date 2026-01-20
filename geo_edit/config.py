from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from google.genai import types
from .constants import (
    MAX_TOOL_CALLS,
    MATHVISION_INPUT_TEMPLATE,
    NOTOOL_INPUT_TEMPLATE,
    SYSTEM_PROMPT,
)

from .environment.action import TOOL_FUNCTIONS_DECLARE, TOOL_FUNCTIONS

@dataclass(frozen=True, slots=True)
class AgentConfigs:
    tools: types.Tool
    tool_config: types.ToolConfig
    generate_config: types.GenerateContentConfig
    direct_generate_config: types.GenerateContentConfig
    force_tool_config: types.ToolConfig
    force_generate_config: types.GenerateContentConfig

def _build_generate_config(**kwargs: object) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(**{k: v for k, v in kwargs.items() if v is not None})

def build_agent_configs(
    *,
    max_output_tokens: Optional[int] = None,
    thinking_level: str = "high",
    include_thoughts: bool = True,
    temperature: float = 1.0,
    system_prompt: str = SYSTEM_PROMPT,
    candidate_count: int = 1,
    tool_mode: str = "AUTO",
    disable_automatic_function_calling: bool = True,
) -> AgentConfigs:
    tools = types.Tool(function_declarations=TOOL_FUNCTIONS_DECLARE)
    if tool_mode=="ANY":
        tool_config= types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode="ANY",
                allowed_function_names=TOOL_FUNCTIONS.keys()),
            
        )
    else:
        tool_config = types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode=tool_mode)
        )
    thinking_config = types.ThinkingConfig(
        thinkingLevel=thinking_level, include_thoughts=include_thoughts
    )
    automatic_function_calling = (
        types.AutomaticFunctionCallingConfig(disable=True)
        if disable_automatic_function_calling
        else None
    )

    generate_config = _build_generate_config(
        tools=[tools],
        thinking_config=thinking_config,
        tool_config=tool_config,
        temperature=temperature,
        system_instruction=[system_prompt],
        max_output_tokens=max_output_tokens,
        candidate_count=candidate_count,
        automatic_function_calling=automatic_function_calling,
    )
    direct_generate_config = _build_generate_config(
        thinking_config=thinking_config,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        candidate_count=candidate_count,
        automatic_function_calling=automatic_function_calling,
    )

    force_generate_tool_config = types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(mode="NONE")
    )
    force_generate_config = _build_generate_config(
        tools=[tools],
        thinking_config=thinking_config,
        tool_config=force_generate_tool_config,
        temperature=temperature,
        system_instruction=[system_prompt],
        max_output_tokens=max_output_tokens,
        candidate_count=candidate_count,
    )

    return AgentConfigs(
        tools=tools,
        tool_config=tool_config,
        generate_config=generate_config,
        direct_generate_config=direct_generate_config,
        force_tool_config=force_generate_tool_config,
        force_generate_config=force_generate_config,
    )

__all__ = [
    "API_KEY",
    "MAX_TOOL_CALLS",
    "MATHVISION_INPUT_TEMPLATE",
    "NOTOOL_INPUT_TEMPLATE",
    "SYSTEM_PROMPT",
    "AgentConfigs",
    "build_agent_configs",
]
