from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from google.genai import types
from .constants import (
    MAX_TOOL_CALLS,
    MATHVISION_INPUT_TEMPLATE,
    NOTOOL_INPUT_TEMPLATE,
)

from .environment.action import TOOL_FUNCTIONS_DECLARE, TOOL_FUNCTIONS

@dataclass(frozen=True, slots=True)
class AgentConfigs:
    tools: types.Tool
    tool_config: types.ToolConfig
    generate_config: types.GenerateContentConfig
    direct_generate_config: types.GenerateContentConfig
    force_generate_config: types.GenerateContentConfig
    force_tool_call_config: Optional[types.GenerateContentConfig] = None


@dataclass(frozen=True, slots=True)
class OpenAIAgentConfigs:
    tools: List[Dict[str, Any]]
    tool_choice: Union[str, Dict[str, Any]]
    generate_config: Dict[str, Any]
    direct_generate_config: Dict[str, Any]
    force_generate_config: Dict[str, Any]
    force_tool_call_config: Dict[str, Any]


@dataclass(frozen=True, slots=True)
class VLLMAgentConfigs:
    tools: List[Dict[str, Any]]
    tool_choice: Union[str, Dict[str, Any]]
    generate_config: Dict[str, Any]
    direct_generate_config: Dict[str, Any]
    force_generate_config: Dict[str, Any]
    system_prompt: Optional[str] = None

def _build_generate_config(**kwargs: object) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(**{k: v for k, v in kwargs.items() if v is not None})


def _build_openai_generate_config(**kwargs: object) -> Dict[str, Any]:
    return {k: v for k, v in kwargs.items() if v is not None}


def _build_openai_tool_specs(tool_declarations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tool_specs = []
    for tool in tool_declarations:
        tool_specs.append(
            {
                "type": "function",
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            }
        )
    return tool_specs


def _build_chat_tool_specs(
    tool_declarations: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    tool_specs = []
    for tool in tool_declarations:
        tool_specs.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                },
            }
        )
    return tool_specs


def _resolve_openai_tool_choice(tool_mode: str) -> str:
    mode = tool_mode.upper()
    if mode in {"ANY", "REQUIRED"}:
        return "required"
    if mode in {"NONE", "OFF"}:
        return "none"
    return "auto"


def _resolve_vllm_tool_choice(tool_mode: str) -> str:
    mode = tool_mode.upper()
    if mode in {"NONE", "OFF"}:
        return "none"
    return "auto"

def build_agent_configs(
    *,
    max_output_tokens: Optional[int] = None,
    thinking_level: Optional[str] = "high",
    include_thoughts: Optional[bool] = True,
    temperature: float = 1.0,
    system_prompt: Optional[str] = None,
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
    thinking_config = None
    if thinking_level is not None:
        thinking_config = types.ThinkingConfig(
            thinkingLevel=thinking_level,
            include_thoughts=True if include_thoughts is None else include_thoughts,
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
        system_instruction=[system_prompt] if system_prompt is not None else None,
        max_output_tokens=max_output_tokens,
        candidate_count=candidate_count,
        automatic_function_calling=automatic_function_calling,
    )
    force_tool_call_config = _build_generate_config(
        tools=[tools],
        thinking_config=thinking_config,
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode="ANY")
        ),
        temperature=temperature,
        system_instruction=[system_prompt] if system_prompt is not None else None,
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
        system_instruction=[system_prompt] if system_prompt is not None else None,
        max_output_tokens=max_output_tokens,
        candidate_count=candidate_count,
    )

    return AgentConfigs(
        tools=tools,
        tool_config=tool_config,
        generate_config=generate_config,
        direct_generate_config=direct_generate_config,
        force_tool_call_config=force_tool_call_config,
        force_generate_config=force_generate_config,
    )


def build_openai_agent_configs(
    *,
    max_output_tokens: Optional[int] = None,
    temperature: float = 1.0,
    system_prompt: Optional[str] = None,
    tool_mode: str = "AUTO",
    reasoning_level: Optional[str] = None,
) -> OpenAIAgentConfigs:
    tools = _build_openai_tool_specs(TOOL_FUNCTIONS_DECLARE)
    tool_choice = _resolve_openai_tool_choice(tool_mode)
    
    force_tool_call_config=_build_openai_generate_config(
        tools=tools,
        tool_choice="required",
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        instructions=system_prompt if system_prompt is not None else None,
        reasoning={"effort":reasoning_level}
    )

    generate_config = _build_openai_generate_config(
        tools=tools,
        tool_choice=tool_choice,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        instructions=system_prompt if system_prompt is not None else None,
        reasoning={"effort":reasoning_level}
    )
    direct_generate_config = _build_openai_generate_config(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        instructions=system_prompt if system_prompt is not None else None,
        reasoning={"effort":reasoning_level},
    )
    force_generate_config = _build_openai_generate_config(
        tools=tools,
        tool_choice="none",
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        instructions=system_prompt if system_prompt is not None else None,
        reasoning={"effort":reasoning_level},
    )

    return OpenAIAgentConfigs(
        tools=tools,
        tool_choice=tool_choice,
        generate_config=generate_config,
        direct_generate_config=direct_generate_config,
        force_generate_config=force_generate_config,
        force_tool_call_config=force_tool_call_config,
    )


def build_vllm_agent_configs(
    *,
    max_output_tokens: Optional[int] = None,
    temperature: float = 1.0,
    tool_mode: str = "AUTO",
    system_prompt: Optional[str] = None,
) -> VLLMAgentConfigs:
    tools = _build_chat_tool_specs(TOOL_FUNCTIONS_DECLARE)
    tool_choice = _resolve_vllm_tool_choice(tool_mode)

    generate_config = _build_openai_generate_config(
        tools=tools,
        tool_choice=tool_choice,
        temperature=temperature,
        max_tokens=max_output_tokens,
        instructions=system_prompt if system_prompt is not None else None,
    )
    direct_generate_config = _build_openai_generate_config(
        temperature=temperature,
        max_tokens=max_output_tokens,
    )
    force_generate_config = _build_openai_generate_config(
        tools=tools,
        tool_choice="none",
        temperature=temperature,
        max_tokens=max_output_tokens,
    )
    return VLLMAgentConfigs(
        tools=tools,
        tool_choice=tool_choice,
        generate_config=generate_config,
        direct_generate_config=direct_generate_config,
        force_generate_config=force_generate_config,
        system_prompt=system_prompt,
    )

__all__ = [
    "API_KEY",
    "MAX_TOOL_CALLS",
    "MATHVISION_INPUT_TEMPLATE",
    "NOTOOL_INPUT_TEMPLATE",
    "SYSTEM_PROMPT",
    "AgentConfigs",
    "OpenAIAgentConfigs",
    "VLLMAgentConfigs",
    "build_agent_configs",
    "build_openai_agent_configs",
    "build_vllm_agent_configs",
]
