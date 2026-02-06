from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from google.genai import types
from geo_edit.constants import (
    MAX_TOOL_CALLS,
)
from geo_edit.environment.action import TOOL_FUNCTIONS_DECLARE, TOOL_FUNCTIONS

@dataclass(frozen=True, slots=True)
class GoogleAgentConfigs:
    tools: types.Tool
    tool_config: types.ToolConfig
    generate_config: types.GenerateContentConfig
    force_final_generate_config: types.GenerateContentConfig

@dataclass(frozen=True, slots=True)
class OpenAIAgentConfigs:
    tools: List[Dict[str, Any]]
    tool_choice: Union[str, Dict[str, Any]]
    generate_config: Dict[str, Any]
    force_final_generate_config: Dict[str, Any]


@dataclass(frozen=True, slots=True)
class VLLMAgentConfigs:
    tools: List[Dict[str, Any]]
    tool_choice: Union[str, Dict[str, Any]]
    generate_config: Dict[str, Any]
    force_final_generate_config: Dict[str, Any]
    system_prompt: Optional[str] = None


@dataclass(frozen=True, slots=True)
class SGLangAgentConfigs:
    tools: List[Dict[str, Any]]
    tool_choice: Union[str, Dict[str, Any]]
    generate_config: Dict[str, Any]
    force_final_generate_config: Dict[str, Any]

def build_google_agent_configs(
    *,
    max_output_tokens: Optional[int] = None,
    thinking_level: Optional[str] = "high",
    include_thoughts: Optional[bool] = True,
    temperature: float = 1.0,
    system_prompt: Optional[str|None] = None,
    tool_mode:  Optional[str]  = None,
) -> GoogleAgentConfigs:
    
    tools = types.Tool(function_declarations=TOOL_FUNCTIONS_DECLARE)

    if tool_mode=="force":
        tool_config= types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode="ANY",
                allowed_function_names=TOOL_FUNCTIONS.keys()), 
        )
    elif tool_mode=="auto":
        tool_config = types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode=tool_mode)
        )
    elif tool_mode=="direct":
        tool_config=None
        tools=None
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
        thinking_config = thinking_config,
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode="NONE")
        ),
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

def build_openai_agent_configs(
    *,
    max_output_tokens: Optional[int] = None,
    temperature: float = 1.0,
    tool_mode: Optional[str] = None,
    reasoning_level: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> OpenAIAgentConfigs:
    
    tools =[]
    for tool in TOOL_FUNCTIONS_DECLARE:
        tools.append(
            {   
                "type": "function",
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            }
        )
    
    if tool_mode =="direct":
        tool_mode=None
        tools=None
    elif tool_mode =="force":
        tool_mode="required"
    elif tool_mode =="auto":
        tool_mode="auto"
    else:
        raise ValueError(f"Invalid tool_mode: {tool_mode}")
    
    generate_config = {
        "tools": tools,
        "tool_choice": tool_mode,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "reasoning": {"effort":reasoning_level}
    }
    if system_prompt is not None:
        generate_config["instructions"] = system_prompt
    
    force_final_generate_config = {
        "tools": tools,
        "tool_choice": "none",
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "reasoning": {"effort":reasoning_level},
    }
    if system_prompt is not None:
        force_final_generate_config["instructions"] = system_prompt

    return OpenAIAgentConfigs(
        tools=tools,
        tool_choice=tool_mode,
        generate_config=generate_config,
        force_final_generate_config=force_final_generate_config,
    )


def build_vllm_agent_configs(
    *,
    max_output_tokens: Optional[int] = None,
    temperature: float = 1.0,
    tool_mode: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> VLLMAgentConfigs:
    tools = []
    for tool in TOOL_FUNCTIONS_DECLARE:
        tools.append({
            "type": "function", "name": tool["name"],
            "description": tool["description"], "parameters": tool["parameters"],
        })

    if tool_mode == "direct":
        tool_mode = None
        tools = None
    elif tool_mode == "force":
        tool_mode = "required"
    elif tool_mode == "auto":
        tool_mode = "auto"
    else:
        raise ValueError(f"Invalid tool_mode: {tool_mode}")

    generate_config = {
        "tools": tools, "tool_choice": tool_mode,
        "temperature": temperature, "max_output_tokens": max_output_tokens,
    }

    force_final_generate_config = {
        "tools": tools, "tool_choice": "none",
        "temperature": temperature, "max_output_tokens": max_output_tokens,
    }
    return VLLMAgentConfigs(
        tools=tools,
        tool_choice=tool_mode,
        generate_config=generate_config,
        force_final_generate_config=force_final_generate_config,
        system_prompt=system_prompt,
    )


def build_sglang_agent_configs(
    *,
    max_output_tokens: Optional[int] = None,
    temperature: float = 1.0,
    tool_mode: Optional[str] = None,
) -> SGLangAgentConfigs:
    tools = []
    for tool in TOOL_FUNCTIONS_DECLARE:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                },
            }
        )

    if tool_mode == "direct":
        tool_mode = None
        tools = None
    elif tool_mode == "force":
        tool_mode = "required"
    elif tool_mode == "auto":
        tool_mode = "auto"
    else:
        raise ValueError(f"Invalid tool_mode: {tool_mode}")

    generate_config = {
        "tools": tools,
        "tool_choice": tool_mode,
        "temperature": temperature,
        "max_tokens": max_output_tokens,
    }

    force_final_generate_config = {
        "tools": tools,
        "tool_choice": "none",
        "temperature": temperature,
        "max_tokens": max_output_tokens,
    }
    return SGLangAgentConfigs(
        tools=tools,
        tool_choice=tool_mode,
        generate_config=generate_config,
        force_final_generate_config=force_final_generate_config,
    )

__all__ = [
    "API_KEY",
    "MAX_TOOL_CALLS",
    "SYSTEM_PROMPT",
    "AgentConfigs",
    "OpenAIAgentConfigs",
    "VLLMAgentConfigs",
    "SGLangAgentConfigs",
    "build_agent_configs",
    "build_openai_agent_configs",
    "build_vllm_agent_configs",
    "build_sglang_agent_configs",
]
