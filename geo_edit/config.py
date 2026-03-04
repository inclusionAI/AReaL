from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from google.genai import types
from geo_edit.tool_definitions.router import ToolRouter


# =============================================================================
# Config Dataclasses
# =============================================================================

@dataclass(frozen=True, slots=True)
class GoogleAgentConfigs:
    tools: types.Tool
    tool_config: types.ToolConfig
    generate_config: types.GenerateContentConfig
    force_final_generate_config: types.GenerateContentConfig


@dataclass(frozen=True, slots=True)
class APIAgentConfigs:
    """Unified config for OpenAI-compatible APIs (OpenAI, vLLM, SGLang)."""
    tools: Optional[List[Dict[str, Any]]]
    tool_choice: Optional[Union[str, Dict[str, Any]]]
    generate_config: Dict[str, Any]
    force_final_generate_config: Dict[str, Any]
    system_prompt: Optional[str] = None


# =============================================================================
# Config Field Mappings
# =============================================================================

# Google API tool_mode mapping
GOOGLE_TOOL_MODE_MAP = {
    "force": "ANY",
    "auto": "AUTO",
    "direct": None,
    "none": "NONE",
}

# OpenAI API tool_choice mapping
API_TOOL_CHOICE_MAP = {
    "force": "required",
    "auto": "auto",
    "direct": "none",
    "none": "none",
}

# Fields to copy from Google base config
GOOGLE_CONFIG_FIELDS = [
    "temperature",
    "max_output_tokens",
    "candidate_count",
    "automatic_function_calling",
    "tools",
]

# Fields to remove when disabling reasoning (API)
API_REASONING_FIELDS = ["_reasoning_level", "reasoning"]

# System prompt field mapping by API type
# Note: For chat_completions, '_system_prompt' is extracted and injected as system message
SYSTEM_PROMPT_FIELD_MAP = {
    "google": "system_instruction",
    "responses": "instructions",
    "chat_completions": "_system_prompt",
}


# =============================================================================
# Google Config Builders
# =============================================================================

def build_google_agent_configs(
    tool_router: "ToolRouter",
    *,
    max_output_tokens: Optional[int] = None,
    thinking_level: Optional[str] = "high",
    include_thoughts: Optional[bool] = True,
    temperature: float = 1.0,
    system_prompt: Optional[str | None] = None,
) -> GoogleAgentConfigs:
    """Build Google Gemini agent configs using ToolRouter."""
    tool_mode = tool_router.tool_mode
    tool_declarations = tool_router.get_available_declarations()
    tool_names = list(tool_router.get_available_tools().keys())

    # Build tools
    if tool_mode == "direct":
        tools = None
        tool_config = None
    else:
        tools = types.Tool(function_declarations=tool_declarations) if tool_declarations else None
        mode = GOOGLE_TOOL_MODE_MAP.get(tool_mode, "AUTO")
        fc_config = types.FunctionCallingConfig(mode=mode, allowed_function_names=tool_names if tool_mode == "force" else None)
        tool_config = types.ToolConfig(function_calling_config=fc_config)

    # Build thinking config
    thinking_config = None
    if thinking_level is not None:
        thinking_config = types.ThinkingConfig(
            thinkingLevel=thinking_level,
            includeThoughts=True if include_thoughts is None else include_thoughts,
        )

    # Build generate config
    generate_kwargs = {
        "thinking_config": thinking_config,
        "temperature": temperature,
        "system_instruction": system_prompt,
        "max_output_tokens": max_output_tokens,
        "candidate_count": 1,
        "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
    }
    if tools is not None:
        generate_kwargs["tools"] = [tools]
        generate_kwargs["tool_config"] = tool_config

    generate_config = types.GenerateContentConfig(**generate_kwargs)

    # Build force_final config
    force_final_kwargs = {
        "thinking_config": thinking_config,
        "tool_config": types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="NONE")),
        "temperature": temperature,
        "system_instruction": system_prompt,
        "max_output_tokens": max_output_tokens,
        "candidate_count": 1,
    }
    if tools is not None:
        force_final_kwargs["tools"] = [tools]

    force_final_generate_config = types.GenerateContentConfig(**force_final_kwargs)

    return GoogleAgentConfigs(
        tools=tools,
        tool_config=tool_config,
        generate_config=generate_config,
        force_final_generate_config=force_final_generate_config,
    )


def derive_google_config(
    base_config: types.GenerateContentConfig,
    *,
    system_prompt: Optional[str] = None,
    tool_mode: Optional[str] = None,
) -> types.GenerateContentConfig:
    """Derive a new Google config by copying base_config and modifying specified fields.

    Args:
        base_config: Base config to copy from.
        system_prompt: Override system prompt (None = keep original).
        tool_mode: "NONE"/"ANY"/"AUTO" to override, None = keep original.
    """
    # Copy all fields from base_config
    kwargs = base_config.model_dump(exclude_none=True)

    # Apply modifications
    if system_prompt is not None:
        kwargs[SYSTEM_PROMPT_FIELD_MAP["google"]] = system_prompt

    if tool_mode is not None:
        kwargs["tool_config"] = types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode=tool_mode)  # type: ignore
        )

    return types.GenerateContentConfig(**kwargs)


# =============================================================================
# API Config Builders
# =============================================================================

def _build_tools_list(tool_router: "ToolRouter", api_mode: str) -> List[Dict[str, Any]]:
    """Build tools list in format based on API mode."""
    tool_declarations = tool_router.get_available_declarations()
    tools = []

    for tool in tool_declarations:
        if api_mode == "responses":
            tools.append({
                "type": "function",
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            })
        elif api_mode == "chat_completions":
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
    tool_router: "ToolRouter",
    *,
    api_mode: str = "responses",
    max_output_tokens: Optional[int] = None,
    temperature: float = 1.0,
    reasoning_level: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> APIAgentConfigs:
    """Build unified config for OpenAI-compatible APIs."""
    if api_mode not in ("responses", "chat_completions"):
        raise ValueError(f"Invalid api_mode: {api_mode}")

    tool_mode = tool_router.tool_mode
    tools = _build_tools_list(tool_router, api_mode)
    tool_choice = API_TOOL_CHOICE_MAP.get(tool_mode, "auto")

    if tool_mode == "direct":
        tool_choice = "none" if api_mode == "chat_completions" else None
        tools = None

    # Field name differs by API mode
    max_tokens_key = "max_output_tokens" if api_mode == "responses" else "max_tokens"

    # Build base config
    generate_config: Dict[str, Any] = {"temperature": temperature}
    if max_output_tokens is not None:
        generate_config[max_tokens_key] = max_output_tokens
    if tools is not None:
        generate_config["tools"] = tools
        generate_config["tool_choice"] = tool_choice
        generate_config["parallel_tool_calls"] = True

    if api_mode == "responses":
        if reasoning_level is not None:
            generate_config["reasoning"] = {"effort": reasoning_level}
        if system_prompt is not None:
            generate_config["instructions"] = system_prompt
    elif api_mode == "chat_completions":
        if system_prompt is not None:
            generate_config["_system_prompt"] = system_prompt
        if reasoning_level is not None:
            generate_config["_reasoning_level"] = reasoning_level

    # Build force_final config
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


def derive_api_config(
    base_config: Dict[str, Any],
    *,
    api_mode: str,
    system_prompt: Optional[str] = None,
    tool_choice: Optional[str] = None,
) -> Dict[str, Any]:
    """Derive a new API config by copying base_config and modifying specified fields.

    Args:
        base_config: Base config to copy from.
        api_mode: "responses" or "chat_completions".
        system_prompt: Override system prompt (None = keep original).
        tool_choice: Override tool_choice (None = keep original).
    """
    # Copy all fields from base_config
    config = dict(base_config)

    # Apply modifications
    if system_prompt is not None:
        config[SYSTEM_PROMPT_FIELD_MAP[api_mode]] = system_prompt

    if tool_choice is not None:
        config["tool_choice"] = tool_choice

    return config


__all__ = [
    "GoogleAgentConfigs",
    "APIAgentConfigs",
    "GOOGLE_TOOL_MODE_MAP",
    "API_TOOL_CHOICE_MAP",
    "build_google_agent_configs",
    "build_api_agent_configs",
    "derive_google_config",
    "derive_api_config",
]
