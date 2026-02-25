"""Unified prompt management for geo_edit.

This module centralizes all prompt definitions:
- System prompts for main agents (API, vLLM, etc.)
- Tool agent prompts
- Evaluation prompts
"""

from geo_edit.prompts.system_prompts import (
    API_NO_TOOL_SYSTEM_PROMPT,
    VLLM_NO_TOOL_SYSTEM_PROMPT,
    VLLM_FORCE_TOOL_CALL_PROMPT,
    TOOL_EXECUTION_SUCCESS_PROMPT,
    TOOL_EXECUTION_FAILURE_PROMPT,
    get_system_prompt,
)
from geo_edit.prompts.eval_prompts import (
    EVAL_SYSTEM_PROMPT,
    EVAL_QUERY_PROMPT,
)
from geo_edit.prompts.tool_agent_prompts import (
    DEFAULT_TOOL_AGENT_PROMPT,
    TOOL_AGENT_PROMPTS,
    get_tool_agent_prompt,
    register_tool_agent_prompt,
    list_tool_agents,
)

__all__ = [
    # System prompts
    "API_NO_TOOL_SYSTEM_PROMPT",
    "VLLM_NO_TOOL_SYSTEM_PROMPT",
    "VLLM_FORCE_TOOL_CALL_PROMPT",
    "TOOL_EXECUTION_SUCCESS_PROMPT",
    "TOOL_EXECUTION_FAILURE_PROMPT",
    "get_system_prompt",
    # Eval prompts
    "EVAL_SYSTEM_PROMPT",
    "EVAL_QUERY_PROMPT",
    # Tool agent prompts
    "DEFAULT_TOOL_AGENT_PROMPT",
    "TOOL_AGENT_PROMPTS",
    "get_tool_agent_prompt",
    "register_tool_agent_prompt",
    "list_tool_agents",
]
