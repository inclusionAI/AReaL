"""Tool Agent prompts - centralized prompt definitions for all tool agents."""

from __future__ import annotations

from typing import Dict, Optional

# Default prompt for tool agents without specialized prompts
DEFAULT_TOOL_AGENT_PROMPT = (
    "You are a tool agent. Analyze the image and answer the question. "
    "Return JSON with at least one field in {analysis, text, result, error}."
)

# Agent-specific system prompts
# Maps tool agent name to its specialized prompt
TOOL_AGENT_PROMPTS: Dict[str, str] = {
    "multimath": (
        "You are a math-vision analysis agent. "
        "Analyze the mathematical content in the image and solve the problem step by step. "
        "Return JSON with at least one field in {analysis, text, result, error}."
    ),
    "gllava": (
        "You are a visual language model analysis agent. "
        "Analyze the image carefully and provide a concise, accurate response. "
        "Return JSON with at least one field in {analysis, text, result, error}."
    ),
    "chartmoe": (
        "You are a chart analysis agent specialized in interpreting plots and graphs. "
        "Extract data, identify trends, and provide insights from the visualization. "
        "Return JSON with at least one field in {analysis, text, result, error}."
    ),
}


def get_tool_agent_prompt(tool_name: str) -> str:
    """Get the system prompt for a specific tool agent.

    Args:
        tool_name: Name of the tool agent.

    Returns:
        System prompt string. Returns DEFAULT_TOOL_AGENT_PROMPT if not found.
    """
    return TOOL_AGENT_PROMPTS.get(tool_name, DEFAULT_TOOL_AGENT_PROMPT)


def register_tool_agent_prompt(tool_name: str, prompt: str) -> None:
    """Register a custom system prompt for a tool agent.

    Args:
        tool_name: Name of the tool agent.
        prompt: System prompt to use.
    """
    TOOL_AGENT_PROMPTS[tool_name] = prompt


def list_tool_agents() -> list:
    """List all registered tool agents with specialized prompts.

    Returns:
        List of tool agent names.
    """
    return list(TOOL_AGENT_PROMPTS.keys())
