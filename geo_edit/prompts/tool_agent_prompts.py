"""Tool Agent prompts - imports from tool agent definitions."""

from __future__ import annotations

from geo_edit.tool_definitions.agents import AGENT_SYSTEM_PROMPTS


def get_tool_agent_prompt(tool_name: str) -> str:
    """Get the system prompt for a specific tool agent.

    Args:
        tool_name: Name of the tool agent.

    Returns:
        System prompt string. Returns empty string if not found.
    """
    return AGENT_SYSTEM_PROMPTS.get(tool_name, "")


def list_tool_agents() -> list:
    """List all registered tool agents with system prompts.

    Returns:
        List of tool agent names.
    """
    return list(AGENT_SYSTEM_PROMPTS.keys())
