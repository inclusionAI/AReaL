"""Tool Agent System - VLM models with GPU-resident weights via Ray Actors.

This module provides low-level APIs for managing Tool Agents.
For high-level usage, use ToolRouter from geo_edit.tool_definitions.

Example:
    # High-level API (recommended)
    from geo_edit.tool_definitions import ToolRouter
    router = ToolRouter(tool_mode="auto", node_resource="tool_agent")
    # ... use tools ...
    router.shutdown_agents()

    # Low-level API (for advanced use)
    from geo_edit.environment.tool_agents import get_manager, call_agent
    manager = get_manager()
    result = call_agent("chartmoe", [image], 0, "Analyze this chart")
"""

from typing import TYPE_CHECKING

from .actor import BaseToolModelActor

# Lazy imports to avoid circular dependency
if TYPE_CHECKING:
    from .manager import ToolAgentManager

__all__ = [
    "ToolAgentManager",
    "BaseToolModelActor",
    "get_manager",
    "call_agent",
]


def get_manager() -> "ToolAgentManager":
    """Get the singleton ToolAgentManager instance."""
    from .manager import get_manager as _get_manager
    return _get_manager()


def call_agent(tool_name: str, image_list, image_index: int, **kwargs) -> str:
    """Call a Tool Agent to analyze an image.

    Args:
        tool_name: Name of the agent tool (e.g., "chartmoe", "multimath").
        image_list: List of PIL images.
        image_index: Index of the image to analyze.
        **kwargs: Tool-specific parameters (e.g., question, text_prompt, mode, etc.).

    Returns:
        Analysis result as string.
    """
    from .manager import get_manager as _get_manager
    return _get_manager().call(tool_name, image_list, image_index, **kwargs)
