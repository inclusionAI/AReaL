"""Tool Agent System - VLM models with GPU-resident weights via Ray Actors."""

from .manager import ToolAgentManager, get_manager
from .actor import ToolModelActor

__all__ = [
    "ToolAgentManager",
    "ToolModelActor",
    "get_manager",
    "create_agents",
    "call_agent",
    "shutdown_agents",
]


def create_agents(configs: dict, ray_address: str = "auto"):
    """Create Tool Agents with the given configurations."""
    return get_manager().create_agents(configs, ray_address)


def call_agent(tool_name: str, image_list, image_index: int, question: str) -> str:
    """Call a Tool Agent to analyze an image."""
    return get_manager().call(tool_name, image_list, image_index, question)


def shutdown_agents(tool_names: list = None):
    """Shutdown Tool Agents."""
    get_manager().shutdown(tool_names)