"""Tool Agent System - VLM models with GPU-resident weights via Ray Actors."""

from typing import Any, Dict, List, Literal, Optional

from .manager import ToolAgentManager, get_manager
from .actor import ToolModelActor

__all__ = [
    "ToolAgentManager",
    "ToolModelActor",
    "get_manager",
    "create_agents",
    "auto_create_agents",
    "call_agent",
    "shutdown_agents",
]


def create_agents(configs: dict, ray_address: str = "auto"):
    """Create Tool Agents with the given configurations."""
    return get_manager().create_agents(configs, ray_address)


def auto_create_agents(
    ray_address: str = "auto",
    node_resource: Optional[str] = None,
    tool_mode: Literal["auto", "force", "direct"] = "auto",
) -> Dict[str, Any]:
    """Auto-create enabled agent tools based on config.yaml.

    Reads enabled agents from config.yaml and creates Ray actors for each.

    Args:
        ray_address: Ray cluster address.
        node_resource: Custom resource name to schedule agents on specific node.
            E.g., "node1_gpu" will add {"node1_gpu": 1} to each agent's resources.
        tool_mode: Tool mode ("auto", "force", "direct").

    Returns:
        Dict mapping agent names to their Ray actor handles.

    Example:
        # Start worker node with: ray start --address='head:6379' --resources='{"worker_gpu": 8}'
        auto_create_agents(node_resource="worker_gpu")
    """
    from geo_edit.tool_definitions.router import ToolRouter

    router = ToolRouter(tool_mode=tool_mode)
    agent_configs = router.get_enabled_agent_configs()

    if not agent_configs:
        return {}

    # Add node resource to each agent config if specified
    if node_resource:
        for name in agent_configs:
            agent_configs[name] = agent_configs[name].copy()
            agent_configs[name]["resources"] = {node_resource: 1}

    return get_manager().create_agents(agent_configs, ray_address)


def call_agent(tool_name: str, image_list, image_index: int, question: str) -> str:
    """Call a Tool Agent to analyze an image."""
    return get_manager().call(tool_name, image_list, image_index, question)


def shutdown_agents(tool_names: Optional[List[str]] = None):
    """Shutdown Tool Agents."""
    get_manager().shutdown(tool_names)