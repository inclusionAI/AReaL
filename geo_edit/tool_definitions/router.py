"""Tool Router - central registry for all tools.

Controls tool availability based on:
1. config.yaml - static enable/disable per tool
2. tool_mode - runtime mode ("auto", "force", "direct")
   - "auto"/"force": use enabled tools from config
   - "direct": disable ALL tools (no tool system prompt)
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

import yaml
from PIL import Image
from geo_edit.tool_definitions.functions import FUNCTION_TOOLS
from geo_edit.tool_definitions.agents import (
    AGENT_DECLARATIONS,
    AGENT_RETURN_TYPES,
    AGENT_CONFIGS,
)
from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)

# =============================================================================
# Configuration
# =============================================================================

_CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
    _TOOL_CONFIG: Dict[str, bool] = yaml.safe_load(f)


def _make_agent_execute(agent_name: str) -> Callable[[List[Image.Image], int, str], str]:
    """Create an execute function for a specific agent."""
    def execute(image_list: List[Image.Image], image_index: int, question: str) -> str:
        from geo_edit.environment.tool_agents import call_agent
        return call_agent(agent_name, image_list, image_index, question)
    return execute


def _build_tool_registry() -> Dict[str, tuple]:
    """Build the complete tool registry from functions and agents."""
    registry = dict(FUNCTION_TOOLS)

    # Add agent tools with dynamically created execute functions
    for name, declaration in AGENT_DECLARATIONS.items():
        registry[name] = (
            declaration,
            _make_agent_execute(name),
            "agent",
            AGENT_RETURN_TYPES[name],
        )

    return registry


_TOOL_REGISTRY: Dict[str, tuple] = _build_tool_registry()


# =============================================================================
# ToolRouter Class
# =============================================================================

class ToolRouter:
    """Router for dynamic tool selection and agent lifecycle management.

    Args:
        tool_mode: "auto"/"force" to use enabled tools, "direct" to disable all tools.
        node_resource: Ray custom resource name to schedule agents on specific nodes.
            E.g., "tool_agent" will add {"tool_agent": 1} to each agent's resources.
        ray_address: Ray cluster address for agent initialization.
        skip_agent_init: If True, skip automatic agent initialization (useful for worker processes
            that should connect to existing agents without re-initializing them).
    """

    def __init__(
        self,
        tool_mode: Literal["auto", "force", "direct"] = "auto",
        node_resource: str = "tool_agent",
        ray_address: str = "auto",
        skip_agent_init: bool = False,
    ):
        self.tool_mode = tool_mode
        self._agents: Dict[str, Any] = {}

        # Auto-initialize agents if any are enabled (unless explicitly skipped)
        if not skip_agent_init and self.get_enabled_agents():
            self._agents = self._create_agents(ray_address, node_resource)

    def _get_enabled_tool_names(self) -> List[str]:
        """Get names of all enabled tools (respects tool_mode and config.yaml)."""
        if self.tool_mode == "direct":
            return []
        return [name for name in _TOOL_REGISTRY if _TOOL_CONFIG.get(name, False)]

    def _create_agents(
        self,
        ray_address: str,
        node_resource: Optional[str],
    ) -> Dict[str, Any]:
        """Create Ray Actors for enabled agent tools."""
        from geo_edit.environment.tool_agents import get_manager

        agent_configs = self.get_enabled_agent_configs()
        if not agent_configs:
            return {}

        # Add node resource to each agent config if specified
        if node_resource:
            for name in agent_configs:
                agent_configs[name] = agent_configs[name].copy()
                agent_configs[name]["resources"] = {node_resource: 1}

        logger.info("Initializing Tool Agents (Ray Actors)...")
        agents = get_manager().create_agents(agent_configs, ray_address)
        logger.info(f"Created {len(agents)} tool agents: {list(agents.keys())}")
        return agents

    def get_available_declarations(self) -> List[Dict]:
        """Get tool declarations for available tools."""
        return [_TOOL_REGISTRY[name][0] for name in self._get_enabled_tool_names()]

    def get_available_tools(self) -> Dict[str, Callable[..., Image.Image | str]]:
        """Get tool functions for available tools."""
        return {name: _TOOL_REGISTRY[name][1] for name in self._get_enabled_tool_names()}

    def get_tool_return_types(self) -> Dict[str, str]:
        """Get return types for all available tools."""
        return {name: _TOOL_REGISTRY[name][3] for name in self._get_enabled_tool_names()}

    def get_enabled_agents(self) -> List[str]:
        """Get list of enabled agent tool names (for Ray Actor initialization)."""
        return [
            name for name in self._get_enabled_tool_names()
            if _TOOL_REGISTRY[name][2] == "agent"
        ]

    def get_enabled_agent_configs(self) -> Dict[str, dict]:
        """Get configs for enabled agent tools.

        Returns:
            Dict mapping agent name to its config dict.
        """
        return {
            name: AGENT_CONFIGS[name]
            for name in self.get_enabled_agents()
            if name in AGENT_CONFIGS
        }

    def is_tool_enabled(self) -> bool:
        """Check if any tools are enabled."""
        return bool(self._get_enabled_tool_names())

    def is_agent_enabled(self) -> bool:
        """Check if any agent tools are enabled and initialized."""
        return bool(self._agents)

    def shutdown_agents(self, tool_names: Optional[List[str]] = None):
        """Shutdown Tool Agents.

        Args:
            tool_names: List of agent names to shutdown. If None, shutdown all.
        """
        if not self._agents:
            return

        from geo_edit.environment.tool_agents import get_manager

        logger.info("Shutting down Tool Agents...")
        get_manager().shutdown(tool_names)
        self._agents = {}
        logger.info("Tool Agents shutdown complete.")
