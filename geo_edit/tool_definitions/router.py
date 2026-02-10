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
from geo_edit.environment.tool_agents import get_manager
from geo_edit.tool_definitions.functions import FUNCTION_TOOLS
from geo_edit.tool_definitions.agents import AGENT_TOOLS, AGENT_CONFIGS
from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)

# =============================================================================
# Configuration
# =============================================================================

_CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
    _TOOL_CONFIG: Dict[str, bool] = yaml.safe_load(f)

# Merge all tool registries
_TOOL_REGISTRY: Dict[str, tuple] = {**FUNCTION_TOOLS, **AGENT_TOOLS}


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
    """

    def __init__(
        self,
        tool_mode: Literal["auto", "force", "direct"] = "auto",
        node_resource: Optional[str] = None,
        ray_address: str = "auto",
    ):
        self.tool_mode = tool_mode
        self._agents: Dict[str, Any] = {}

        # Auto-initialize agents if any are enabled
        if self._has_enabled_agents():
            self._agents = self._create_agents(ray_address, node_resource)

    def _has_enabled_agents(self) -> bool:
        """Check if any agent tools are enabled (internal use before full init)."""
        if self.tool_mode == "direct":
            return False
        return any(
            typ == "agent" and _TOOL_CONFIG.get(name, False)
            for name, (_, _, typ, _) in _TOOL_REGISTRY.items()
        )

    def _create_agents(
        self,
        ray_address: str,
        node_resource: Optional[str],
    ) -> Dict[str, Any]:
        """Create Ray Actors for enabled agent tools."""
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
        if self.tool_mode == "direct":
            return []
        return [
            _TOOL_REGISTRY[name][0]
            for name in _TOOL_REGISTRY
            if _TOOL_CONFIG.get(name, False)
        ]

    def get_available_tools(self) -> Dict[str, Callable[..., Image.Image | str]]:
        """Get tool functions for available tools."""
        if self.tool_mode == "direct":
            return {}
        return {
            name: _TOOL_REGISTRY[name][1]
            for name in _TOOL_REGISTRY
            if _TOOL_CONFIG.get(name, False)
        }

    def get_tool_return_type(self, name: str) -> str:
        """Get return type for a tool."""
        return _TOOL_REGISTRY[name][3]

    def get_enabled_agents(self) -> List[str]:
        """Get list of enabled agent tool names (for Ray Actor initialization)."""
        if self.tool_mode == "direct":
            return []
        return [
            name
            for name, (_, _, typ, _) in _TOOL_REGISTRY.items()
            if typ == "agent" and _TOOL_CONFIG.get(name, False)
        ]

    def get_enabled_agent_configs(self) -> Dict[str, dict]:
        """Get configs for enabled agent tools.

        Returns:
            Dict mapping agent name to its config dict.
        """
        if self.tool_mode == "direct":
            return {}
        return {
            name: AGENT_CONFIGS[name]
            for name in self.get_enabled_agents()
            if name in AGENT_CONFIGS
        }

    def is_tool_enabled(self) -> bool:
        """Check if any tools are enabled."""
        if self.tool_mode == "direct":
            return False
        return any(_TOOL_CONFIG.get(name, False) for name in _TOOL_REGISTRY)

    def is_agent_enabled(self) -> bool:
        """Check if any agent tools are enabled and initialized."""
        return len(self._agents) > 0

    def shutdown_agents(self, tool_names: Optional[List[str]] = None):
        """Shutdown Tool Agents.

        Args:
            tool_names: List of agent names to shutdown. If None, shutdown all.
        """
        if not self._agents:
            return

        logger.info("Shutting down Tool Agents...")
        get_manager().shutdown(tool_names)
        self._agents = {}
        logger.info("Tool Agents shutdown complete.")
