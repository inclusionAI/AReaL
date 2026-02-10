"""Tool Router - central registry for all tools.

Controls tool availability based on:
1. config.yaml - static enable/disable per tool
2. tool_mode - runtime mode ("auto", "force", "direct")
   - "auto"/"force": use enabled tools from config
   - "direct": disable ALL tools (no tool system prompt)
"""

from pathlib import Path
from typing import Callable, Dict, List, Literal

import yaml
from PIL import Image

from geo_edit.tool_definitions.functions import FUNCTION_TOOLS
from geo_edit.tool_definitions.agents import AGENT_TOOLS

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
    """Router for dynamic tool selection.

    Args:
        tool_mode: "auto"/"force" to use enabled tools, "direct" to disable all tools.
    """

    def __init__(self, tool_mode: Literal["auto", "force", "direct"] = "auto"):
        self.tool_mode = tool_mode

    def get_available_declarations(self) -> List[Dict]:
        """Get tool declarations for available tools."""
        if self.tool_mode == "direct":
            return []
        return [
            _TOOL_REGISTRY[name][0]
            for name in _TOOL_REGISTRY
            if _TOOL_CONFIG.get(name, False)
        ]

    def get_available_functions(self) -> Dict[str, Callable[..., Image.Image | str]]:
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

    def is_tool_enabled(self) -> bool:
        """Check if any tools are enabled."""
        if self.tool_mode == "direct":
            return False
        return any(_TOOL_CONFIG.get(name, False) for name in _TOOL_REGISTRY)

    def is_agent_enabled(self) -> bool:
        """Check if any agent tools are enabled."""
        return len(self.get_enabled_agents()) > 0
