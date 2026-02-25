"""Agent Tools Registry - static definitions only.

This module exports static agent definitions (declarations, configs, prompts).
Execute functions are created dynamically by ToolRouter to avoid circular dependencies.
"""

from typing import Dict

from geo_edit.tool_definitions.agents import multimath, gllava, chartmoe

# Agent declarations for API tool definitions
AGENT_DECLARATIONS: Dict[str, dict] = {
    "multimath": multimath.DECLARATION,
    "gllava": gllava.DECLARATION,
    "chartmoe": chartmoe.DECLARATION,
}

# Agent return types
AGENT_RETURN_TYPES: Dict[str, str] = {
    "multimath": multimath.RETURN_TYPE,
    "gllava": gllava.RETURN_TYPE,
    "chartmoe": chartmoe.RETURN_TYPE,
}

# Export model configs for tool_agents manager
AGENT_CONFIGS: Dict[str, dict] = {
    "multimath": multimath.agent_config,
    "gllava": gllava.agent_config,
    "chartmoe": chartmoe.agent_config,
}

# Export system prompts for tool_agents manager
AGENT_SYSTEM_PROMPTS: Dict[str, str] = {
    "multimath": multimath.SYSTEM_PROMPT,
    "gllava": gllava.SYSTEM_PROMPT,
    "chartmoe": chartmoe.SYSTEM_PROMPT,
}
