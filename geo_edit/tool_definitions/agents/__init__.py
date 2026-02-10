"""Agent Tools Registry - auto-discovers tools from this folder."""

from typing import Dict

from geo_edit.tool_definitions.agents import multimath, gllava, chartmoe

# Tool registry: name -> (declaration, function, type, return_type)
AGENT_TOOLS: Dict[str, tuple] = {
    "multimath": (multimath.DECLARATION, multimath.execute, "agent", multimath.RETURN_TYPE),
    "gllava": (gllava.DECLARATION, gllava.execute, "agent", gllava.RETURN_TYPE),
    "chartmoe": (chartmoe.DECLARATION, chartmoe.execute, "agent", chartmoe.RETURN_TYPE),
}

# Export model configs for tool_agents manager
AGENT_CONFIGS: Dict[str, dict] = {
    "multimath": multimath.agent_config,
    "gllava": gllava.agent_config,
    "chartmoe": chartmoe.agent_config,
}
