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
    "multimath": {"model_path": multimath.MODEL_PATH, "max_tokens": multimath.MAX_TOKENS},
    "gllava": {"model_path": gllava.MODEL_PATH, "max_tokens": gllava.MAX_TOKENS},
    "chartmoe": {"model_path": chartmoe.MODEL_PATH, "max_tokens": chartmoe.MAX_TOKENS},
}
