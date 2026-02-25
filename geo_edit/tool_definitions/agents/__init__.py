"""Agent Tools Registry - static definitions only.

This module exports static agent definitions (declarations, configs, prompts).
Execute functions are created dynamically by ToolRouter to avoid circular dependencies.
"""

from typing import Dict, Type

from geo_edit.tool_definitions.agents import multimath, gllava, chartmoe, ovr

# Agent declarations for API tool definitions
AGENT_DECLARATIONS: Dict[str, dict] = {
    "multimath": multimath.DECLARATION,
    "gllava": gllava.DECLARATION,
    "chartmoe": chartmoe.DECLARATION,
    "ovr": ovr.DECLARATION,
}

# Agent return types
AGENT_RETURN_TYPES: Dict[str, str] = {
    "multimath": multimath.RETURN_TYPE,
    "gllava": gllava.RETURN_TYPE,
    "chartmoe": chartmoe.RETURN_TYPE,
    "ovr": ovr.RETURN_TYPE,
}

# Export model configs for tool_agents manager
AGENT_CONFIGS: Dict[str, dict] = {
    "multimath": multimath.agent_config,
    "gllava": gllava.agent_config,
    "chartmoe": chartmoe.agent_config,
    "ovr": ovr.agent_config,
}

# Export system prompts for tool_agents manager
AGENT_SYSTEM_PROMPTS: Dict[str, str] = {
    "multimath": multimath.SYSTEM_PROMPT,
    "gllava": gllava.SYSTEM_PROMPT,
    "chartmoe": chartmoe.SYSTEM_PROMPT,
    "ovr": ovr.SYSTEM_PROMPT,
}

# Export Actor classes for tool_agents manager
AGENT_ACTOR_CLASSES: Dict[str, Type] = {
    "multimath": multimath.ACTOR_CLASS,
    "gllava": gllava.ACTOR_CLASS,
    "chartmoe": chartmoe.ACTOR_CLASS,
    "ovr": ovr.ACTOR_CLASS,
}


def get_actor_class(agent_name: str) -> Type:
    """Get the Actor class for a specific agent.

    Args:
        agent_name: Name of the agent.

    Returns:
        The Actor class for the agent.

    Raises:
        KeyError: If agent_name is not found.
    """
    return AGENT_ACTOR_CLASSES[agent_name]
