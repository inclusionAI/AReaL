"""Agent Tools Registry - static definitions only.

This module exports static agent definitions (declarations, configs, prompts).
Execute functions are created dynamically by ToolRouter to avoid circular dependencies.

Supports both legacy single-tool declarations and new multi-tool declarations (DECLARATIONS dict).
"""

from typing import Dict, Type, Any

from geo_edit.tool_definitions.agents import multimath, gllava, chartr1, ovr, sam3, grounding_dino, paddleocr_tool, thinkmorph

# Agent declarations for API tool definitions
# Note: paddleocr and sam3 legacy tools removed - use fine-grained multi-tools instead
# Note: chartmoe deprecated - use chartr1 (Chart-R1) instead
AGENT_DECLARATIONS: Dict[str, dict] = {
    "multimath": multimath.DECLARATION,
    "gllava": gllava.DECLARATION,
    "chartr1": chartr1.DECLARATION,
    "ovr": ovr.DECLARATION,
    "grounding_dino": grounding_dino.DECLARATION,
}

# Agent return types
AGENT_RETURN_TYPES: Dict[str, str] = {
    "multimath": multimath.RETURN_TYPE,
    "gllava": gllava.RETURN_TYPE,
    "chartr1": chartr1.RETURN_TYPE,
    "ovr": ovr.RETURN_TYPE,
    "grounding_dino": grounding_dino.RETURN_TYPE,
}

# Export model configs for tool_agents manager
AGENT_CONFIGS: Dict[str, dict] = {
    "multimath": multimath.agent_config,
    "gllava": gllava.agent_config,
    "chartr1": chartr1.agent_config,
    "ovr": ovr.agent_config,
    "sam3": sam3.agent_config,
    "grounding_dino": grounding_dino.agent_config,
    "paddleocr": paddleocr_tool.agent_config,
    "thinkmorph": thinkmorph.agent_config,
}

# Export system prompts for tool_agents manager
AGENT_SYSTEM_PROMPTS: Dict[str, str] = {
    "multimath": multimath.SYSTEM_PROMPT,
    "gllava": gllava.SYSTEM_PROMPT,
    "chartr1": chartr1.SYSTEM_PROMPT,
    "ovr": ovr.SYSTEM_PROMPT,
    "sam3": sam3.SYSTEM_PROMPT,
    "grounding_dino": grounding_dino.SYSTEM_PROMPT,
    "paddleocr": paddleocr_tool.SYSTEM_PROMPT,
    "thinkmorph": thinkmorph.SYSTEM_PROMPT,
}

# Export Actor classes for tool_agents manager
AGENT_ACTOR_CLASSES: Dict[str, Type] = {
    "multimath": multimath.ACTOR_CLASS,
    "gllava": gllava.ACTOR_CLASS,
    "chartr1": chartr1.ACTOR_CLASS,
    "ovr": ovr.ACTOR_CLASS,
    "sam3": sam3.ACTOR_CLASS,
    "grounding_dino": grounding_dino.ACTOR_CLASS,
    "paddleocr": paddleocr_tool.ACTOR_CLASS,
    "thinkmorph": thinkmorph.ACTOR_CLASS,
}

# =============================================================================
# Multi-tool declarations - new fine-grained tool definitions
# =============================================================================

# Mapping from new tool name to (base_agent_name, declaration_dict)
# This allows router to know which Actor to use for each fine-grained tool
MULTI_TOOL_DECLARATIONS: Dict[str, Dict[str, Any]] = {}

# Register PaddleOCR multi-tools
if hasattr(paddleocr_tool, 'DECLARATIONS'):
    for tool_name, decl in paddleocr_tool.DECLARATIONS.items():
        MULTI_TOOL_DECLARATIONS[tool_name] = {
            "declaration": decl,
            "base_agent": "paddleocr",
            "actor_class": paddleocr_tool.ACTOR_CLASS,
            "agent_config": paddleocr_tool.agent_config,
            "system_prompt": paddleocr_tool.SYSTEM_PROMPT,
        }

# Register SAM3 multi-tools (replaces SAM2)
if hasattr(sam3, 'DECLARATIONS'):
    for tool_name, decl in sam3.DECLARATIONS.items():
        MULTI_TOOL_DECLARATIONS[tool_name] = {
            "declaration": decl,
            "base_agent": "sam3",
            "actor_class": sam3.ACTOR_CLASS,
            "agent_config": sam3.agent_config,
            "system_prompt": sam3.SYSTEM_PROMPT,
        }

# Register Multimath multi-tools
if hasattr(multimath, 'DECLARATIONS'):
    for tool_name, decl in multimath.DECLARATIONS.items():
        MULTI_TOOL_DECLARATIONS[tool_name] = {
            "declaration": decl,
            "base_agent": "multimath",
            "actor_class": multimath.ACTOR_CLASS,
            "agent_config": multimath.agent_config,
            # Use tool-specific prompt if available
            "system_prompt": decl.get("fixed_prompt", multimath.SYSTEM_PROMPT),
        }

# Register ThinkMorph multi-tools
if hasattr(thinkmorph, 'DECLARATIONS'):
    for tool_name, decl in thinkmorph.DECLARATIONS.items():
        MULTI_TOOL_DECLARATIONS[tool_name] = {
            "declaration": decl,
            "base_agent": "thinkmorph",
            "actor_class": thinkmorph.ACTOR_CLASS,
            "agent_config": thinkmorph.agent_config,
            "system_prompt": thinkmorph.SYSTEM_PROMPT,
        }

# Register Chart-R1 multi-tools (replaces ChartMoE)
if hasattr(chartr1, 'DECLARATIONS'):
    for tool_name, decl in chartr1.DECLARATIONS.items():
        MULTI_TOOL_DECLARATIONS[tool_name] = {
            "declaration": decl,
            "base_agent": "chartr1",
            "actor_class": chartr1.ACTOR_CLASS,
            "agent_config": chartr1.agent_config,
            "system_prompt": chartr1.SYSTEM_PROMPT,
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
