"""VLM Analysis Tool - declarations, prompts, and Tool Agent calls."""

from __future__ import annotations

from typing import Dict, List, Optional

from PIL import Image

from geo_edit.environment.tool_agents import call_agent

# Tool parameters schema
_AGENT_PARAMETERS = {
    "type": "object",
    "properties": {
        "image_index": {
            "type": "integer",
            "description": "Observation image index, such as 0 for Observation 0.",
        },
        "question": {
            "type": "string",
            "description": "What you want to ask about the selected image.",
        },
    },
    "required": ["image_index", "question"],
}

# Tool declarations
multimath_function_declaration = {
    "name": "multimath",
    "description": "Use a math-vision agent to analyze the selected image and provide analysis for the final answer.",
    "parameters": _AGENT_PARAMETERS,
}

gllava_function_declaration = {
    "name": "gllava",
    "description": "Use a VLM analysis agent on the selected image and return concise analysis for the final answer.",
    "parameters": _AGENT_PARAMETERS,
}

chartmoe_function_declaration = {
    "name": "chartmoe",
    "description": "Use a chart-analysis agent to reason over plots and provide analysis for the final answer.",
    "parameters": _AGENT_PARAMETERS,
}

# Agent-specific system prompts
# Override the default prompt from tool_agents/prompts.py for specialized behavior
AGENT_PROMPTS: Dict[str, str] = {
    "multimath": (
        "You are a math-vision analysis agent. "
        "Analyze the mathematical content in the image and solve the problem step by step. "
        "Return JSON with at least one field in {analysis, text, result, error}."
    ),
    "gllava": (
        "You are a visual language model analysis agent. "
        "Analyze the image carefully and provide a concise, accurate response. "
        "Return JSON with at least one field in {analysis, text, result, error}."
    ),
    "chartmoe": (
        "You are a chart analysis agent specialized in interpreting plots and graphs. "
        "Extract data, identify trends, and provide insights from the visualization. "
        "Return JSON with at least one field in {analysis, text, result, error}."
    ),
}


def get_agent_prompt(tool_name: str) -> Optional[str]:
    """Get the system prompt for a specific tool agent.

    Args:
        tool_name: Name of the tool agent.

    Returns:
        System prompt string if defined, None otherwise.
    """
    return AGENT_PROMPTS.get(tool_name)


def register_agent_prompt(tool_name: str, prompt: str) -> None:
    """Register a custom system prompt for a tool agent.

    Args:
        tool_name: Name of the tool agent.
        prompt: System prompt to use.
    """
    AGENT_PROMPTS[tool_name] = prompt


def list_agents() -> list:
    """List all registered tool agents.

    Returns:
        List of tool agent names.
    """
    return list(AGENT_PROMPTS.keys())


# Tool functions
def multimath_function(image_list: List[Image.Image], image_index: int, question: str) -> str:
    return call_agent("multimath", image_list, image_index, question)


def gllava_function(image_list: List[Image.Image], image_index: int, question: str) -> str:
    return call_agent("gllava", image_list, image_index, question)


def chartmoe_function(image_list: List[Image.Image], image_index: int, question: str) -> str:
    return call_agent("chartmoe", image_list, image_index, question)
