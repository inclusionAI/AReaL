from __future__ import annotations

import json
import re
from typing import List

from PIL import Image

from geo_edit.environment.action.tool_agent import call_tool_agent_text

_AGENT_PARAMETERS = {
    "type": "object",
    "properties": {
        "image_index": {
            "type": "integer",
            "description": "Observation image index, such as 0 for Observation 0.",
        },
        "question": {
            "type": "string",
            "description": "Question for image analysis. Provide enough detail to derive the final answer.",
        },
    },
    "required": ["image_index", "question"],
}


multimath_function_declaration = {
    "name": "multimath",
    "description": "Use a math-vision agent to analyze the selected image and provide analysis toward a final answer.",
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

bbox_agent_function_declaration = {
    "name": "bbox_agent",
    "description": "Use a bounding box analysis agent to locate regions and provide bounding box oriented analysis for the final answer.",
    "parameters": _AGENT_PARAMETERS,
}


def multimath_function(image_list: List[Image.Image], image_index: int, question: str) -> str:
    return call_tool_agent_text("multimath", image_list=image_list, image_index=image_index, question=question)


def gllava_function(image_list: List[Image.Image], image_index: int, question: str) -> str:
    return call_tool_agent_text("gllava", image_list=image_list, image_index=image_index, question=question)


def chartmoe_function(image_list: List[Image.Image], image_index: int, question: str) -> str:
    return call_tool_agent_text("chartmoe", image_list=image_list, image_index=image_index, question=question)


def bbox_agent_function(image_list: List[Image.Image], image_index: int, question: str) -> str:
    return call_tool_agent_text("bbox_agent", image_list=image_list, image_index=image_index, question=question)


