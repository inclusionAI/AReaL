from typing import Callable, Dict, List

from PIL import Image

from geo_edit.environment.action.image_edition_tool import (
    bounding_box_function,
    bounding_box_function_declaration,
    draw_line_function,
    draw_line_function_declaration,
    image_crop_function,
    image_crop_function_declaration,
    image_highlight_function,
    image_highlight_function_declaration,
    image_label_function,
    image_label_function_declaration,
)
from geo_edit.environment.action.model_analysis_tool import (
    chartmoe_function,
    gllava_function,
    multimath_function,
    chartmoe_function_declaration,
    gllava_function_declaration,
    multimath_function_declaration,
)

# Local image-editing tools (no extra model server required).
TOOL_FUNCTIONS_DECLARE = [
    image_crop_function_declaration,
    image_label_function_declaration,
    draw_line_function_declaration,
    bounding_box_function_declaration,
    image_highlight_function_declaration,
]
TOOL_FUNCTIONS = {
    "image_crop": image_crop_function,
    "image_label": image_label_function,
    "draw_line": draw_line_function,
    "bounding_box": bounding_box_function,
    "image_highlight": image_highlight_function,
}

# Model-based analysis tools (must call tool-agent servers).
TOOL_AGENT_DECLARE = [
    # multimath_function_declaration,
    # gllava_function_declaration,
    # chartmoe_function_declaration,
]
TOOL_AGENTS = {
    "multimath": multimath_function,
    "gllava": gllava_function,
    "chartmoe": chartmoe_function,
}

TOOL_RETURN_TYPES = {
    "image_crop": "image",
    "image_label": "image",
    "draw_line": "image",
    "bounding_box": "image",
    "image_highlight": "image",
    "multimath": "text",
    "gllava": "text",
    "chartmoe": "text",
}


def is_tool_agent_enabled() -> bool:
    return len(TOOL_AGENT_DECLARE) > 0


def get_tool_declarations() -> List[Dict[str, object]]:
    declarations = TOOL_FUNCTIONS_DECLARE + TOOL_AGENT_DECLARE
    return declarations


def get_tool_functions() -> Dict[str, Callable[..., Image.Image | str]]:
    tool_functions = TOOL_FUNCTIONS.copy()
    tool_functions.update(TOOL_AGENTS)
    return tool_functions
