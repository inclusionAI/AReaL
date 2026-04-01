"""Function Tools Registry - auto-discovers tools from this folder."""

from typing import Dict

from geo_edit.tool_definitions.functions import crop, label, draw_line, draw_path, bbox, highlight

FUNCTION_TOOLS: Dict[str, tuple] = {
    "image_crop": (crop.DECLARATION, crop.execute, "function", crop.RETURN_TYPE),
    "image_label": (label.DECLARATION, label.execute, "function", label.RETURN_TYPE),
    "draw_line": (draw_line.DECLARATION, draw_line.execute, "function", draw_line.RETURN_TYPE),
    "draw_path": (draw_path.DECLARATION, draw_path.execute, "function", draw_path.RETURN_TYPE),
    "bounding_box": (bbox.DECLARATION, bbox.execute, "function", bbox.RETURN_TYPE),
    "image_highlight": (highlight.DECLARATION, highlight.execute, "function", highlight.RETURN_TYPE),
}
