"""
geo_edit function tools — lightweight CPU-only image manipulation.

Tools: image_crop, image_label, draw_line, draw_path, bounding_box, image_highlight.
No GPU, no Ray, no server dependency.

tool_type = "geo_edit_function"
"""

import importlib.util
import logging
import os

from .base import register_tool
from .geo_edit_base import GeoEditToolBase, _AREAL_ROOT

logger = logging.getLogger(__name__)


def _load_function_tools():
    """Load local function tools via importlib (no Ray)."""
    tools_dir = os.path.join(_AREAL_ROOT, "geo_edit", "tool_definitions", "functions")
    modules = {
        "image_crop": "crop.py",
        "image_label": "label.py",
        "draw_line": "draw_line.py",
        "draw_path": "draw_path.py",
        "bounding_box": "bbox.py",
        "image_highlight": "highlight.py",
    }
    result = {}
    for tool_name, filename in modules.items():
        filepath = os.path.join(tools_dir, filename)
        if not os.path.exists(filepath):
            logger.warning(f"Function tool file not found: {filepath}")
            continue
        spec = importlib.util.spec_from_file_location(filename[:-3], filepath)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        result[tool_name] = (mod.DECLARATION, mod.execute, "function", mod.RETURN_TYPE)
    return result


@register_tool
class GeoEditFunctionTool(GeoEditToolBase):
    """Pure-function image tools (CPU only, no server needed)."""

    tool_type = "geo_edit_function"
    enable_tools = [
        "image_crop", "image_label", "draw_line",
        "draw_path", "bounding_box", "image_highlight",
    ]

    def __init__(self, num_workers=1):
        super().__init__(num_workers)
        self.function_tools = _load_function_tools()
        logger.info(
            f"GeoEditFunctionTool loaded {len(self.function_tools)} tools: "
            f"{sorted(self.function_tools.keys())}"
        )

    def get_usage_inst(self):
        return (
            "Image manipulation: image_crop, image_label, draw_line, "
            "draw_path, bounding_box, image_highlight."
        )
