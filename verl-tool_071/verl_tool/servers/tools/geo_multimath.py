"""
Multimath agent — math image analysis tools via Ray GPU actor.

Tools: math_latex_ocr, math_image_describe.

tool_type = "geo_multimath"
"""

from .base import register_tool
from .geo_edit_base import GeoEditAgentToolBase


@register_tool
class GeoMultimathTool(GeoEditAgentToolBase):
    tool_type = "geo_multimath"
    agent_name = "multimath"
    enable_tools = ["math_latex_ocr", "math_image_describe"]
