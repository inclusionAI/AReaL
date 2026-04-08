"""
G-LLaVA agent — geometry VLM for geometric problem analysis via Ray GPU actor.

Tools: gllava.

tool_type = "geo_gllava"
"""

from .base import register_tool
from .geo_edit_base import GeoEditAgentToolBase


@register_tool
class GeoGllavaTool(GeoEditAgentToolBase):
    tool_type = "geo_gllava"
    agent_name = "gllava"
    enable_tools = ["gllava"]
