"""
Chart-R1 agent — chart analysis tools via Ray GPU actor.

Tools: chart_reasoning, chart_data_extract, chart_trend_analysis.

tool_type = "geo_chartr1"
"""

from .base import register_tool
from .geo_edit_base import GeoEditAgentToolBase


@register_tool
class GeoChartr1Tool(GeoEditAgentToolBase):
    tool_type = "geo_chartr1"
    agent_name = "chartr1"
    enable_tools = ["chart_reasoning", "chart_data_extract", "chart_trend_analysis"]
