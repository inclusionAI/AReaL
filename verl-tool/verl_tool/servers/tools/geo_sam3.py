"""
SAM3 agent — segmentation tools via Ray GPU actor.

Tools: auto_segment, bbox_segment, text_segment,
       exemplar_segment, concept_count, presence_check.

tool_type = "geo_sam3"
"""

from .base import register_tool
from .geo_edit_base import GeoEditAgentToolBase


@register_tool
class GeoSam3Tool(GeoEditAgentToolBase):
    tool_type = "geo_sam3"
    agent_name = "sam3"
    enable_tools = [
        "auto_segment", "bbox_segment", "text_segment",
        "exemplar_segment", "concept_count", "presence_check",
    ]
