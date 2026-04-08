"""
PaddleOCR agent — OCR/text recognition tools via Ray GPU actor.

Tools: text_ocr, table_ocr, formula_ocr, chart_text_ocr,
       text_spotting, seal_ocr, map_text_ocr.

tool_type = "geo_paddleocr"
"""

from .base import register_tool
from .geo_edit_base import GeoEditAgentToolBase


@register_tool
class GeoPaddleocrTool(GeoEditAgentToolBase):
    tool_type = "geo_paddleocr"
    agent_name = "paddleocr"
    enable_tools = [
        "text_ocr", "table_ocr", "formula_ocr", "chart_text_ocr",
        "text_spotting", "seal_ocr", "map_text_ocr",
    ]
