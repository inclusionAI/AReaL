"""ChartMoE VLM Agent Tool."""

from typing import List
from PIL import Image

from geo_edit.environment.tool_agents import call_agent

MODEL_PATH = "/storage/openpsi/models/Qwen3-VL-235B-A22B-Thinking"
MAX_TOKENS = 1024

DECLARATION = {
    "name": "chartmoe",
    "description": "Use a chart-analysis agent to reason over plots and provide analysis for the final answer.",
    "parameters": {
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
    },
}

RETURN_TYPE = "text"


def execute(image_list: List[Image.Image], image_index: int, question: str) -> str:
    return call_agent("chartmoe", image_list, image_index, question)
