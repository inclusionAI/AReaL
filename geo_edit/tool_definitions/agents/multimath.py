"""Multimath VLM Agent Tool."""

from typing import List
from PIL import Image

from geo_edit.environment.tool_agents import call_agent

# Model configuration
agent_config = {
    "model_name_or_path": "/storage/openpsi/models/Qwen3-VL-235B-A22B-Thinking",
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.8,
    "temperature": 0.0,
    "max_tokens": 1024,
    "num_gpus": 1,
}

DECLARATION = {
    "name": "multimath",
    "description": "Use a math-vision agent to analyze the selected image and provide analysis for the final answer.",
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
    return call_agent("multimath", image_list, image_index, question)
