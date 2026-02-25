"""ChartMoE VLM Agent Tool."""

# System prompt for this agent (empty string means no system prompt)
SYSTEM_PROMPT = ""

# Model configuration
agent_config = {
    "model_name_or_path": "/storage/openpsi/models/chartmoe",
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.8,
    "temperature": 0.0,
    "max_tokens": 4096,
    "num_gpus": 1,
}

DECLARATION = {
    "name": "chartmoe",
    "description": "Use a chart-analysis VLM tool to read charts from images and return structured outputs (e.g., table/JSON) plus chart-grounded analysis. It can extract chart elements, visible values/relationships, trends, and key observations for downstream reasoning.",
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
