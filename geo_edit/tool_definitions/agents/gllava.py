"""GLLaVA VLM Agent Tool."""

# System prompt for this agent
SYSTEM_PROMPT = (
    "You are a visual language model analysis agent. "
    "Analyze the image carefully and provide your reasoning step by step, but do NOT give the final answer."
)

# Model configuration
agent_config = {
    "model_name_or_path": "/storage/openpsi/models/G-LLaVA-7B",
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.8,
    "temperature": 0.0,
    "max_tokens": 4096,
    "num_gpus": 1,
}

DECLARATION = {
    "name": "gllava",
    "description": "Use a geometry-analysis VLM tool to analyze geometric figures and return image-grounded intermediate reasoning (elements, relations, constraints, and useful derivations).",
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
