"""Multimath VLM Agent Tool."""

# System prompt for this agent
SYSTEM_PROMPT = '''
Prompt for Caption
You are a math expert. You will be given an image extracted from a math problem. Follow the instructions carefully.
If the image contains only mathematical expressions, please only output its LaTeX. Your response should only contain its OCR result without other content. For example: $$ x^2 + y^2 = z^2 $$.
Otherwise, execute the following command: Please describe the image in detail in English so that the graphic can be accurately drawn and used to solve a math problem based on your text description. Ensure that your description includes all necessary details, such as text, symbols, geometric markers, etc., if any.
'''
# Model configuration
agent_config = {
    "model_name_or_path": "/storage/openpsi/models/multimath-7b-llava-v1.5",
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.8,
    "temperature": 0.0,
    "max_tokens": 4096,
    "num_gpus": 1,
}

DECLARATION = {
    "name": "multimath",
    "description": "Use a multimodal math-vision parsing tool to process math-problem images, returning either LaTeX OCR (for expression-only images) or a detailed English description of diagrams/plots/charts with all visible text, symbols, labels, and markers for downstream reasoning.",
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
