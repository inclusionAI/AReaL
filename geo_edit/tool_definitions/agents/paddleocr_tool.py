"""PaddleOCR Tool Agent - Optical Character Recognition using PaddleOCR-VL-1.5."""

import base64
import json
from io import BytesIO
from typing import Optional

from geo_edit.environment.tool_agents.actor import BaseToolModelActor
from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)

# PaddleOCR doesn't need a system prompt (not a language model)
SYSTEM_PROMPT = ""

# Model configuration
agent_config = {
    "model_name_or_path": "PaddlePaddle/PaddleOCR-VL-1.5",
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.8,
    "temperature": 0.0,
    "max_tokens":4096,
    "num_gpus": 1,
}

# Local model configuration
# Set model_path to a local path or HuggingFace model ID
LOCAL_MODEL_CONFIG = {
    "model_path": "PaddlePaddle/PaddleOCR-VL-1.5",  # Model path (HuggingFace or local)
    "tensor_parallel_size": 1,  # Number of GPUs for tensor parallelism
    "max_model_len": 8192,      # Maximum context length
    "max_num_seqs": 5,          # Maximum number of sequences to process in parallel
    "max_new_tokens": 4096,     # Maximum tokens to generate
}


# Task prompts for PaddleOCR-VL
TASK_PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
    "spotting": "Spotting:",
    "seal": "Seal Recognition:",
}


class PaddleOCRActor(BaseToolModelActor):
    """PaddleOCR Actor using PaddleOCR-VL-1.5 with vLLM for high-performance inference."""

    def __init__(
        self,
        model_name: str,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.8,
        system_prompt: Optional[str] = None,
    ):
        from vllm import LLM, SamplingParams

        self.setup_gpu()  # Configure GPU based on Ray assignment

        self.model_name = model_name
        model_path = LOCAL_MODEL_CONFIG["model_path"]

        logger.info("Loading PaddleOCR-VL model with vLLM: %s", model_path)

        # Initialize vLLM
        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=LOCAL_MODEL_CONFIG.get("tensor_parallel_size", 1),
            max_model_len=LOCAL_MODEL_CONFIG.get("max_model_len", max_model_len),
            max_num_seqs=LOCAL_MODEL_CONFIG.get("max_num_seqs", 5),
        )

        self.max_new_tokens = LOCAL_MODEL_CONFIG.get("max_new_tokens", 4096)
        self._initialized = True

        logger.info("PaddleOCR-VL (vLLM) initialized on GPU %s: %s", self.gpu_ids, model_path)

    def analyze(
        self,
        image_b64: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs,
    ) -> str:
        """Run PaddleOCR-VL and return JSON with OCR results.

        Args:
            image_b64: Base64-encoded image string.
            temperature: Temperature for sampling.
            max_tokens: Maximum tokens to generate.
            **kwargs: Tool-specific parameters, expects 'task' (ocr, table, formula, chart, spotting, seal).

        Returns:
            JSON string with OCR results.
        """
        from vllm import SamplingParams
        from PIL import Image

        # Extract parameters
        task = kwargs.get("task", "ocr").strip().lower()  # ocr, table, formula, chart, spotting, seal

        if task not in TASK_PROMPTS:
            return json.dumps({
                "error": f"Invalid task: {task}. Must be one of: {list(TASK_PROMPTS.keys())}",
                "task": task
            })

        # Decode image
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Preprocess image for spotting task
        orig_w, orig_h = image.size
        spotting_upscale_threshold = 1500

        if task == "spotting" and orig_w < spotting_upscale_threshold and orig_h < spotting_upscale_threshold:
            process_w, process_h = orig_w * 2, orig_h * 2
            try:
                resample_filter = Image.Resampling.LANCZOS
            except AttributeError:
                resample_filter = Image.LANCZOS
            image = image.resize((process_w, process_h), resample_filter)

        try:
            # Prepare conversation with image and prompt
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": TASK_PROMPTS[task]},
                    ]
                }
            ]

            # Create sampling parameters
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=self.max_new_tokens,
            )

            # Run inference with vLLM
            outputs = self.llm.chat(
                messages=conversation,
                sampling_params=sampling_params,
            )

            # Extract result
            result = outputs[0].outputs[0].text

            # Return formatted result
            return json.dumps({
                "task": task,
                "text": result.strip(),
                "success": True
            })

        except Exception as e:
            logger.error("PaddleOCR-VL failed: %s", e)
            return json.dumps({
                "error": str(e),
                "task": task,
                "text": "",
                "success": False
            })

    def health_check(self) -> dict:
        """Return health status of the actor."""
        return {
            "model": self.model_name,
            "initialized": self._initialized,
        }


ACTOR_CLASS = PaddleOCRActor

DECLARATION = {
    "name": "paddleocr",
    "description": """PaddleOCR-VL text recognition tool. Advanced vision-language model for OCR with multiple task modes:
- task='ocr': Standard text recognition (default)
- task='table': Table structure recognition
- task='formula': Mathematical formula recognition
- task='chart': Chart/diagram recognition
- task='spotting': Text spotting with precise localization
- task='seal': Seal/stamp recognition

Supports multilingual text recognition with high accuracy.""",
    "parameters": {
        "type": "object",
        "properties": {
            "image_index": {
                "type": "integer",
                "description": "The index of the image to analyze. Each image is assigned an index like 'Observation 0', 'Observation 1', etc."
            },
            "task": {
                "type": "string",
                "enum": ["ocr", "table", "formula", "chart", "spotting", "seal"],
                "description": "Recognition task type. Default is 'ocr' for standard text recognition."
            }
        },
        "required": ["image_index"]
    }
}

RETURN_TYPE = "text"
