"""PaddleOCR Tool Agent - Optical Character Recognition."""

import base64
import json
from io import BytesIO
from typing import Optional

import numpy as np

from geo_edit.environment.tool_agents.actor import BaseToolModelActor
from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)

# PaddleOCR doesn't need a system prompt (not a language model)
SYSTEM_PROMPT = ""

# Model configuration
agent_config = {
    "model_name_or_path": "paddleocr",  # Uses default PP-OCRv4
    "max_model_len": 8192,        # Unused, interface compatibility
    "gpu_memory_utilization": 0.8, # Unused, interface compatibility
    "temperature": 0.0,           # Unused
    "max_tokens": 4096,           # Unused
    "num_gpus": 1,
}


def format_text_mode(ocr_results) -> str:
    """Format OCR results as concatenated text.

    Args:
        ocr_results: PaddleOCR output results.

    Returns:
        JSON string with concatenated text.
    """
    if not ocr_results or not ocr_results[0]:
        return json.dumps({"mode": "text", "text": "", "num_lines": 0})

    texts = []
    for line in ocr_results[0]:
        text = line[1][0]  # Extract text from result
        texts.append(text)

    full_text = " ".join(texts)
    return json.dumps({
        "mode": "text",
        "text": full_text,
        "num_lines": len(texts)
    })


def format_lines_mode(ocr_results) -> str:
    """Format OCR results as structured lines with bboxes.

    Args:
        ocr_results: PaddleOCR output results.

    Returns:
        JSON string with structured line data.
    """
    if not ocr_results or not ocr_results[0]:
        return json.dumps({"mode": "lines", "lines": [], "num_lines": 0})

    lines = []
    for line in ocr_results[0]:
        bbox = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        text = line[1][0]
        confidence = float(line[1][1])

        lines.append({
            "text": text,
            "bbox": [[int(p[0]), int(p[1])] for p in bbox],
            "confidence": round(confidence, 3)
        })

    return json.dumps({
        "mode": "lines",
        "lines": lines,
        "num_lines": len(lines)
    })


class PaddleOCRActor(BaseToolModelActor):
    """PaddleOCR Actor using PaddleOCR library."""

    def __init__(
        self,
        model_name: str,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.8,
        system_prompt: Optional[str] = None,
    ):
        from paddleocr import PaddleOCR

        self.setup_gpu()  # Configure GPU based on Ray assignment

        self.model_name = model_name

        # Load PaddleOCR immediately (no lazy loading)
        logger.info("Loading PaddleOCR model: %s", self.model_name)

        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang='ch',
            use_gpu=True,
            show_log=False
        )
        self._initialized = True

        logger.info("PaddleOCRActor initialized on GPU %s: %s", self.gpu_ids, model_name)

    def analyze(
        self,
        image_b64: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs,
    ) -> str:
        """Run PaddleOCR and return JSON with text results.

        Args:
            image_b64: Base64-encoded image string.
            temperature: Unused.
            max_tokens: Unused.
            **kwargs: Tool-specific parameters, expects 'mode' ("text" or "lines").

        Returns:
            JSON string with OCR results based on mode.
        """
        from PIL import Image

        # Extract mode from kwargs (can be 'question' or 'mode')
        mode = kwargs.get("mode", kwargs.get("question", "")).strip().lower()
        if mode not in ["text", "lines"]:
            return json.dumps({
                "error": f"Invalid mode: {mode}. Must be 'text' or 'lines'.",
                "mode": mode
            })

        # Decode image
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        try:
            # Run OCR
            results = self.ocr.ocr(np.array(image), cls=True)

            # Format output based on mode
            if mode == "text":
                return format_text_mode(results)
            else:  # mode == "lines"
                return format_lines_mode(results)

        except Exception as e:
            logger.error("PaddleOCR failed: %s", e)
            return json.dumps({
                "error": str(e),
                "mode": mode,
                "text": "" if mode == "text" else None,
                "lines": [] if mode == "lines" else None,
                "num_lines": 0
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
    "description": """PaddleOCR text recognition tool. Extracts text from images with two output modes:
- mode='text': Returns concatenated full text as a single string
- mode='lines': Returns structured data with text, bounding boxes, and confidence scores for each line

Supports multiple languages (Chinese, English, etc.)""",
    "parameters": {
        "type": "object",
        "properties": {
            "image_index": {
                "type": "integer",
                "description": "The index of the image to analyze. Each image is assigned an index like 'Observation 0', 'Observation 1', etc."
            },
            "mode": {
                "type": "string",
                "enum": ["text", "lines"],
                "description": "Output format mode. 'text' returns concatenated full text. 'lines' returns structured line data with bounding boxes."
            }
        },
        "required": ["image_index", "mode"]
    }
}

RETURN_TYPE = "text"
