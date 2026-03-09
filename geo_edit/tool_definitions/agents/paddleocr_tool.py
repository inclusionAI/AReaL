"""PaddleOCR Tool Agent - Optical Character Recognition using PaddleOCR-VL-1.5."""
import os
import base64
import re
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
    "model_name_or_path": "/storage/openpsi/models/PaddleOCR-VL-1.5",
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.8,
    "temperature": 0.0,
    "max_tokens": 4096,
    "num_gpus": 1,
    "tensor_parallel_size": 1,  # Number of GPUs for tensor parallelism
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
        from vllm import LLM

        self.setup_gpu()  # Configure GPU based on Ray assignment

        self.model_name = model_name
        model_path = agent_config["model_name_or_path"]

        logger.info("Loading PaddleOCR-VL model with vLLM: %s", model_path)
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"]="True"
        # Initialize vLLM with PaddleOCR-VL model
        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=agent_config.get("tensor_parallel_size", 1),
            max_model_len=agent_config.get("max_model_len", max_model_len),
            gpu_memory_utilization=agent_config.get("gpu_memory_utilization", gpu_memory_utilization),
            limit_mm_per_prompt={"image": 10},  # Allow up to 10 images per prompt
        )

        self.max_new_tokens = agent_config.get("max_tokens", 4096)
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
            # Prepare message with image and task prompt
            # vLLM expects PIL Image object in the message content
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
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
                messages=messages,
                sampling_params=sampling_params,
            )

            # Extract result from vLLM output
            result = outputs[0].outputs[0].text

            if task == "spotting":
                loc_re = re.compile(r"<\|LOC_(\d+)\|>")

                lines = []
                for raw in result.splitlines():
                    s = raw.strip()
                    if not s:
                        continue

                    locs = [int(x) for x in loc_re.findall(s)]
                    text = loc_re.sub("", s).strip()

                    # Need exactly 8 coords -> x1,y1,x2,y2,x3,y3,x4,y4
                    if len(locs) < 8:
                        continue
                    locs = locs[:8]

                    xs = [locs[0], locs[2], locs[4], locs[6]]
                    ys = [locs[1], locs[3], locs[5], locs[7]]
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

                    lines.append({
                        "text": text,
                        "bbox": [x1, y1, x2, y2],
                    })

                return json.dumps({
                    "task": task,
                    "text": lines,
                })

            # Return formatted result
            return json.dumps({
                "task": task,
                "text": result.strip(),
            })

        except Exception as e:
            logger.error("PaddleOCR-VL failed: %s", e)
            return json.dumps({
                "error": str(e),
                "task": task,
                "text": "",
            })

    def health_check(self) -> dict:
        """Return health status of the actor."""
        return {
            "model": self.model_name,
            "initialized": self._initialized,
        }


ACTOR_CLASS = PaddleOCRActor
RETURN_TYPE = "text"

# Multi-tool declarations - each tool has a fixed task mode
DECLARATIONS = {
    "text_ocr": {
        "name": "text_ocr",
        "description": "General text recognition tool. Extracts all visible text from images with support for 111 languages. Best for: documents, labels, signs, natural scene text.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_index": {
                    "type": "integer",
                    "description": "The index of the image to analyze (e.g., 0 for Observation 0)."
                }
            },
            "required": ["image_index"]
        },
        "fixed_task": "ocr",
        "return_type": "text"
    },
    "table_ocr": {
        "name": "table_ocr",
        "description": "Table structure recognition tool. Extracts tabular data including rows, columns, and cell contents. Best for: spreadsheet images, data tables, structured documents.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_index": {
                    "type": "integer",
                    "description": "The index of the image to analyze (e.g., 0 for Observation 0)."
                }
            },
            "required": ["image_index"]
        },
        "fixed_task": "table",
        "return_type": "text"
    },
    "formula_ocr": {
        "name": "formula_ocr",
        "description": "Mathematical formula recognition tool. Converts mathematical expressions and equations to LaTeX format. Best for: math formulas, equations, scientific notation.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_index": {
                    "type": "integer",
                    "description": "The index of the image to analyze (e.g., 0 for Observation 0)."
                }
            },
            "required": ["image_index"]
        },
        "fixed_task": "formula",
        "return_type": "text"
    },
    "chart_text_ocr": {
        "name": "chart_text_ocr",
        "description": "Chart text extraction tool. Recognizes and extracts text elements from charts and diagrams including labels, values, legends, and axis titles. Best for: bar charts, line graphs, pie charts.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_index": {
                    "type": "integer",
                    "description": "The index of the image to analyze (e.g., 0 for Observation 0)."
                }
            },
            "required": ["image_index"]
        },
        "fixed_task": "chart",
        "return_type": "text"
    },
    "text_spotting": {
        "name": "text_spotting",
        "description": "Text spotting tool with precise localization. Returns both text content AND bounding box coordinates for each detected text region. Best for: maps, annotated images, scene text with location needs.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_index": {
                    "type": "integer",
                    "description": "The index of the image to analyze (e.g., 0 for Observation 0)."
                }
            },
            "required": ["image_index"]
        },
        "fixed_task": "spotting",
        "return_type": "text"
    },
    "seal_ocr": {
        "name": "seal_ocr",
        "description": "Seal and stamp recognition tool. Specialized for recognizing text from official seals, stamps, and signatures. Best for: official documents, certificates, contracts with seals.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_index": {
                    "type": "integer",
                    "description": "The index of the image to analyze (e.g., 0 for Observation 0)."
                }
            },
            "required": ["image_index"]
        },
        "fixed_task": "seal",
        "return_type": "text"
    }
}
