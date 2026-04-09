"""PaddleOCR Tool Agent - Optical Character Recognition using PaddleOCR-VL-1.5."""

import os
import base64
import re
import json
from io import BytesIO
from typing import Optional

import PIL

PIL.Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError for large images

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
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        # Initialize vLLM with PaddleOCR-VL model
        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=agent_config.get("tensor_parallel_size", 1),
            max_model_len=agent_config.get("max_model_len", max_model_len),
            gpu_memory_utilization=agent_config.get(
                "gpu_memory_utilization", gpu_memory_utilization
            ),
            limit_mm_per_prompt={"image": 10},  # Allow up to 10 images per prompt
        )

        self.max_new_tokens = agent_config.get("max_tokens", 4096)
        self._initialized = True

        logger.info(
            "PaddleOCR-VL (vLLM) initialized on GPU %s: %s", self.gpu_ids, model_path
        )

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
        task = (
            kwargs.get("task", "ocr").strip().lower()
        )  # ocr, table, formula, chart, spotting, seal

        if task not in TASK_PROMPTS:
            return json.dumps(
                {
                    "error": f"Invalid task: {task}. Must be one of: {list(TASK_PROMPTS.keys())}",
                    "task": task,
                },
                ensure_ascii=False,
            )

        # Decode image
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Preprocess image for spotting task
        orig_w, orig_h = image.size
        spotting_upscale_threshold = 1500

        if (
            task == "spotting"
            and orig_w < spotting_upscale_threshold
            and orig_h < spotting_upscale_threshold
        ):
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
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                        {"type": "text", "text": TASK_PROMPTS[task]},
                    ],
                }
            ]

            # Create sampling parameters with repetition penalty to prevent loops
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=self.max_new_tokens,
                repetition_penalty=1.2,  # Penalize repeated tokens
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

                    lines.append(
                        {
                            "text": text,
                            "bbox": [x1, y1, x2, y2],
                        }
                    )

                return json.dumps(
                    {
                        "task": task,
                        "text": lines,
                    },
                    ensure_ascii=False,
                )

            # Return formatted result
            return json.dumps(
                {
                    "task": task,
                    "text": result.strip(),
                },
                ensure_ascii=False,
            )

        except Exception as e:
            logger.error("PaddleOCR-VL failed: %s", e)
            return json.dumps(
                {
                    "error": str(e),
                    "task": task,
                    "text": "",
                },
                ensure_ascii=False,
            )

    def health_check(self) -> dict:
        """Return health status of the actor."""
        return {
            "model": self.model_name,
            "initialized": self._initialized,
        }


ACTOR_CLASS = PaddleOCRActor
RETURN_TYPE = "text"


# ============ Map OCR Post-processing Functions ============


def filter_map_text(text_results: list) -> list:
    """Filter map OCR results to keep meaningful text only.

    Filters out:
    - Pure numbers (e.g., "123", "45.6")
    - Single characters (usually noise)
    - Pure symbols

    Keeps:
    - Mixed text like "A1出口", "3号线", "北京路123号"
    """
    filtered = []
    for item in text_results:
        text = item["text"].strip()
        # 1. Filter pure numbers (including decimals)
        if re.match(r"^[\d.,\s]+$", text):
            continue
        # 2. Filter single characters (usually noise)
        if len(text) <= 1:
            continue
        # 3. Filter pure symbols
        if re.match(r"^[^\w\u4e00-\u9fff]+$", text):
            continue
        filtered.append(item)
    return filtered


def merge_nearby_text(text_results: list, distance_threshold: int = 50) -> list:
    """Merge spatially adjacent text blocks.

    Args:
        text_results: List of {"text": str, "bbox": [x1,y1,x2,y2]}
        distance_threshold: Max pixel distance to consider as "nearby"

    Returns:
        Merged text results
    """
    if not text_results:
        return []

    # Sort by y-coordinate, then by x-coordinate
    sorted_results = sorted(text_results, key=lambda x: (x["bbox"][1], x["bbox"][0]))

    merged = []
    current = sorted_results[0].copy()

    for item in sorted_results[1:]:
        curr_bbox = current["bbox"]
        next_bbox = item["bbox"]

        # Check if on same line (y-coords close) and horizontally adjacent
        same_line = abs(curr_bbox[1] - next_bbox[1]) < distance_threshold
        horizontal_near = next_bbox[0] - curr_bbox[2] < distance_threshold

        if same_line and horizontal_near:
            # Merge text and bbox
            current["text"] = current["text"] + " " + item["text"]
            current["bbox"] = [
                min(curr_bbox[0], next_bbox[0]),
                min(curr_bbox[1], next_bbox[1]),
                max(curr_bbox[2], next_bbox[2]),
                max(curr_bbox[3], next_bbox[3]),
            ]
        else:
            merged.append(current)
            current = item.copy()

    merged.append(current)
    return merged


def process_map_ocr_result(text_results: list) -> list:
    """Full processing pipeline for map OCR: filter -> merge."""
    filtered = filter_map_text(text_results)
    merged = merge_nearby_text(filtered)
    return merged


def filter_map_text_string(text: str) -> str:
    """Filter map OCR plain text results to keep meaningful text only.

    Filters out lines/tokens that are:
    - Pure numbers (e.g., "123", "45.6")
    - Single characters (usually noise)
    - Pure symbols

    Args:
        text: Plain text string from OCR result.

    Returns:
        Filtered text with each valid item on a new line.
    """
    filtered_lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # 1. Filter pure numbers (including decimals)
        if re.match(r"^[\d.,\s]+$", line):
            continue
        # 2. Filter single characters (usually noise)
        if len(line) <= 1:
            continue
        # 3. Filter pure symbols
        if re.match(r"^[^\w\u4e00-\u9fff]+$", line):
            continue
        filtered_lines.append(line)
    return "\n".join(filtered_lines)


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
                    "description": "The index of the image to analyze (e.g., 0 for Observation 0).",
                }
            },
            "required": ["image_index"],
        },
        "fixed_task": "ocr",
        "return_type": "text",
    },
    "table_ocr": {
        "name": "table_ocr",
        "description": "Table structure recognition tool. Extracts tabular data including rows, columns, and cell contents. Best for: spreadsheet images, data tables, structured documents.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_index": {
                    "type": "integer",
                    "description": "The index of the image to analyze (e.g., 0 for Observation 0).",
                }
            },
            "required": ["image_index"],
        },
        "fixed_task": "table",
        "return_type": "text",
    },
    "formula_ocr": {
        "name": "formula_ocr",
        "description": "Mathematical formula recognition tool. Converts mathematical expressions and equations to LaTeX format. Best for: math formulas, equations, scientific notation.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_index": {
                    "type": "integer",
                    "description": "The index of the image to analyze (e.g., 0 for Observation 0).",
                }
            },
            "required": ["image_index"],
        },
        "fixed_task": "formula",
        "return_type": "text",
    },
    "chart_text_ocr": {
        "name": "chart_text_ocr",
        "description": "Chart text extraction tool. Recognizes and extracts text elements from charts and diagrams including labels, values, legends, and axis titles. Best for: bar charts, line graphs, pie charts.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_index": {
                    "type": "integer",
                    "description": "The index of the image to analyze (e.g., 0 for Observation 0).",
                }
            },
            "required": ["image_index"],
        },
        "fixed_task": "chart",
        "return_type": "text",
    },
    "text_spotting": {
        "name": "text_spotting",
        "description": "Text spotting tool with precise localization. Returns both text content AND bounding box coordinates for each detected text region. Best for: maps, annotated images, scene text with location needs.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_index": {
                    "type": "integer",
                    "description": "The index of the image to analyze (e.g., 0 for Observation 0).",
                }
            },
            "required": ["image_index"],
        },
        "fixed_task": "spotting",
        "return_type": "text",
    },
    "seal_ocr": {
        "name": "seal_ocr",
        "description": "Seal and stamp recognition tool. Specialized for recognizing text from official seals, stamps, and signatures. Best for: official documents, certificates, contracts with seals.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_index": {
                    "type": "integer",
                    "description": "The index of the image to analyze (e.g., 0 for Observation 0).",
                }
            },
            "required": ["image_index"],
        },
        "fixed_task": "seal",
        "return_type": "text",
    },
    "map_text_ocr": {
        "name": "map_text_ocr",
        "description": "Text recognition optimized for maps. Extracts place names, road names, and landmarks while filtering out noise like pure numbers, distances, and scale markers.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_index": {
                    "type": "integer",
                    "description": "The index of the image to analyze (e.g., 0 for Observation 0).",
                }
            },
            "required": ["image_index"],
        },
        "fixed_task": "ocr",
        "filter_map": True,
        "return_type": "text",
    },
}
