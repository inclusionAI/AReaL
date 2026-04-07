"""Chart-R1 VLM Agent Tool - Chart reasoning with chain-of-thought.

Based on: https://huggingface.co/DocTron/Chart-R1
Model: Qwen2.5-VL-7B-Instruct based chart understanding model with R1-style reasoning.
"""

import base64
import re
from io import BytesIO
from typing import Optional

from geo_edit.environment.tool_agents.actor import BaseToolModelActor
from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)

# System prompt to trigger <think>/<answer> format
SYSTEM_PROMPT = """You are a chart analysis expert. When analyzing charts, think through your reasoning inside <think></think> tags, then provide a concise analysis inside <answer></answer> tags."""

# Model configuration
agent_config = {
    "model_name_or_path": "/storage/openpsi/models/Chart-R1",
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.8,
    "temperature": 0.01,  # Chart-R1 recommendation
    "max_tokens": 2048,   # Chart-R1 recommendation
    "num_gpus": 1,
    "num_replicas": 3, 
}

# Fixed prompts for different chart analysis modes
CHART_DATA_EXTRACT_PROMPT = """Extract data from this chart. Include chart type, data points with values, axis information, and legend items."""

CHART_TREND_ANALYSIS_PROMPT = """Analyze trends in this chart. Describe overall direction, notable changes, comparisons between series, and key patterns."""


class ChartR1Actor(BaseToolModelActor):
    """Chart-R1 VLM Actor using vLLM for Qwen2.5-VL inference.

    Features:
    - Chain-of-thought reasoning with <think>/<answer> tags
    - High-resolution chart understanding
    - Multiple tool modes: reasoning, data extraction, trend analysis
    """

    def __init__(
        self,
        model_name: str,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.8,
        system_prompt: Optional[str] = None,
    ):
        """Initialize Chart-R1 actor with vLLM.

        Args:
            model_name: Path to Chart-R1 model.
            max_model_len: Maximum sequence length.
            gpu_memory_utilization: Fraction of GPU memory to use.
            system_prompt: Optional system prompt override.
        """
        from vllm import LLM

        self.setup_gpu()  # Configure GPU based on Ray assignment

        self.model_name = model_name
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.max_model_len = max_model_len

        # Initialize vLLM engine for Qwen2.5-VL
        logger.info("Loading Chart-R1 model with vLLM: %s", model_name)
        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype="bfloat16",
            # Qwen2.5-VL specific: limit images per prompt
            limit_mm_per_prompt={"image": 1},
        )

        self._initialized = True
        logger.info("ChartR1Actor (vLLM) initialized on GPU %s: %s", self.gpu_ids, model_name)

    def analyze(
        self,
        image_b64: str,
        temperature: float = 0.01,
        max_tokens: int = 2048,
        **kwargs,
    ) -> str:
        """Analyze a chart image with chain-of-thought reasoning.

        Args:
            image_b64: Base64-encoded chart image.
            temperature: Sampling temperature (default 0.01 for near-deterministic).
            max_tokens: Maximum tokens to generate.
            **kwargs: Tool-specific parameters including 'question'.

        Returns:
            Analysis result (only <answer> content, <think> is not exposed).
        """
        from vllm import SamplingParams

        # Extract question from kwargs
        question = kwargs.get("question", "")

        # Build Qwen2.5-VL style messages with base64 data URL
        image_data_url = f"data:image/jpeg;base64,{image_b64}"

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_url}},
                {"type": "text", "text": question},
            ],
        })

        # Sampling parameters following Chart-R1 recommendations
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.001,  # From Chart-R1 paper
            top_k=1,      # From Chart-R1 paper
        )

        # Generate with vLLM
        outputs = self.llm.chat(messages, sampling_params=sampling_params)
        generated_text = outputs[0].outputs[0].text

        # Parse output: only return <answer> content
        return self._parse_output(generated_text)

    def _parse_output(self, text: str) -> str:
        """Parse Chart-R1 output, only return <answer> content.

        The <think> tag contains internal reasoning process and is NOT exposed.
        The <answer> tag contains the analysis result for the main model.

        Args:
            text: Raw model output.

        Returns:
            Content inside <answer> tags, or full text as fallback.
        """
        answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)

        if answer_match:
            return answer_match.group(1).strip()

        # Fallback: return full text if no answer tags found
        return text.strip()

    def health_check(self) -> dict:
        """Return health status of the actor."""
        return {"model": self.model_name, "initialized": self._initialized}


ACTOR_CLASS = ChartR1Actor

# Legacy single declaration (backward compatibility)
DECLARATION = {
    "name": "chartr1",
    "description": "Use a chart reasoning VLM tool to analyze charts with chain-of-thought capabilities. Returns analytical insights about chart content, data relationships, and patterns for downstream reasoning.",
    "parameters": {
        "type": "object",
        "properties": {
            "image_index": {
                "type": "integer",
                "description": "Observation image index, such as 0 for Observation 0.",
            },
            "question": {
                "type": "string",
                "description": "What you want to analyze about the selected chart.",
            },
        },
        "required": ["image_index", "question"],
    },
}

RETURN_TYPE = "text"

# Multi-tool declarations
DECLARATIONS = {
    "chart_reasoning": {
        "name": "chart_reasoning",
        "description": "Chart reasoning tool with chain-of-thought capabilities. Analyzes charts to extract insights, understand data relationships, and identify key observations for downstream reasoning.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_index": {
                    "type": "integer",
                    "description": "The index of the chart image to analyze (e.g., 0 for Observation 0).",
                },
                "question": {
                    "type": "string",
                    "description": "What you want to analyze about the chart.",
                },
            },
            "required": ["image_index", "question"],
        },
        "return_type": "text",
    },
    "chart_data_extract": {
        "name": "chart_data_extract",
        "description": "Chart data extraction tool. Extracts data points, values, axis information, legend items, and structure from charts into structured format for downstream reasoning.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_index": {
                    "type": "integer",
                    "description": "The index of the chart image to analyze (e.g., 0 for Observation 0).",
                },
            },
            "required": ["image_index"],
        },
        "fixed_prompt": CHART_DATA_EXTRACT_PROMPT,
        "return_type": "text",
    },
    "chart_trend_analysis": {
        "name": "chart_trend_analysis",
        "description": "Chart trend analysis tool. Identifies trends, patterns, comparisons, and key changes in charts. Describes overall direction, notable peaks/valleys, and significant observations for downstream reasoning.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_index": {
                    "type": "integer",
                    "description": "The index of the chart image to analyze (e.g., 0 for Observation 0).",
                },
            },
            "required": ["image_index"],
        },
        "fixed_prompt": CHART_TREND_ANALYSIS_PROMPT,
        "return_type": "text",
    },
}
