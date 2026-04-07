"""OVR-7B-RL Visual Reasoning Agent Tool (vLLM backend)."""

import base64
import json
from io import BytesIO
from typing import Optional

from geo_edit.environment.tool_agents.actor import BaseToolModelActor
from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)

SYSTEM_PROMPT = (
    "You are a math problem solving expert with visual reasoning capabilities. "
    "According to the user's request, analyze the image carefully and provide your reasoning step by step, but do NOT give the final answer."
)

agent_config = {
    "model_name_or_path": "/storage/openpsi/models/OVR-7B-RL",
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.8,
    "temperature": 0.0,
    "max_tokens": 4096,
    "num_gpus": 1,
}


class OVRActor(BaseToolModelActor):
    """OVR-7B-RL Visual Reasoning Actor using vLLM inference."""

    def __init__(
        self,
        model_name: str,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.8,
        system_prompt: Optional[str] = None,
    ):
        from geo_edit.tool_definitions.agents._vllm_compat import LLM

        self.setup_gpu()  # Configure GPU based on Ray assignment (sets CUDA_VISIBLE_DEVICES)

        self.model_name = model_name
        self.system_prompt = system_prompt or ""
        self.max_model_len = max_model_len

        # Initialize vLLM engine (uses CUDA_VISIBLE_DEVICES set by setup_gpu)
        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype="bfloat16",
        )
        self._initialized = True
        logger.info("OVRActor (vLLM) initialized on GPU %s: %s", self.gpu_ids, model_name)

    def analyze(
        self,
        image_b64: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs,
    ) -> str:
        """Analyze an image and answer the question using vLLM."""
        from PIL import Image
        from geo_edit.tool_definitions.agents._vllm_compat import SamplingParams

        # Extract question from kwargs
        question = kwargs.get("question", "")

        # Decode base64 image
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Build Qwen2-VL style messages with base64 data URL for vLLM
        # vLLM expects image_url format, not raw PIL image
        image_data_url = f"data:image/jpeg;base64,{image_b64}"

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {"type": "text", "text": question},
                ],
            }
        )

        # Set sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Generate with vLLM
        outputs = self.llm.chat(messages, sampling_params=sampling_params)
        generated_text = outputs[0].outputs[0].text

        return self._parse_output(generated_text)

    def _parse_output(self, text: str) -> str:
        """Parse output."""
        try:
            payload = json.loads(text)
            if payload.get("error"):
                return f"Error: {payload['error']}"
            for key in ["analysis", "text", "result"]:
                if key in payload:
                    return payload[key]
        except json.JSONDecodeError:
            pass
        return text

    def health_check(self) -> dict:
        """Return health status of the actor."""
        return {"model": self.model_name, "initialized": self._initialized}


ACTOR_CLASS = OVRActor

DECLARATION = {
    "name": "ovr",
    "description": "Use an RL-enhanced visual reasoning VLM to analyze math problem images with step-by-step reasoning capabilities.",
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
