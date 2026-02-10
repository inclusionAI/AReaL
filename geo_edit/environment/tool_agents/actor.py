"""Tool Model Ray Actor - VLM with GPU-resident weights."""

import json
from typing import Optional

import ray

from geo_edit.prompts import DEFAULT_TOOL_AGENT_PROMPT
from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)


@ray.remote(num_gpus=1)
class ToolModelActor:
    """VLM Tool Agent - model weights stay resident in GPU memory."""

    def __init__(
        self,
        model_name: str,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.8,
        system_prompt: Optional[str] = None,
    ):
        from vllm import LLM

        self.model_name = model_name
        self.system_prompt = system_prompt or DEFAULT_TOOL_AGENT_PROMPT
        self.llm = LLM(
            model=model_name,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )
        self._initialized = True
        logger.info("ToolModelActor initialized: %s", model_name)

    def analyze(
        self,
        image_b64: str,
        question: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Analyze an image and answer the question."""
        from vllm import SamplingParams

        messages = self._build_messages(image_b64, question)
        params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        outputs = self.llm.chat(messages, sampling_params=params)

        if outputs and outputs[0].outputs:
            return self._parse_output(outputs[0].outputs[0].text)
        return '{"error": "No output generated"}'

    def _build_messages(self, image_b64: str, question: str) -> list:
        """Build chat messages for the model."""
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ],
            },
        ]

    def _parse_output(self, text: str) -> str:
        """Parse output (following ToolAgent._parse_response logic)."""
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