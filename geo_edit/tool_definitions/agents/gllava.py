"""GLLaVA VLM Agent Tool."""

import base64
import json
from io import BytesIO
from typing import Optional

from geo_edit.environment.tool_agents.actor import BaseToolModelActor
from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)

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


class GLLaVAActor(BaseToolModelActor):
    """GLLaVA VLM Actor using transformers inference."""

    def __init__(
        self,
        model_name: str,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.8,
        system_prompt: Optional[str] = None,
    ):
        import torch
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        self.setup_gpu()  # Configure GPU based on Ray assignment

        self.model_name = model_name
        self.system_prompt = system_prompt or ""
        self.max_model_len = max_model_len

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=self.device_map,
            trust_remote_code=True,
        )
        self._initialized = True
        logger.info("GLLaVAActor initialized on GPU %s: %s", self.gpu_ids, model_name)

    def analyze(
        self,
        image_b64: str,
        question: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Analyze an image and answer the question."""
        import torch
        from PIL import Image

        # Decode base64 image
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Build prompt with system prompt
        if self.system_prompt:
            prompt = f"USER: <image>\n{self.system_prompt}\n{question}\nASSISTANT:"
        else:
            prompt = f"USER: <image>\n{question}\nASSISTANT:"

        # Process inputs
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            generate_kwargs = {
                "max_new_tokens": max_tokens,
                "do_sample": temperature > 0,
            }
            if temperature > 0:
                generate_kwargs["temperature"] = temperature
            output_ids = self.model.generate(**inputs, **generate_kwargs)

        # Decode output (skip input tokens)
        input_len = inputs["input_ids"].shape[1]
        generated_text = self.processor.decode(
            output_ids[0][input_len:], skip_special_tokens=True
        )

        return self._parse_output(generated_text)

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


ACTOR_CLASS = GLLaVAActor

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
