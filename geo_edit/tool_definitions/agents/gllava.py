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
    """GLLaVA VLM Actor using LLaVA model builder."""

    def __init__(
        self,
        model_name: str,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.8,
        system_prompt: Optional[str] = None,
    ):
        self.setup_gpu()  # Configure GPU based on Ray assignment

        self.model_name = model_name
        self.system_prompt = system_prompt or ""
        self.max_model_len = max_model_len

        # Use G-LLaVA model builder for proper loading
        from geo_edit.models.gllava.model.builder import load_pretrained_model
        from geo_edit.models.gllava.mm_utils import get_model_name_from_path

        model_path = model_name
        model_base = None
        llava_model_name = get_model_name_from_path(model_path)

        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path,
            model_base,
            llava_model_name,
            device_map=self.device_map,
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

        from geo_edit.models.gllava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from geo_edit.models.gllava.conversation import conv_templates
        from geo_edit.models.gllava.mm_utils import tokenizer_image_token, process_images

        # Decode base64 image
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Process image
        image_tensor = process_images([image], self.image_processor, self.model.config)
        if isinstance(image_tensor, list):
            image_tensor = [img.to(self.model.device, dtype=torch.float16) for img in image_tensor]
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

        # Build prompt with conversation template
        conv = conv_templates["llava_v1"].copy()
        if self.system_prompt:
            full_question = f"{self.system_prompt}\n{question}"
        else:
            full_question = question

        # Add image token to the question
        inp = DEFAULT_IMAGE_TOKEN + "\n" + full_question
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Tokenize
        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_ids = input_ids.unsqueeze(0).to(self.model.device)

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                max_new_tokens=max_tokens,
                use_cache=False,
            )

        # Decode output
        vocab_size = len(self.tokenizer)
        output_ids_list = output_ids[0].tolist()
        logger.info("Generated %d tokens, vocab_size=%d", len(output_ids_list), vocab_size)
        logger.info("Token ID range: min=%d, max=%d", min(output_ids_list), max(output_ids_list))
        invalid_ids = [tid for tid in output_ids_list if tid < 0 or tid >= vocab_size]
        if invalid_ids:
            logger.warning("Found %d invalid token ids (out of vocab range): %s", len(invalid_ids), invalid_ids[:10])
        generated_text = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        ).strip()
        logger.info("Generated text: %s", generated_text)

        # Remove the prompt from output if present
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()

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
