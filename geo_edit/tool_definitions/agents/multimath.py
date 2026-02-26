"""Multimath VLM Agent Tool."""

import base64
import json
from io import BytesIO
from typing import Optional

from geo_edit.environment.tool_agents.actor import BaseToolModelActor
from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)

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


class MultiMathActor(BaseToolModelActor):
    """MultiMath VLM Actor using LLaVA model builder."""

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

        # Use LLaVA model builder for proper loading
        from geo_edit.models.multimath.model.builder import load_pretrained_model
        from geo_edit.models.multimath.mm_utils import get_model_name_from_path

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
        logger.info("MultiMathActor initialized on GPU %s: %s", self.gpu_ids, model_name)

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

        from geo_edit.models.multimath.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from geo_edit.models.multimath.conversation import conv_templates
        from geo_edit.models.multimath.mm_utils import tokenizer_image_token, process_images

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
                image_sizes=[image.size],
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                max_new_tokens=max_tokens,
                use_cache=True,
            )

        # Decode output
        generated_text = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        ).strip()

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


ACTOR_CLASS = MultiMathActor

DECLARATION = {
    "name": "multimath",
    "description": "Use a multimodal math-vision parsing tool to process math-problem images, returning either LaTeX OCR (for expression-only images) or a detailed English description of diagrams/plots/charts with all visible text, symbols, labels, and markers for downstream reasoning.  You should input the index of the image to analyze and a question about it, the question should contain clear instructions and necessary information for the analysis.",
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
