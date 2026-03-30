"""ThinkMorph Visual Reasoning Tool Agent - Auxiliary reasoning image generation.

Uses ThinkMorph (ByteDance) interleaved text-image generation with visual thinking
to produce auxiliary reasoning images (geometric constructions, route drawings,
spatial annotations) that aid downstream reasoning.
"""

import base64
from io import BytesIO
from typing import Optional

from PIL import Image

from geo_edit.environment.tool_agents.actor import BaseToolModelActor
from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)

# ThinkMorph handles its own system prompt internally
SYSTEM_PROMPT = ""

# Model configuration
agent_config = {
    "model_name_or_path": "/storage/openpsi/models/ThinkMorph-7B",
    "num_gpus": 1,
}


class ThinkMorphActor(BaseToolModelActor):
    """ThinkMorph Actor for visual reasoning image generation.

    Uses ThinkMorphInference with HIGH_QUALITY_CONFIG (100 diffusion steps,
    cfg_text_scale=6.0, cfg_img_scale=2.5) for maximum image quality.
    Generates 1024x1024 images.
    """

    def __init__(self, model_name: str):
        """Initialize ThinkMorph actor.

        Args:
            model_name: Path to ThinkMorph-7B model directory.
        """
        from geo_edit.models.thinkmorph_vllm import ThinkMorphInference

        self.setup_gpu()  # Configure GPU based on Ray assignment

        self.model_name = model_name

        # Use HIGH_QUALITY_CONFIG parameters for best diffusion quality
        logger.info("Loading ThinkMorph model: %s", model_name)
        self.model = ThinkMorphInference(
            model_path=model_name,
            device=0,  # After setup_gpu(), cuda:0 = the Ray-assigned GPU
            num_timesteps=100,
            cfg_text_scale=6.0,
            cfg_img_scale=2.5,
            text_temperature=0.3,
            max_think_tokens=4096,
            max_rounds=1,  # Tool agent only needs one round of image generation
        )

        self._initialized = True
        logger.info("ThinkMorphActor initialized on GPU %s: %s", self.gpu_ids, model_name)

    def analyze(
        self,
        image_b64: str,
        **kwargs,
    ) -> str:
        """Generate an auxiliary visual reasoning image.

        Args:
            image_b64: Base64-encoded input image.
            **kwargs: Must include 'question' describing what visual reasoning to perform.

        Returns:
            Base64-encoded PNG string of the generated reasoning image,
            or an error message string if no image was generated.
        """
        # Decode input image
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Extract question
        question = kwargs.get("question", "")
        if not question:
            return "Error: No question provided for visual reasoning."

        try:
            # Run inference with visual thinking enabled
            outputs = self.model.infer_single(
                image=image,
                text=question,
                think=True,
                understanding_output=False,
            )

            # Extract the first generated image from outputs
            generated_image = None
            for item in outputs:
                if isinstance(item, Image.Image):
                    generated_image = item
                    break

            if generated_image is None:
                return "Error: ThinkMorph did not generate a reasoning image for this query."

            # Encode generated image to PNG base64
            buffer = BytesIO()
            generated_image.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

        except Exception as e:
            logger.error("ThinkMorph inference failed: %s", e)
            return f"Error: ThinkMorph inference failed: {e}"

    def health_check(self) -> dict:
        """Return health status of the actor."""
        return {"model": self.model_name, "initialized": self._initialized}


ACTOR_CLASS = ThinkMorphActor

# Multi-tool declarations
DECLARATIONS = {
    "visual_reasoning": {
        "name": "visual_reasoning",
        "description": (
            "Generate an auxiliary visual reasoning image to help solve the problem. "
            "Given an input image and a question, this tool uses visual thinking to "
            "produce an annotated or constructed image (e.g., geometric constructions, "
            "route drawings, spatial annotations) that aids downstream reasoning."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "image_index": {
                    "type": "integer",
                    "description": "The index of the image to reason about (e.g., 0 for Observation 0).",
                },
                "question": {
                    "type": "string",
                    "description": "What visual reasoning to perform on the image.",
                },
            },
            "required": ["image_index", "question"],
        },
        "return_type": "image",
    },
}
