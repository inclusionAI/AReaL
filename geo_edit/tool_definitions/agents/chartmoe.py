"""ChartMoE VLM Agent Tool.

Based on official implementation: https://github.com/DataArcTech/ChartMoE
"""

import base64
import json
from io import BytesIO
from typing import Optional

from geo_edit.environment.tool_agents.actor import BaseToolModelActor
from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)

# System prompt for this agent (empty string means no system prompt)
SYSTEM_PROMPT = ""

# Model configuration
agent_config = {
    "model_name_or_path": "/storage/openpsi/models/chartmoe",
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.8,
    "temperature": 0.0,
    "max_tokens": 4096,
    "num_gpus": 1,
}

# ChartMoE prompt template (from official implementation)
CHARTMOE_PROMPT_TEMPLATE = "[UNUSED_TOKEN_146]user\n{}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n"


class ChartMoEActor(BaseToolModelActor):
    """ChartMoE VLM Actor following official implementation.

    Based on: https://github.com/DataArcTech/ChartMoE/blob/master/chartmoe/generation_utils.py
    """

    def __init__(
        self,
        model_name: str,
        system_prompt: Optional[str] = None,
    ):
        """Initialize ChartMoE actor.

        Args:
            model_name: Path to ChartMoE model.
            system_prompt: Optional system prompt for the model.
        """
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.setup_gpu()  # Configure GPU based on Ray assignment

        self.model_name = model_name
        self.system_prompt = system_prompt or ""

        # Load tokenizer and model (following official implementation)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
        ).half().cuda().eval()

        # Set tokenizer on model (required by official implementation)
        self.model.tokenizer = self.tokenizer

        # Patch vision_tower to fix image size mismatch (490x490 vs 336x336)
        self._patch_vision_tower()

        self._initialized = True
        logger.info("ChartMoEActor initialized on GPU %s: %s", self.gpu_ids, model_name)

    def _patch_vision_tower(self):
        """Patch vision_tower.forward to add interpolate_pos_encoding=True.

        Fixes: ValueError: Input image size (490*490) doesn't match model (336*336).
        The patch enables position encoding interpolation for variable-size images.
        """
        # Navigate to vision_tower: model.vit.vision_tower
        if hasattr(self.model, 'vit') and hasattr(self.model.vit, 'vision_tower'):
            vision_tower = self.model.vit.vision_tower
            original_forward = vision_tower.forward

            def patched_forward(pixel_values, output_hidden_states=False, return_dict=True, **kwargs):
                # Always enable interpolate_pos_encoding for variable-size images
                return original_forward(
                    pixel_values,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    interpolate_pos_encoding=True,
                    **kwargs
                )

            vision_tower.forward = patched_forward
            logger.info("Patched vision_tower.forward with interpolate_pos_encoding=True")

    def analyze(
        self,
        image_b64: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs,
    ) -> str:
        """Analyze an image and answer the question.

        Follows official ChartMoE_Robot.chat() implementation.
        """
        import torch
        from PIL import Image

        # Extract question from kwargs
        question = kwargs.get("question", "")

        # Decode base64 image to PIL Image
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Build prompt with system prompt if provided
        if self.system_prompt:
            full_question = f"{self.system_prompt}\n{question}"
        else:
            full_question = question

        # Format with ChartMoE template
        formatted_prompt = CHARTMOE_PROMPT_TEMPLATE.format(full_question)

        # Build embeddings following official implementation
        # Order: [BOS] -> [IMAGE] -> [TEXT]
        embeds = []
        im_mask = []

        # 1. Encode empty text with BOS token (before image at position 0)
        text_embeds = self.model.encode_text("", add_special_tokens=True)
        embeds.append(text_embeds)
        im_mask.append(torch.zeros(text_embeds.shape[:2]).cuda())

        # 2. Encode image
        image_tensor = self.model.vis_processor(image).unsqueeze(0).half().cuda()
        image_embeds = self.model.encode_img(image_tensor)
        embeds.append(image_embeds)
        im_mask.append(torch.ones(image_embeds.shape[:2]).cuda())

        # 3. Encode prompt text (after image)
        text_embeds = self.model.encode_text(formatted_prompt, add_special_tokens=False)
        embeds.append(text_embeds)
        im_mask.append(torch.zeros(text_embeds.shape[:2]).cuda())

        # 3. Concatenate embeddings
        embeds = torch.cat(embeds, dim=1)
        im_mask = torch.cat(im_mask, dim=1).bool()

        # 4. Generate with EOS tokens
        eos_token_id = [
            self.tokenizer.convert_tokens_to_ids(["[UNUSED_TOKEN_145]"])[0],
            self.tokenizer.eos_token_id,
        ]

        with torch.cuda.amp.autocast():
            outputs = self.model.generate(
                inputs_embeds=embeds,
                im_mask=im_mask,
                temperature=temperature if temperature > 0 else 1.0,
                max_new_tokens=max_tokens,
                num_beams=1,
                do_sample=temperature > 0,
                repetition_penalty=1.0,
                eos_token_id=eos_token_id,
            )

        # 5. Decode output
        output_token = outputs[0]
        if output_token[0] == 0 or output_token[0] == 1:
            output_token = output_token[1:]
        output_text = self.tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split("[UNUSED_TOKEN_145]")[0].strip()

        return self._parse_output(output_text)

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


ACTOR_CLASS = ChartMoEActor

# Legacy single declaration (kept for backward compatibility)
DECLARATION = {
    "name": "chartmoe",
    "description": "Use a chart-analysis VLM tool to read charts from images and return structured outputs (e.g., table/JSON) plus chart-grounded analysis. It can extract chart elements, visible values/relationships, trends, and key observations for downstream reasoning. You should input the index of the image to analyze and a question about it, the question should contain clear instructions and necessary information for the analysis.",
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

# Fixed prompts for different chart analysis modes
DATA_EXTRACT_PROMPT = "Extract all data from this chart. Output the data in a structured format (table or JSON). Include all visible values, labels, categories, and data points."

TREND_ANALYSIS_PROMPT = "Analyze this chart and describe the key trends, patterns, and observations. Focus on: overall direction, notable peaks/valleys, comparisons between categories, and any significant insights."

# Multi-tool declarations - split by analysis mode
DECLARATIONS = {
    "chart_data_extract": {
        "name": "chart_data_extract",
        "description": "Chart data extraction tool. Extracts all numerical data and labels from charts into structured format (table/JSON). Captures data points, axis values, legend items, and category labels. Best for: getting exact values from bar charts, line graphs, pie charts, reading specific data points.",
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
        "fixed_prompt": DATA_EXTRACT_PROMPT,
        "return_type": "text"
    },
    "chart_trend_analysis": {
        "name": "chart_trend_analysis",
        "description": "Chart trend analysis tool. Analyzes charts to identify trends, patterns, comparisons, and key insights. Describes overall direction, notable changes, category comparisons, and significant observations. Best for: understanding chart meaning, identifying trends, comparing values, summarizing chart insights.",
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
        "fixed_prompt": TREND_ANALYSIS_PROMPT,
        "return_type": "text"
    }
}
