"""SAM3 (Segment Anything Model 3) Tool Agent.

Replaces SAM2 with SAM3.1, adding open-vocabulary text-prompted segmentation,
exemplar-based segmentation, concept counting, and presence checking.
Uses vendored sam3 package with native Sam3Processor API.
"""

import base64
import json
import re
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from geo_edit.environment.tool_agents.actor import BaseToolModelActor
from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)

# SAM3 doesn't need a system prompt (not a language model)
SYSTEM_PROMPT = ""

# Model configuration
agent_config = {
    "model_name_or_path": "/storage/openpsi/models/sam3.1/sam3.1_multiplex.pt",
    "num_gpus": 1,
}

# Constants
SCORE_THRESHOLD = 0.25
MAX_PROPOSALS = 20
NORMALIZED_SIZE = 1000  # Bounding box coordinate normalization factor
PRESENCE_THRESHOLD = 0.1  # Lower threshold for presence_check (more sensitive)


def _state_to_proposals(
    state: Dict[str, Any],
    max_proposals: int = MAX_PROPOSALS,
) -> List[Dict[str, Any]]:
    """Convert Sam3Processor state output to proposal format.

    Args:
        state: Sam3Processor state dict containing 'masks', 'boxes', 'scores'.
        max_proposals: Maximum number of proposals to return.

    Returns:
        List of proposal dictionaries sorted by score.
    """
    import torch

    masks = state.get("masks")
    boxes = state.get("boxes")
    scores = state.get("scores")

    if masks is None or boxes is None or scores is None:
        return []

    # Convert tensors to numpy
    if isinstance(scores, torch.Tensor):
        scores_np = scores.cpu().numpy()
    else:
        scores_np = np.asarray(scores)

    if isinstance(boxes, torch.Tensor):
        boxes_np = boxes.cpu().numpy()
    else:
        boxes_np = np.asarray(boxes)

    if isinstance(masks, torch.Tensor):
        masks_np = masks.cpu().numpy()
    else:
        masks_np = np.asarray(masks)

    proposals = []
    for i in range(len(scores_np)):
        score = float(scores_np[i])

        # boxes are in [x1, y1, x2, y2] pixel coords from _forward_grounding
        box = boxes_np[i]
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

        # Get mask for area/centroid
        mask = masks_np[i]
        if mask.ndim > 2:
            mask = mask.squeeze()
        if mask.ndim == 2 and mask.any():
            area = int(mask.sum())
            mask_coords = np.where(mask)
            cy = float(np.mean(mask_coords[0]))
            cx = float(np.mean(mask_coords[1]))
        else:
            area = (x2 - x1) * (y2 - y1)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

        proposals.append({
            "score": round(score, 2),
            "bbox_xyxy": [x1, y1, x2, y2],
            "area": area,
            "centroid": [round(cx, 1), round(cy, 1)],
        })

    proposals.sort(key=lambda x: x["score"], reverse=True)
    return proposals[:max_proposals]


class SAM3Actor(BaseToolModelActor):
    """SAM3 Segmentation Actor using vendored sam3 package."""

    def __init__(self, model_name: str):
        """Initialize SAM3 actor.

        Args:
            model_name: Path to SAM3 .pt checkpoint.
        """
        import torch

        self.setup_gpu()  # Configure GPU based on Ray assignment

        self.model_name = model_name

        logger.info("Loading SAM3 model: %s", self.model_name)

        from geo_edit.models.sam3 import build_sam3_image_model
        from geo_edit.models.sam3.model.sam3_image_processor import Sam3Processor

        self._model = build_sam3_image_model(
            checkpoint_path=model_name,
            device=self.device,
            eval_mode=True,
            load_from_HF=False,
        )
        self._processor = Sam3Processor(
            self._model,
            device=self.device,
            confidence_threshold=SCORE_THRESHOLD,
        )
        self._initialized = True

        logger.info("SAM3Actor initialized on GPU %s: %s", self.gpu_ids, model_name)

    def analyze(
        self,
        image_b64: str,
        **kwargs,
    ) -> str:
        """Run SAM3 segmentation/detection and return JSON results.

        Args:
            image_b64: Base64-encoded image string.
            **kwargs: Tool-specific parameters including 'mode', 'bounding_box', 'text_prompt'.

        Returns:
            JSON string with results appropriate to the mode.
        """
        import torch
        from PIL import Image

        # Decode image
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        W, H = image.size  # PIL: (W, H)

        mode = kwargs.get("mode", "auto")

        try:
            with torch.inference_mode():
                if mode == "text_segment":
                    text_prompt = kwargs.get("text_prompt", "")
                    return self._text_segment(image, text_prompt, H, W)

                elif mode == "exemplar_segment":
                    bbox_str = kwargs.get("bounding_box", kwargs.get("question", ""))
                    bbox = self._parse_bbox(bbox_str, W, H)
                    return self._exemplar_segment(image, bbox, H, W)

                elif mode == "concept_count":
                    text_prompt = kwargs.get("text_prompt", "")
                    return self._concept_count(image, text_prompt, H, W)

                elif mode == "presence_check":
                    text_prompt = kwargs.get("text_prompt", "")
                    return self._presence_check(image, text_prompt, H, W)

                elif mode == "bbox":
                    bbox_str = kwargs.get("bounding_box", kwargs.get("question", ""))
                    bbox = self._parse_bbox(bbox_str, W, H)
                    return self._bbox_segment(image, bbox, H, W)

                else:  # "auto"
                    return self._auto_segment(image, H, W)

        except Exception as e:
            logger.error("SAM3 %s failed: %s", mode, e)
            return json.dumps({"error": str(e), "image_size": [H, W], "proposals": []})

    def _parse_bbox(self, question: str, width: int, height: int) -> Optional[List[float]]:
        """Parse bounding box from question string.

        Args:
            question: May contain \\boxed{x1,y1,x2,y2} in normalized 0-1000 coords.
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            [cx, cy, w, h] in normalized 0-1 coords for Sam3Processor, or None.
        """
        if not question or not question.strip():
            return None

        # Parse \boxed{x1,y1,x2,y2} format
        match = re.search(r'\\boxed\{(\d+),(\d+),(\d+),(\d+)\}', question)
        if not match:
            # Also try without backslash
            match = re.search(r'boxed\{(\d+),(\d+),(\d+),(\d+)\}', question)
        if not match:
            return None

        # Convert from normalized (0-1000) xyxy to normalized (0-1) cxcywh
        coords = [int(x) for x in match.groups()]
        x1, y1, x2, y2 = coords
        cx = (x1 + x2) / 2 / NORMALIZED_SIZE
        cy = (y1 + y2) / 2 / NORMALIZED_SIZE
        w = (x2 - x1) / NORMALIZED_SIZE
        h = (y2 - y1) / NORMALIZED_SIZE

        return [cx, cy, w, h]

    def _auto_segment(self, image, H: int, W: int) -> str:
        """Automatic full-image segmentation detecting all objects."""
        state = self._processor.set_image(image)
        state = self._processor.set_text_prompt(prompt="objects", state=state)

        proposals = _state_to_proposals(state)
        return json.dumps({"image_size": [H, W], "proposals": proposals})

    def _bbox_segment(self, image, bbox: Optional[List[float]], H: int, W: int) -> str:
        """Region-constrained segmentation within a bounding box."""
        if bbox is None:
            return self._auto_segment(image, H, W)

        state = self._processor.set_image(image)
        state = self._processor.add_geometric_prompt(
            box=bbox, label=True, state=state,
        )

        proposals = _state_to_proposals(state)
        return json.dumps({"image_size": [H, W], "proposals": proposals})

    def _text_segment(self, image, text_prompt: str, H: int, W: int) -> str:
        """Open-vocabulary text-prompted segmentation."""
        state = self._processor.set_image(image)
        state = self._processor.set_text_prompt(prompt=text_prompt, state=state)

        proposals = _state_to_proposals(state)
        return json.dumps({
            "image_size": [H, W],
            "query": text_prompt,
            "proposals": proposals,
        })

    def _exemplar_segment(self, image, bbox: Optional[List[float]], H: int, W: int) -> str:
        """Visual exemplar-based segmentation using a bounding box as positive prompt."""
        if bbox is None:
            return json.dumps({
                "error": "No bounding box provided for exemplar_segment",
                "image_size": [H, W],
                "proposals": [],
            })

        state = self._processor.set_image(image)
        state = self._processor.add_geometric_prompt(
            box=bbox, label=True, state=state,
        )

        proposals = _state_to_proposals(state)
        return json.dumps({
            "image_size": [H, W],
            "exemplar_bbox": bbox,
            "proposals": proposals,
        })

    def _concept_count(self, image, text_prompt: str, H: int, W: int) -> str:
        """Count objects matching a text description."""
        state = self._processor.set_image(image)
        state = self._processor.set_text_prompt(prompt=text_prompt, state=state)

        proposals = _state_to_proposals(state)
        # Convert proposals to instance format for counting
        instances = [
            {
                "bbox_xyxy": p["bbox_xyxy"],
                "score": p["score"],
            }
            for p in proposals
        ]

        return json.dumps({
            "image_size": [H, W],
            "query": text_prompt,
            "count": len(instances),
            "instances": instances,
        })

    def _presence_check(self, image, text_prompt: str, H: int, W: int) -> str:
        """Quick check whether a concept is present in the image."""
        import torch

        # Use a lower threshold for more sensitive detection
        old_threshold = self._processor.confidence_threshold
        self._processor.confidence_threshold = PRESENCE_THRESHOLD

        state = self._processor.set_image(image)
        state = self._processor.set_text_prompt(prompt=text_prompt, state=state)

        # Restore threshold
        self._processor.confidence_threshold = old_threshold

        scores = state.get("scores")
        if scores is not None and len(scores) > 0:
            if isinstance(scores, torch.Tensor):
                all_scores = scores.cpu().numpy()
            else:
                all_scores = np.asarray(scores)
            confidence = float(all_scores.max())
            # Count only those above the normal threshold
            count = int((all_scores >= SCORE_THRESHOLD).sum())
        else:
            confidence = 0.0
            count = 0

        present = confidence >= SCORE_THRESHOLD

        return json.dumps({
            "image_size": [H, W],
            "query": text_prompt,
            "present": present,
            "confidence": round(confidence, 3),
            "count": count,
        })

    def health_check(self) -> dict:
        """Return health status of the actor."""
        return {
            "model": self.model_name,
            "initialized": self._initialized,
        }


ACTOR_CLASS = SAM3Actor
RETURN_TYPE = "text"

# Multi-tool declarations - 6 tools covering all SAM3 capabilities
DECLARATIONS = {
    "auto_segment": {
        "name": "auto_segment",
        "description": "Automatic image segmentation tool. Detects and segments ALL objects in an image without any prior knowledge. Returns JSON with mask proposals including bounding boxes, confidence scores, areas, and centroids. Best for: discovering unknown objects, general scene understanding, counting objects.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_index": {
                    "type": "integer",
                    "description": "The index of the image to segment (e.g., 0 for Observation 0)."
                }
            },
            "required": ["image_index"]
        },
        "fixed_mode": "auto",
        "return_type": "text"
    },
    "bbox_segment": {
        "name": "bbox_segment",
        "description": "Region-constrained segmentation tool. Performs precise segmentation within a specified bounding box region. Returns refined mask proposals for objects in the target area. Best for: segmenting specific objects, refining detection results, focused analysis.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_index": {
                    "type": "integer",
                    "description": "The index of the image to segment (e.g., 0 for Observation 0)."
                },
                "bounding_box": {
                    "type": "string",
                    "description": "Bounding box coordinates in format '\\boxed{x1,y1,x2,y2}' where values are 0-1000 normalized coordinates."
                }
            },
            "required": ["image_index", "bounding_box"]
        },
        "fixed_mode": "bbox",
        "return_type": "text"
    },
    "text_segment": {
        "name": "text_segment",
        "description": "Text-prompted segmentation tool. Segments objects described by a natural language text prompt using SAM 3.1 open-vocabulary understanding (270K+ concepts). Returns JSON with mask proposals for matching objects. Best for: finding specific objects by description, semantic segmentation, targeted object isolation. Example prompts: 'player in white jersey', 'red car on the left', 'all trees'.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_index": {
                    "type": "integer",
                    "description": "The index of the image to segment (e.g., 0 for Observation 0)."
                },
                "text_prompt": {
                    "type": "string",
                    "description": "Natural language description of the object(s) to segment. Be specific for best results."
                }
            },
            "required": ["image_index", "text_prompt"]
        },
        "fixed_mode": "text_segment",
        "return_type": "text"
    },
    "exemplar_segment": {
        "name": "exemplar_segment",
        "description": "Visual exemplar-based segmentation tool. Given a bounding box as a visual exemplar, finds and segments all similar objects in the image. Uses SAM 3.1 visual matching to discover objects sharing visual characteristics with the exemplar region. Best for: 'find more like this', repeating pattern detection, similar object discovery.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_index": {
                    "type": "integer",
                    "description": "The index of the image to segment (e.g., 0 for Observation 0)."
                },
                "bounding_box": {
                    "type": "string",
                    "description": "Bounding box of the exemplar region in format '\\boxed{x1,y1,x2,y2}' where values are 0-1000 normalized coordinates. Objects similar to this region will be found."
                }
            },
            "required": ["image_index", "bounding_box"]
        },
        "fixed_mode": "exemplar_segment",
        "return_type": "text"
    },
    "concept_count": {
        "name": "concept_count",
        "description": "Object counting tool. Counts objects matching a text description and returns their locations. Uses SAM 3.1 text-prompted detection. Returns count and bounding boxes of all matching instances. Best for: 'how many X are there', inventory counting, quantity verification.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_index": {
                    "type": "integer",
                    "description": "The index of the image to analyze (e.g., 0 for Observation 0)."
                },
                "text_prompt": {
                    "type": "string",
                    "description": "Natural language description of the objects to count. Example: 'people', 'red cars', 'windows on the building'."
                }
            },
            "required": ["image_index", "text_prompt"]
        },
        "fixed_mode": "concept_count",
        "return_type": "text"
    },
    "presence_check": {
        "name": "presence_check",
        "description": "Quick concept presence verification tool. Rapidly checks whether a described concept exists in the image without full segmentation. Returns a boolean presence flag, confidence score, and object count. Uses SAM 3.1 presence token for efficient verification. Best for: yes/no queries, pre-filtering before detailed analysis, spatial reasoning checks.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_index": {
                    "type": "integer",
                    "description": "The index of the image to check (e.g., 0 for Observation 0)."
                },
                "text_prompt": {
                    "type": "string",
                    "description": "Natural language description of the concept to check for. Example: 'dog', 'stop sign', 'person wearing a hat'."
                }
            },
            "required": ["image_index", "text_prompt"]
        },
        "fixed_mode": "presence_check",
        "return_type": "text"
    },
}
