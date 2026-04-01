"""SAM3 (Segment Anything Model 3) Tool Agent.

Replaces SAM2 with SAM3.1, adding open-vocabulary text-prompted segmentation,
exemplar-based segmentation, concept counting, and presence checking.
Uses HuggingFace transformers API (AutoModel + AutoProcessor).
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
    "model_name_or_path": "/storage/openpsi/models/sam3.1",
    "num_gpus": 1,
}

# Constants
SCORE_THRESHOLD = 0.25
MAX_PROPOSALS = 20
NORMALIZED_SIZE = 1000  # Bounding box coordinate normalization factor
PRESENCE_THRESHOLD = 0.1  # Lower threshold for presence_check (more sensitive)


def masks_to_proposals(
    masks: np.ndarray,
    scores: np.ndarray,
    image_size: Tuple[int, int],
    score_threshold: float = SCORE_THRESHOLD,
    max_proposals: int = MAX_PROPOSALS,
) -> List[Dict[str, Any]]:
    """Convert binary masks to proposal format.

    Args:
        masks: Binary masks array with shape (N, H, W).
        scores: Confidence scores array with shape (N,).
        image_size: Tuple of (H, W) image dimensions.
        score_threshold: Minimum score threshold for proposals.
        max_proposals: Maximum number of proposals to return.

    Returns:
        List of proposal dictionaries sorted by score, limited to max_proposals.
    """
    H, W = image_size
    proposals = []

    for mask, score in zip(masks, scores):
        if score < score_threshold:
            continue

        # Ensure mask is 2D
        if mask.ndim > 2:
            mask = mask.squeeze()
        if mask.ndim != 2:
            continue

        # Find bounding box from mask
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any() or not cols.any():
            continue

        y_indices = np.where(rows)[0]
        x_indices = np.where(cols)[0]
        y1, y2 = int(y_indices[0]), int(y_indices[-1])
        x1, x2 = int(x_indices[0]), int(x_indices[-1])

        # Calculate area and centroid
        area = int(mask.sum())
        mask_coords = np.where(mask)
        cy = float(np.mean(mask_coords[0]))
        cx = float(np.mean(mask_coords[1]))

        proposals.append({
            "score": round(float(score), 2),
            "bbox_xyxy": [x1, y1, x2, y2],
            "area": area,
            "centroid": [round(cx, 1), round(cy, 1)]
        })

    # Sort by score descending, limit to max_proposals
    proposals.sort(key=lambda x: x["score"], reverse=True)
    return proposals[:max_proposals]


def instance_results_to_proposals(
    results: Dict[str, Any],
    max_proposals: int = MAX_PROPOSALS,
) -> List[Dict[str, Any]]:
    """Convert SAM3 post_process_instance_segmentation output to proposal format.

    Args:
        results: Single-image result dict from post_process_instance_segmentation,
                 containing 'scores', 'boxes', and 'masks'.
        max_proposals: Maximum number of proposals to return.

    Returns:
        List of proposal dictionaries sorted by score.
    """
    scores = results.get("scores", [])
    boxes = results.get("boxes", [])
    masks = results.get("masks", [])

    proposals = []
    for i in range(len(scores)):
        score = float(scores[i])
        box = boxes[i]
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

        # Calculate area and centroid from mask if available
        if i < len(masks):
            mask = np.asarray(masks[i], dtype=np.bool_)
            if mask.ndim > 2:
                mask = mask.squeeze()
            area = int(mask.sum()) if mask.ndim == 2 else (x2 - x1) * (y2 - y1)
            if mask.ndim == 2 and mask.any():
                mask_coords = np.where(mask)
                cy = float(np.mean(mask_coords[0]))
                cx = float(np.mean(mask_coords[1]))
            else:
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
        else:
            area = (x2 - x1) * (y2 - y1)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

        proposals.append({
            "score": round(score, 2),
            "bbox_xyxy": [x1, y1, x2, y2],
            "area": area,
            "centroid": [round(cx, 1), round(cy, 1)]
        })

    proposals.sort(key=lambda x: x["score"], reverse=True)
    return proposals[:max_proposals]


class SAM3Actor(BaseToolModelActor):
    """SAM3 Segmentation Actor using HuggingFace transformers."""

    def __init__(self, model_name: str):
        """Initialize SAM3 actor.

        Args:
            model_name: Path to SAM3 model.
        """
        import torch
        from transformers import AutoModel, AutoProcessor

        self.setup_gpu()  # Configure GPU based on Ray assignment

        self.model_name = model_name

        logger.info("Loading SAM3 model: %s", self.model_name)

        self._model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
        ).to(self.device)
        self._processor = AutoProcessor.from_pretrained(self.model_name)
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

    def _parse_bbox(self, question: str, width: int, height: int) -> Optional[List[int]]:
        """Parse bounding box from question string.

        Args:
            question: May contain \\boxed{x1,y1,x2,y2} in normalized 0-1000 coords.
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            [x1, y1, x2, y2] in pixel coordinates, or None for auto mode.
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

        # Convert from normalized (0-1000) to pixel coordinates
        coords = [int(x) for x in match.groups()]
        x1 = int(coords[0] * width / NORMALIZED_SIZE)
        y1 = int(coords[1] * height / NORMALIZED_SIZE)
        x2 = int(coords[2] * width / NORMALIZED_SIZE)
        y2 = int(coords[3] * height / NORMALIZED_SIZE)

        return [x1, y1, x2, y2]

    def _auto_segment(self, image, H: int, W: int) -> str:
        """Automatic full-image segmentation detecting all objects."""
        import torch

        # Use a generic prompt for automatic segmentation
        inputs = self._processor(
            images=image,
            text="objects",
            return_tensors="pt",
        ).to(self.device, dtype=torch.float16)

        outputs = self._model(**inputs)
        results = self._processor.post_process_instance_segmentation(
            outputs,
            target_sizes=[(H, W)],
            threshold=SCORE_THRESHOLD,
        )[0]

        proposals = instance_results_to_proposals(results)
        return json.dumps({"image_size": [H, W], "proposals": proposals})

    def _bbox_segment(self, image, bbox: Optional[List[int]], H: int, W: int) -> str:
        """Region-constrained segmentation within a bounding box."""
        import torch

        if bbox is None:
            # Fallback to auto if no bbox provided
            return self._auto_segment(image, H, W)

        inputs = self._processor(
            images=image,
            text="visual",
            input_boxes=[[[bbox[0], bbox[1], bbox[2], bbox[3]]]],
            input_boxes_labels=[[1]],
            original_sizes=[(H, W)],
            return_tensors="pt",
        ).to(self.device, dtype=torch.float16)

        outputs = self._model(**inputs)
        results = self._processor.post_process_instance_segmentation(
            outputs,
            target_sizes=[(H, W)],
            threshold=SCORE_THRESHOLD,
        )[0]

        proposals = instance_results_to_proposals(results)
        return json.dumps({"image_size": [H, W], "proposals": proposals})

    def _text_segment(self, image, text_prompt: str, H: int, W: int) -> str:
        """Open-vocabulary text-prompted segmentation."""
        import torch

        inputs = self._processor(
            images=image,
            text=text_prompt,
            return_tensors="pt",
        ).to(self.device, dtype=torch.float16)

        outputs = self._model(**inputs)
        results = self._processor.post_process_instance_segmentation(
            outputs,
            target_sizes=[(H, W)],
            threshold=SCORE_THRESHOLD,
        )[0]

        proposals = instance_results_to_proposals(results)
        return json.dumps({
            "image_size": [H, W],
            "query": text_prompt,
            "proposals": proposals,
        })

    def _exemplar_segment(self, image, bbox: Optional[List[int]], H: int, W: int) -> str:
        """Visual exemplar-based segmentation using a bounding box as positive prompt."""
        import torch

        if bbox is None:
            return json.dumps({
                "error": "No bounding box provided for exemplar_segment",
                "image_size": [H, W],
                "proposals": [],
            })

        inputs = self._processor(
            images=image,
            text="visual",
            input_boxes=[[[bbox[0], bbox[1], bbox[2], bbox[3]]]],
            input_boxes_labels=[[1]],
            original_sizes=[(H, W)],
            return_tensors="pt",
        ).to(self.device, dtype=torch.float16)

        outputs = self._model(**inputs)
        results = self._processor.post_process_instance_segmentation(
            outputs,
            target_sizes=[(H, W)],
            threshold=SCORE_THRESHOLD,
        )[0]

        proposals = instance_results_to_proposals(results)
        return json.dumps({
            "image_size": [H, W],
            "exemplar_bbox": bbox,
            "proposals": proposals,
        })

    def _concept_count(self, image, text_prompt: str, H: int, W: int) -> str:
        """Count objects matching a text description."""
        import torch

        inputs = self._processor(
            images=image,
            text=text_prompt,
            return_tensors="pt",
        ).to(self.device, dtype=torch.float16)

        outputs = self._model(**inputs)
        results = self._processor.post_process_object_detection(
            outputs,
            threshold=SCORE_THRESHOLD,
            target_sizes=[(H, W)],
        )[0]

        scores = results.get("scores", [])
        boxes = results.get("boxes", [])

        instances = []
        for i in range(len(scores)):
            box = boxes[i]
            instances.append({
                "bbox_xyxy": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                "score": round(float(scores[i]), 2),
            })

        instances.sort(key=lambda x: x["score"], reverse=True)
        instances = instances[:MAX_PROPOSALS]

        return json.dumps({
            "image_size": [H, W],
            "query": text_prompt,
            "count": len(instances),
            "instances": instances,
        })

    def _presence_check(self, image, text_prompt: str, H: int, W: int) -> str:
        """Quick check whether a concept is present in the image."""
        import torch

        inputs = self._processor(
            images=image,
            text=text_prompt,
            return_tensors="pt",
        ).to(self.device, dtype=torch.float16)

        outputs = self._model(**inputs)

        # Use presence_logits if available for a direct confidence score
        if hasattr(outputs, "presence_logits") and outputs.presence_logits is not None:
            presence_logits = outputs.presence_logits
            confidence = float(torch.sigmoid(presence_logits).max().cpu())
        else:
            # Fallback: use detection with low threshold
            results = self._processor.post_process_object_detection(
                outputs,
                threshold=PRESENCE_THRESHOLD,
                target_sizes=[(H, W)],
            )[0]
            scores = results.get("scores", [])
            confidence = float(max(scores)) if len(scores) > 0 else 0.0

        present = confidence >= SCORE_THRESHOLD
        # Get count via detection
        det_results = self._processor.post_process_object_detection(
            outputs,
            threshold=SCORE_THRESHOLD,
            target_sizes=[(H, W)],
        )[0]
        count = len(det_results.get("scores", []))

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
