"""SAM2 (Segment Anything Model 2) Tool Agent."""

import base64
import json
import re
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from geo_edit.environment.tool_agents.actor import BaseToolModelActor
from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)

# SAM2 doesn't need a system prompt (not a language model)
SYSTEM_PROMPT = ""

# Model configuration
agent_config = {
    "model_name_or_path": "/storage/openpsi/models/sam2.1-hiera-large",
    "max_model_len": 8192,        # Unused, interface compatibility
    "gpu_memory_utilization": 0.8, # Unused, interface compatibility
    "temperature": 0.0,           # Unused
    "max_tokens": 4096,           # Unused
    "num_gpus": 1,
}

# Constants
SCORE_THRESHOLD = 0.25
MAX_PROPOSALS = 20
NORMALIZED_SIZE = 1000  # Bounding box coordinate normalization factor


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


class SAM2Actor(BaseToolModelActor):
    """SAM2 Segmentation Actor using HuggingFace transformers."""

    def __init__(
        self,
        model_name: str,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.8,
        system_prompt: Optional[str] = None,
    ):
        import torch
        from transformers import AutoProcessor, AutoModelForMaskGeneration

        self.setup_gpu()  # Configure GPU based on Ray assignment

        self.model_name = model_name

        # Load model immediately (no lazy loading)
        logger.info("Loading SAM2 model: %s", self.model_name)

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForMaskGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=self.device_map,
        )
        self.model.eval()
        self._initialized = True

        logger.info("SAM2Actor initialized on GPU %s: %s", self.gpu_ids, model_name)

    def analyze(
        self,
        image_b64: str,
        question: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Run SAM2 segmentation and return JSON with proposals.

        Args:
            image_b64: Base64-encoded image string.
            question: Empty for auto mode, or contains \\boxed{x1,y1,x2,y2} for bbox mode.
            temperature: Unused.
            max_tokens: Unused.

        Returns:
            JSON string with image_size and proposals.
        """
        import torch
        from PIL import Image

        # Decode image
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        W, H = image.size  # PIL: (W, H)

        # Parse mode from question
        bbox = self._parse_bbox(question, W, H)

        try:
            if bbox is None:
                # Mode 1: Automatic segmentation
                masks, scores = self._auto_segment(image)
            else:
                # Mode 2: Bbox-constrained segmentation
                masks, scores = self._bbox_segment(image, bbox)

            # Convert to proposals
            proposals = masks_to_proposals(masks, scores, (H, W))

            result = {
                "image_size": [H, W],
                "proposals": proposals
            }
            return json.dumps(result)

        except Exception as e:
            logger.error("SAM2 segmentation failed: %s", e)
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

    def _auto_segment(self, image) -> Tuple[np.ndarray, np.ndarray]:
        """Automatic mask generation for entire image."""
        import torch

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate_masks(**inputs)

        # Extract masks and scores
        masks = outputs.pred_masks.cpu().numpy()
        scores = outputs.iou_scores.cpu().numpy()

        # Flatten batch dimension if present
        if masks.ndim == 4:
            masks = masks.squeeze(0)
        if scores.ndim == 2:
            scores = scores.squeeze(0)

        return masks, scores

    def _bbox_segment(self, image, bbox: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Segment within a bounding box prompt."""
        import torch

        # Prepare inputs with box prompt
        inputs = self.processor(
            images=image,
            input_boxes=[[[bbox]]],
            return_tensors="pt"
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extract masks and scores
        masks = outputs.pred_masks.cpu().numpy()
        scores = outputs.iou_scores.cpu().numpy()

        # Flatten dimensions
        masks = masks.squeeze()
        scores = scores.squeeze()

        # Ensure proper shape
        if masks.ndim == 2:
            masks = masks[np.newaxis, ...]
        if scores.ndim == 0:
            scores = np.array([float(scores)])

        return masks, scores

    def health_check(self) -> dict:
        """Return health status of the actor."""
        return {
            "model": self.model_name,
            "initialized": self._initialized,
        }


ACTOR_CLASS = SAM2Actor

DECLARATION = {
    "name": "sam2",
    "description": """SAM2 (Segment Anything Model 2) automatic segmentation tool.
- Mode 1 (auto): Input image_index only → returns automatic mask proposals for entire image
- Mode 2 (bbox): Input image_index + bounding_box → returns refined mask within bounding box

The bounding_box should be provided in the format "\\boxed{x1,y1,x2,y2}" where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner. All (x,y) values should be between 0-1000 (normalized coordinates).

Returns JSON with mask proposals including scores, bounding boxes, areas, and centroids. Max 20 proposals with score >= 0.25.""",
    "parameters": {
        "type": "object",
        "properties": {
            "image_index": {
                "type": "integer",
                "description": "The index of the image to segment. Each image is assigned an index like 'Observation 0', 'Observation 1', etc."
            },
            "bounding_box": {
                "type": "string",
                "description": "Optional. Bounding box coordinates in format '\\boxed{x1,y1,x2,y2}' to constrain segmentation. If omitted, performs automatic full-image segmentation."
            }
        },
        "required": ["image_index"]
    }
}

RETURN_TYPE = "text"
