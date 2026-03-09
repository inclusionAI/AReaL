"""GroundingDINO Tool Agent - Open-vocabulary Object Detection."""

import base64
import json
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from geo_edit.environment.tool_agents.actor import BaseToolModelActor
from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)

# GroundingDINO doesn't need a system prompt (not a language model)
SYSTEM_PROMPT = ""

# Model configuration
agent_config = {
    "model_name_or_path": "/storage/openpsi/models/grounding-dino-base",
    "num_gpus": 1,
}

# Detection constants
BOX_THRESHOLD = 0.25      # Confidence threshold for boxes
TEXT_THRESHOLD = 0.25     # Text-image matching threshold
NMS_THRESHOLD = 0.8       # NMS IoU threshold
MAX_DETECTIONS = 20       # Maximum detections to return


def apply_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: List[str],
    iou_threshold: float = NMS_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Apply Non-Maximum Suppression.

    Args:
        boxes: Bounding boxes array (N, 4) in xyxy format.
        scores: Confidence scores (N,).
        labels: Detection labels (N,).
        iou_threshold: IoU threshold for NMS.

    Returns:
        Filtered boxes, scores, and labels after NMS.
    """
    import torch
    import torchvision.ops as ops

    if len(boxes) == 0:
        return boxes, scores, labels

    boxes_tensor = torch.from_numpy(boxes).float()
    scores_tensor = torch.from_numpy(scores).float()

    keep_indices = ops.nms(boxes_tensor, scores_tensor, iou_threshold)
    keep_indices = keep_indices.numpy()

    return (
        boxes[keep_indices],
        scores[keep_indices],
        [labels[i] for i in keep_indices]
    )


def format_detections(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: List[str],
    max_detections: int = MAX_DETECTIONS,
) -> List[Dict[str, Any]]:
    """Format detections for output.

    Args:
        boxes: Bounding boxes (N, 4) in pixel xyxy format.
        scores: Confidence scores (N,).
        labels: Detection labels (N,).
        max_detections: Maximum detections to return.

    Returns:
        List of detection dictionaries with pixel coordinates.
    """
    detections = []

    # Sort by score descending
    sorted_indices = np.argsort(scores)[::-1]

    for idx in sorted_indices[:max_detections]:
        box = boxes[idx]
        detections.append({
            "bbox_xyxy": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
            "score": round(float(scores[idx]), 3),
            "label": labels[idx]
        })

    return detections


class GroundingDINOActor(BaseToolModelActor):
    """GroundingDINO Object Detection Actor using HuggingFace transformers."""

    def __init__(self, model_name: str):
        """Initialize GroundingDINO actor.

        Args:
            model_name: Path to GroundingDINO model.
        """
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        self.setup_gpu()  # Configure GPU based on Ray assignment

        self.model_name = model_name

        # Load model immediately (no lazy loading)
        logger.info("Loading GroundingDINO model: %s", self.model_name)

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            device_map=self.device_map,
        )
        self.model.eval()
        self._initialized = True

        logger.info("GroundingDINOActor initialized on GPU %s: %s", self.gpu_ids, model_name)

    def analyze(
        self,
        image_b64: str,
        **kwargs,
    ) -> str:
        """Run GroundingDINO detection and return JSON with detections.

        Args:
            image_b64: Base64-encoded image string.
            **kwargs: Tool-specific parameters, expects 'question' with text prompt.

        Returns:
            JSON string with image_size, detections, and num_detections.
        """
        import torch
        from PIL import Image

        # Decode image
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        W, H = image.size  # PIL: (W, H)

        # Extract text prompt from kwargs
        text_prompt = kwargs.get("question", "").strip()
        if not text_prompt:
            return json.dumps({
                "error": "No text prompt provided",
                "image_size": [H, W],
                "detections": [],
                "num_detections": 0
            })

        try:
            # Process inputs
            inputs = self.processor(
                images=image,
                text=text_prompt,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Post-process using the processor's built-in method
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
                target_sizes=[(H, W)]  # (height, width)
            )

            # Extract results (first image in batch)
            result = results[0]
            boxes = result["boxes"].cpu().numpy()  # Already in pixel coords
            scores = result["scores"].cpu().numpy()
            labels = result["labels"]

            # Apply NMS
            boxes, scores, labels = apply_nms(boxes, scores, labels, NMS_THRESHOLD)

            # Format output
            detections = format_detections(boxes, scores, labels, MAX_DETECTIONS)

            # Parse requested labels from text prompt (separated by periods)
            requested_labels = [label.strip() for label in text_prompt.split('.') if label.strip()]

            # Identify which labels were found
            detected_labels = set(det["label"] for det in detections)
            not_found_labels = [label for label in requested_labels if label not in detected_labels]

            output = {
                "image_size": [H, W],
                "detections": detections,
                "num_detections": len(detections)
            }

            # Add not_found field if some labels were not detected
            if not_found_labels:
                output["not_found"] = not_found_labels

            return json.dumps(output)

        except Exception as e:
            logger.error("GroundingDINO detection failed: %s", e)
            return json.dumps({
                "error": str(e),
                "image_size": [H, W],
                "detections": [],
                "num_detections": 0
            })

    def health_check(self) -> dict:
        """Return health status of the actor."""
        return {
            "model": self.model_name,
            "initialized": self._initialized,
        }


ACTOR_CLASS = GroundingDINOActor

DECLARATION = {
    "name": "grounding_dino",
    "description": """GroundingDINO open-vocabulary object detection tool.
Detects objects in an image based on text descriptions.

Input:
- image_index: Index of the image to analyze
- question: Object labels to detect, separated by periods (e.g., "cat. dog. red car.")

Output: JSON with detected objects including bounding boxes (pixel coordinates), confidence scores, and labels.
Returns up to 20 detections with score >= 0.35, sorted by confidence.""",
    "parameters": {
        "type": "object",
        "properties": {
            "image_index": {
                "type": "integer",
                "description": "The index of the image to analyze. Each image is assigned an index like 'Observation 0', 'Observation 1', etc."
            },
            "question": {
                "type": "string",
                "description": "Object descriptions to detect, separated by periods. Example: 'a cat. a dog. a red car.'"
            }
        },
        "required": ["image_index", "question"]
    }
}

RETURN_TYPE = "text"
