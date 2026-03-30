"""Reward functions for GeoEdit environment."""

import re
from typing import Optional


def normalize_answer(text: str) -> str:
    """Normalize answer text for comparison."""
    text = text.strip().lower()
    # Remove common formatting
    text = re.sub(r"\s+", " ", text)
    text = text.strip(".")
    return text


def compute_reward(
    predicted: str,
    ground_truth: str,
    task_type: str = "exact",
) -> float:
    """Compute reward by comparing predicted answer to ground truth.

    Args:
        predicted: Model's predicted answer text.
        ground_truth: Ground truth answer.
        task_type: Reward computation strategy:
            - "exact": Exact match after normalization
            - "contains": Ground truth contained in prediction
            - "numeric": Numeric comparison with tolerance
            - "option": Multiple-choice option matching (A/B/C/D)

    Returns:
        1.0 if correct, 0.0 otherwise.
    """
    pred = normalize_answer(predicted)
    gt = normalize_answer(ground_truth)

    if not pred or not gt:
        return 0.0

    if task_type == "exact":
        return 1.0 if pred == gt else 0.0

    elif task_type == "contains":
        return 1.0 if gt in pred else 0.0

    elif task_type == "numeric":
        return _numeric_reward(pred, gt)

    elif task_type == "option":
        return _option_reward(pred, gt)

    else:
        # Default to exact match
        return 1.0 if pred == gt else 0.0


def _numeric_reward(pred: str, gt: str, tolerance: float = 0.01) -> float:
    """Compare numeric answers with tolerance."""
    try:
        pred_val = float(re.findall(r"[-+]?\d*\.?\d+", pred)[-1])
        gt_val = float(re.findall(r"[-+]?\d*\.?\d+", gt)[-1])
        return 1.0 if abs(pred_val - gt_val) <= tolerance * max(abs(gt_val), 1.0) else 0.0
    except (ValueError, IndexError):
        return 0.0


def _option_reward(pred: str, gt: str) -> float:
    """Compare multiple-choice option answers."""
    # Extract option letter (A/B/C/D/E)
    pred_match = re.search(r"\b([a-e])\b", pred)
    gt_match = re.search(r"\b([a-e])\b", gt)

    if pred_match and gt_match:
        return 1.0 if pred_match.group(1) == gt_match.group(1) else 0.0

    # Fallback to exact match
    return 1.0 if pred == gt else 0.0
