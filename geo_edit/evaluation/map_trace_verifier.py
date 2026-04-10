"""Rule-based verifier for MapTrace route-tracing task.

Parses coordinate sequences from model output, computes NDTW distance
against ground truth, and provides a TrajectoryJudge-compatible interface.

NDTW: D(i,j) = d(i,j) + min(D(i-1,j), D(i,j-1), D(i-1,j-1))
where d(i,j) = Euclidean distance between normalised [0,1] coordinates.
Lower = better.  Threshold default = 1.0.
"""
from __future__ import annotations

import ast
import math
import re
from typing import Any, Dict, List, Optional, Tuple


def parse_coordinates(text: str) -> List[Tuple[float, float]]:
    """Extract a list of (x, y) coordinate pairs from text.

    Handles formats:
      - Python list literal: [(0.1, 0.2), (0.3, 0.4)]
      - Parenthesised pairs on separate lines
      - Comma-separated bare pairs: 0.1, 0.2, 0.3, 0.4
    """
    answer_m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    body = answer_m.group(1).strip() if answer_m else text.strip()

    try:
        parsed = ast.literal_eval(body)
        if isinstance(parsed, list) and all(
            isinstance(p, (tuple, list)) and len(p) == 2 for p in parsed
        ):
            return [(float(p[0]), float(p[1])) for p in parsed]
    except (ValueError, SyntaxError):
        pass

    pairs = re.findall(r"\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)", body)
    if pairs:
        return [(float(x), float(y)) for x, y in pairs]

    nums = re.findall(r"[\d.]+", body)
    if len(nums) >= 4 and len(nums) % 2 == 0:
        return [
            (float(nums[i]), float(nums[i + 1])) for i in range(0, len(nums), 2)
        ]

    return []


def compute_ndtw(
    pred: List[Tuple[float, float]],
    gt: List[Tuple[float, float]],
) -> float:
    """Compute NDTW distance between two coordinate paths.

    Both paths should already be in normalised [0,1] coordinates.
    Returns cumulative DTW cost (lower = better).
    """
    m, n = len(pred), len(gt)
    if m == 0 or n == 0:
        return float("inf")

    dp = [[float("inf")] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = 0.0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dx = pred[i - 1][0] - gt[j - 1][0]
            dy = pred[i - 1][1] - gt[j - 1][1]
            cost = math.sqrt(dx * dx + dy * dy)
            dp[i][j] = cost + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n]


def map_trace_score(
    response: str,
    ground_truth: str,
    ndtw_threshold: float = 1.0,
) -> Tuple[float, bool, str]:
    """Score a MapTrace prediction.

    Returns (ndtw, is_success, reason).
    ``is_success`` is True if coordinates could be parsed.
    ``ndtw < ndtw_threshold`` indicates the prediction is accepted.
    """
    pred_coords = parse_coordinates(response)
    if not pred_coords:
        return float("inf"), False, "parse_failed"

    gt_coords = parse_coordinates(ground_truth)
    if not gt_coords:
        return float("inf"), False, "gt_parse_failed"

    ndtw = compute_ndtw(pred_coords, gt_coords)
    reason = f"ndtw={ndtw:.4f}"
    return ndtw, True, reason


def map_trace_judge(
    question: str,
    ground_truth: str,
    prediction: str,
    image_path: str = "",
    meta_info_extra: Optional[Dict[str, Any]] = None,
    ndtw_threshold: float = 1.0,
) -> Tuple[bool, str]:
    """TrajectoryJudge-compatible interface for MapTrace verification."""
    ndtw, is_success, reason = map_trace_score(prediction, ground_truth, ndtw_threshold)

    if not is_success:
        return False, f"map_trace_failed: {reason}"

    if ndtw <= ndtw_threshold:
        return True, f"map_trace_accepted: {reason}"

    return False, f"map_trace_rejected: {reason} (threshold={ndtw_threshold})"
