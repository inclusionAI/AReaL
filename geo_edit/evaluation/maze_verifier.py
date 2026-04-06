"""Maze answer verification module for VisWorld-Eval.

Algorithmically verifies maze path solutions by checking wall collisions
on the maze image. Ported from thuml/Reasoning-Visual-World eval/maze.py.

Scoring logic:
1. Extract predicted path from <point>X Y</point> tags
2. Validate each path segment against maze walls (Bresenham line)
3. Score = progress along ground truth path (0.0 to 1.0)
"""

import base64
import copy
import io
import re
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

try:
    from skimage.draw import line as skimage_line

    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


# ---------------------------------------------------------------------------
# Bresenham fallback (used when scikit-image is unavailable)
# ---------------------------------------------------------------------------


def _bresenham_line(
    y0: int, x0: int, y1: int, x1: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Pure-Python Bresenham line algorithm.

    Matches skimage.draw.line signature: (r0, c0, r1, c1) -> (rr, cc).
    """
    points_r, points_c = [], []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        points_r.append(y0)
        points_c.append(x0)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return np.array(points_r, dtype=np.intp), np.array(points_c, dtype=np.intp)


def _draw_line(r0: int, c0: int, r1: int, c1: int) -> Tuple[np.ndarray, np.ndarray]:
    """Draw a line using skimage if available, otherwise fallback to Bresenham."""
    if HAS_SKIMAGE:
        return skimage_line(r0, c0, r1, c1)
    return _bresenham_line(r0, c0, r1, c1)


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------


def _load_image(image: Union[str, Image.Image, np.ndarray]) -> Image.Image:
    """Load image from various sources: file path, base64 string, PIL Image, or numpy array."""
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, np.ndarray):
        return Image.fromarray(image)
    if isinstance(image, str):
        # Try as file path first
        try:
            return Image.open(image)
        except (FileNotFoundError, OSError):
            pass
        # Try as base64 string
        try:
            image_bytes = base64.b64decode(image)
            return Image.open(io.BytesIO(image_bytes))
        except Exception:
            pass
        raise ValueError(f"Cannot load image from string: {image[:100]}...")
    raise TypeError(f"Unsupported image type: {type(image)}")


# ---------------------------------------------------------------------------
# Core verification functions
# ---------------------------------------------------------------------------


def extract_maze_answer(pred_str: str) -> List[Tuple[int, int]]:
    """Extract maze path from model response.

    Parses <point>X Y</point> tags and returns pixel coordinates directly.
    Takes the last sequence of consecutive <point> tags found in the string.

    The VisWorld-Eval maze dataset uses pixel coordinates (matching the image
    dimensions, typically 512x512), so no coordinate scaling is applied.

    Returns:
        List of (x, y) tuples in pixel coordinates, or empty list if parsing fails.
    """
    # Find all consecutive sequences of <point> tags
    sequences = re.findall(r"(?:<point>\d+\s+\d+</point>)+", pred_str)
    if not sequences:
        return []

    # Take the last sequence
    last_seq = sequences[-1]
    line_pattern = re.compile(r"<point>(.*?)</point>", re.DOTALL)
    line_contents = line_pattern.findall(last_seq)

    try:
        coords = [(int(p.split()[0]), int(p.split()[1])) for p in line_contents]
    except (ValueError, IndexError):
        return []

    return coords


def parse_gt_path(answer_str: str) -> List[Tuple[int, int]]:
    """Parse ground truth path from space-separated integer string.

    Args:
        answer_str: "X1 Y1 X2 Y2 ..." format string.

    Returns:
        List of (x, y) tuples.
    """
    nums = list(map(int, re.findall(r"\d+", answer_str)))
    return [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)]


def get_maze_bounds(
    gray_array: np.ndarray,
    white_threshold: int = 250,
) -> Tuple[int, int, int, int]:
    """Detect the bounding box of the maze region (non-white area).

    Scans the grayscale image for pixels with value < white_threshold.
    These pixels form the maze area. Returns their bounding box.

    Args:
        gray_array: 2D grayscale numpy array of the maze image.
        white_threshold: Pixels >= this value are considered white (outside maze).

    Returns:
        (x_min, x_max, y_min, y_max) bounding box of the maze region.

    Raises:
        ValueError: If no maze region is detected.
    """
    mask = gray_array < white_threshold
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("No maze region detected")

    return xs.min(), xs.max(), ys.min(), ys.max()


def is_valid_step(
    gray_array: np.ndarray,
    point1: Tuple[int, int],
    point2: Tuple[int, int],
    threshold: int = 45,
) -> Tuple[bool, str]:
    """Validate a single path segment between two points.

    Uses Bresenham line algorithm to check every pixel along the segment.
    A pixel is a wall if its grayscale value < threshold.

    Args:
        gray_array: 2D grayscale array of the maze.
        point1: (x, y) start point.
        point2: (x, y) end point.
        threshold: Grayscale value below which a pixel is considered a wall.

    Returns:
        (is_valid, reason) tuple.
    """
    height, width = gray_array.shape

    x1, y1 = int(point1[0]), int(point1[1])
    x2, y2 = int(point2[0]), int(point2[1])

    # Check image bounds
    if not (
        0 <= x1 < width and 0 <= y1 < height and 0 <= x2 < width and 0 <= y2 < height
    ):
        return False, "Endpoints outside image bounds"

    # Get maze region bounds
    x_min, x_max, y_min, y_max = get_maze_bounds(gray_array)

    # Check endpoints are within maze region and not on walls
    for x, y in [(x1, y1), (x2, y2)]:
        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            return False, "Endpoint in outer white border"
        if gray_array[y, x] < threshold:
            return False, "Endpoint on wall"

    # Trace line and check each pixel
    rr, cc = _draw_line(y1, x1, y2, x2)
    rr = np.clip(rr, 0, height - 1)
    cc = np.clip(cc, 0, width - 1)

    for r, c in zip(rr, cc):
        if not (x_min <= c <= x_max and y_min <= r <= y_max):
            return False, "Path crosses outer white border"
        if gray_array[r, c] < threshold:
            return False, "Path crosses wall"

    return True, "Valid"


def wall_judge(
    response: str,
    image: Union[str, Image.Image, np.ndarray],
    answer: str,
    maze_size: str = "5",
) -> Tuple[float, np.ndarray]:
    """Score a maze path prediction against ground truth.

    Args:
        response: Model's response text containing <point> tags.
        image: Maze image (PIL Image, file path, base64 string, or numpy array).
        answer: Ground truth path as space-separated integers "X1 Y1 X2 Y2 ...".
        maze_size: Maze difficulty level (affects min_delta threshold).

    Returns:
        (score, drawn_array) where score is 0.0-1.0 and drawn_array is the
        maze image with the predicted path drawn on it.
    """
    # Parse ground truth path
    gt_path = parse_gt_path(answer)

    # Parse predicted path
    pred_path = extract_maze_answer(response)

    # Load image and convert to grayscale
    pil_image = _load_image(image)
    gray_array = np.array(pil_image.convert("RGB")).max(-1)
    drawn_array = copy.deepcopy(gray_array)

    if len(pred_path) <= 1:
        return 0, drawn_array

    # Draw predicted path on visualization
    for i in range(1, len(pred_path)):
        drawn_array = cv2.line(
            drawn_array, pred_path[i - 1], pred_path[i], (0, 0, 0), 2
        )

    # Compute min_delta based on maze size
    if maze_size == "5":
        min_delta = 100
    else:
        raise NotImplementedError(f"Unsupported maze_size: {maze_size}")

    min_delta = min_delta / 1500 * max(gray_array.shape)

    # Check start point proximity to GT start
    first_point = pred_path[0]
    dist_from_start = np.sqrt(
        (first_point[0] - gt_path[0][0]) ** 2
        + (first_point[1] - gt_path[0][1]) ** 2
        + 0.01
    )
    if dist_from_start >= min_delta:
        return 0, drawn_array

    # Validate each step - any wall crossing → score 0
    last_point = pred_path[0]
    for i in range(1, len(pred_path)):
        valid, reason = is_valid_step(gray_array, pred_path[i - 1], pred_path[i])
        if not valid:
            return 0, drawn_array
        if i == len(pred_path) - 1:
            last_point = pred_path[i]

    # Find closest GT waypoint to the end of valid predicted path
    index = 0
    current_min_delta = min_delta
    for i, (px, py) in enumerate(gt_path):
        dist = np.sqrt((px - last_point[0]) ** 2 + (py - last_point[1]) ** 2 + 0.01)
        if dist < current_min_delta:
            current_min_delta = dist
            index = i

    score = index / (len(gt_path) - 1) if len(gt_path) > 1 else 0
    return score, drawn_array


# ---------------------------------------------------------------------------
# TrajectoryJudge-compatible interface
# ---------------------------------------------------------------------------


def maze_judge(
    question: str,
    ground_truth: str,
    prediction: str,
    image_path: str,
    score_threshold: float = 1.0,
) -> Tuple[bool, str]:
    """Judge maze answer correctness, compatible with TrajectoryJudge interface.

    Args:
        question: The question (unused, kept for interface compatibility).
        ground_truth: Ground truth path string.
        prediction: Model's prediction text containing <point> tags.
        image_path: Path to the maze image file.
        score_threshold: Minimum score to consider correct (default 1.0 = perfect path).

    Returns:
        (is_correct, reason) tuple matching TrajectoryJudge.judge_correctness signature.
    """
    try:
        score, _ = wall_judge(prediction, image_path, ground_truth)
    except Exception as e:
        return False, f"maze_judge_error: {e}"

    if score >= score_threshold:
        return True, "valid"

    # Extract predicted path for diagnostics
    pred_path = extract_maze_answer(prediction)
    if not pred_path:
        return False, f"maze_no_path: no <point> tags found in prediction"
    if len(pred_path) <= 1:
        return False, f"maze_insufficient_points: only {len(pred_path)} point(s)"

    return False, f"maze_score={score:.3f} (threshold={score_threshold})"
