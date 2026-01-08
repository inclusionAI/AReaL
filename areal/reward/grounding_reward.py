import math
import re
import numpy as np
import json, random
from typing import Any, List, Sequence, Tuple, Union


Number = Union[int, float]
Box = Tuple[Number, Number, Number, Number]  # (x1, y1, x2, y2)
BOX_RE = re.compile(r"<box>\s*\[(.*?)\]\s*</box>", flags=re.IGNORECASE | re.DOTALL)
INT_RE = re.compile(r"-?\d+")
_NUM_RE = re.compile(r'[-+]?\d*\.?\d+')
_NUM_FULL_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_ANSWER_RE = re.compile(r'<answer>(.*?)</answer>', flags=re.S | re.I)

def extract_box_from_text(s: str) -> List[float]:
    if not isinstance(s, str) or not s:
        return []

    def _extract_from_text(text: str) -> List[float]:
        boxes = re.findall(r'\[([^\[\]]+)\]', text)
        if boxes:
            candidate = boxes[-1]
            nums = _NUM_RE.findall(candidate)
        else:
            nums = _NUM_RE.findall(text)
            if len(nums) >= 4:
                nums = nums[-4:]
            else:
                return []

        if len(nums) < 4:
            return []

        vals: List[float] = []
        for v in nums[:4]:
            if not _NUM_FULL_RE.match(v.strip()):
                return []
            x = float(v)
            if not math.isfinite(x):
                return []
            vals.append(x)
        
        x1, y1, x2, y2 = vals
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1
        return [x1, y1, x2, y2]

    ans_match = _ANSWER_RE.search(s)
    if ans_match:
        inner = ans_match.group(1)
        box = _extract_from_text(inner)
        if box:
            return box

    return _extract_from_text(s)

def _iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    if any(math.isnan(v) for v in (ax1, ay1, ax2, ay2, bx1, by1, bx2, by2)):
        return 0.0

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union

def scale_bbox(bbox, w, h):
    x1, y1, x2, y2 = bbox
    x1 = round(x1 * w / 1000.0)
    x2 = round(x2 * w / 1000.0)
    y1 = round(y1 * h / 1000.0)
    y2 = round(y2 * h / 1000.0)


    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    if x1 == x2 and w > 1: x2 = min(x1 + 1, w - 1)
    if y1 == y2 and h > 1: y2 = min(y1 + 1, h - 1)
    return [float(x1), float(y1), float(x2), float(y2)]

def to_box_tuple(b: Sequence[Union[int, float]]) -> Box:
    if not isinstance(b, (list, tuple)) or len(b) != 4:
        return (math.nan, math.nan, math.nan, math.nan)
    return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))

def _to_tokens(text: Any):
    s = "" if text is None else str(text)
    return s.split()

def _length_penalty_tokens(tokens, max_tokens: int = 512, step: int = 50, per_step: float = 0.01) -> float:
    n = len(tokens)
    if n <= max_tokens:
        return 0.0
    over = n - max_tokens
    chunks = (over + step - 1) // step  # ceil(over/step) 且不足50按50计算
    return chunks * per_step

def _has_consecutive_repeats(tokens, min_run: int = 7) -> bool:
    if not tokens:
        return False
    run = 1
    prev = tokens[0]
    for t in tokens[1:]:
        if t == prev:
            run += 1
            if run >= min_run:
                return True
        else:
            prev = t
            run = 1
    return False


def compute_score(
    prompt: Any,
    completions: Any,
    prompt_ids: Any,
    completion_ids: Any,
    **task_data,
):
    """
    Compute IoU between predicted and ground-truth bounding boxes (single example).

    Inputs:
      - completions: str containing the coords (e.g., "<box>[x1,y1,x2,y2]</box>")
      - task_data: should provide ground truth box and image size (height/width)

    Returns:
      - float IoU for the single example.
    """
    extra_info = task_data["extra_info"]
    reward_model = task_data["reward_model"] 

    split = extra_info["split"]
    is_test = split == "test"
    
    reward_type = "mix"
    if is_test:
        reward_type = "eval"

    alpha = 0.5
    iou_threshold =0.5

    if alpha is None or iou_threshold is None or reward_type is None:
        raise ValueError("Missing required parameters: reward_type, alpha, threshold.")
    
    gt_box_raw = reward_model["ground_truth"]
    width = extra_info["width"]
    height = extra_info["height"]
    
    if gt_box_raw is None:
        raise ValueError("Missing ground truth bounding box in task_data.")
    if width is None or height is None:
        raise ValueError("Height and width must be provided in task_data for scaling bounding boxes.")
    w = float(width)
    h = float(height)
    pb = extract_box_from_text(completions)
    has_valid_pred = isinstance(pb, (list, tuple)) and len(pb) == 4
    gt_box = scale_bbox(gt_box_raw, w, h)
    if has_valid_pred:
        pred_box = scale_bbox(pb, w, h)
        iou_val = float(_iou(to_box_tuple(pred_box), to_box_tuple(gt_box)))
    else:
        pred_box = [0.0, 0.0, 0.0, 0.0]
        iou_val = 0.0
    
    if reward_type == "mix":
        pass_val = 1.0 if iou_val >= iou_threshold else 0.0
        reward = alpha * pass_val + (1.0 - alpha) * iou_val
    elif reward_type == "sigmoid":
        reward = 1.0 / (1.0 + math.exp(-8.0 * (iou_val - iou_threshold)))
    elif reward_type == "penalty":
        pass_val = 1.0 if iou_val >= iou_threshold else 0.0
        base = alpha * pass_val + (1.0 - alpha) * iou_val
        toks = _to_tokens(completions)
        len_pen = _length_penalty_tokens(
            toks,
            max_tokens=512,
            step=50,
            per_step=0.01,
        )
        rep_pen = 0.3 if _has_consecutive_repeats(toks, min_run=7) else 0.0
        reward = base - (len_pen + rep_pen)
        
    elif reward_type == "normalized_exp":
        K = np.log(16.0)  
        DENOM = 1.0 - np.exp(-K) 
        reward = (1.0 - np.exp(-K * iou_val)) / DENOM
    elif reward_type == "eval":
        reward = 1.0 if iou_val >= iou_threshold else 0.0
    else:
        raise ValueError("Unknown reward_type. Use 'mix', 'sigmoid', or 'raw'.")
    if random.random() < 0.001:
        case = {
            "split": split,
            "reward_type": reward_type,
            "alpha": alpha,
            "threshold": iou_threshold,
            "image_size": {"w": w, "h": h},
            "solution_str": completions,
            "pb": list(pb),
            "ground_truth": list(gt_box_raw),
            "pred_box": list(pred_box),
            "gt_box": list(gt_box),
            "iou": iou_val,
            "reward": reward,
        }
        print("[CASE]", json.dumps(case, ensure_ascii=False))

    return reward

