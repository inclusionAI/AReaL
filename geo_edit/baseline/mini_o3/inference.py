"""Mini-o3 inference on geo_edit datasets.

Faithfully reproduces Mini-o3's multi-turn crop-based visual search loop:

  System prompt tells model it can zoom in via <grounding>{"bbox_2d": ...}</grounding>

  Turn 1..N-1 (tool calls):
    Model: <think>...</think> <grounding>{"bbox_2d":[x0,y0,x1,y1],"source":"..."}</grounding>
    System: crop image at bbox → resize → inject as new observation image
  Turn N (final):
    Model: <think>...</think> <answer>final answer</answer>

  Up to max_rounds turns. Coordinates are relative (0~1). The model decides
  which image to crop from (original or any previous observation).

Prerequisites:
  - vLLM server running with Mini-o3 checkpoint:
      bash geo_edit/scripts/launch_vllm_generate.sh /path/to/Mini-o3-7B-v1 8000
  - geo_edit installed (pip install -e .)

Usage:
  python geo_edit/baseline/mini_o3/inference.py \
      --dataset_path /path/to/data.parquet \
      --dataset_name chartqa \
      --model_name_or_path /path/to/Mini-o3-7B-v1 \
      --api_base http://localhost:8000 \
      --output_dir ./mini_o3_results
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import datetime
import math
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from openai import OpenAI

from geo_edit.datasets.task_registry import DATASET_SPECS, get_dataset_spec
from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# System prompt (ported from Mini-o3/verl/trainer/constants.py)
# ---------------------------------------------------------------------------

MINI_O3_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question based on the image provided. "
    "Output your thinking process within the <think> and </think> tags. "
    "Whenever you find anything unclear, you can zoom in a specific region in the given image "
    "to see more clearly by outputing <grounding>{\"bbox_2d\": [x0, y0, x1, y1], "
    "\"source\": \"original_image\"}</grounding>, where (x0, y0) and (x1, y1) are the top-left "
    "and bottom-right coordinates of the region that you want to zoom in, respectively "
    "(suppose the width and height of the image are 1.0), and 'source' refers to the image "
    "that you zoom in and could be either 'original_image' or 'observation_i'. "
    "Once the final answer is confirmed, put it within <answer> and </answer>."
)

TOOL_CALL_OBSERVATION_TEMPLATE = (
    "After the above Action {action_turn}, here is the the zoom-in image (Observation {observation_turn}):\n"
    "Continue your reasoning process inside <think> and </think>. "
    "If needed, you can continue to zoom in on the original image or any of the observations, "
    "by outputting <grounding> and </grounding> as before. "
    "If the final answer is confirmed, put your final answer inside <answer> and </answer>."
)

ERROR_OBSERVATION_TEMPLATE = (
    "Please analyze the error information obtained from the function tool and adjust your response. "
    "Continue your reasoning process inside <think> and </think>."
)

# ---------------------------------------------------------------------------
# Crop tool (ported from Mini-o3/verl/workers/rollout/vllm_rollout/function_tools.py)
# ---------------------------------------------------------------------------


def crop_image(image: Image.Image, bbox: List[int], resize: int = 1) -> Image.Image:
    image_crop = image.crop(tuple(bbox))
    crop_w, crop_h = image_crop.size
    if resize > 1:
        target_w = max(28, int(crop_w * resize))
        target_h = max(28, int(crop_h * resize))
        image_crop = image_crop.resize((target_w, target_h), resample=Image.Resampling.LANCZOS)
    if image_crop.width < 28 or image_crop.height < 28:
        scale = 28 / min(image_crop.width, image_crop.height)
        image_crop = image_crop.resize(
            (max(28, int(image_crop.width * scale) + 1), max(28, int(image_crop.height * scale) + 1)),
            resample=Image.Resampling.LANCZOS,
        )
    return image_crop


def parse_grounding(text: str) -> Optional[Dict[str, Any]]:
    """Extract bbox_2d and source from <grounding>...</grounding>."""
    pattern = r"<grounding>\s*\{.*?\"bbox_2d\"\s*:\s*(\[.*?\]).*?\"source\"\s*:\s*['\"]([^'\"]+)['\"].*?\}\s*</grounding>"
    m = re.search(pattern, text, re.DOTALL)
    if not m:
        return None
    try:
        bbox = json.loads(m.group(1))
        source = m.group(2)
        if len(bbox) != 4:
            return None
        return {"bbox_2d": bbox, "source": source}
    except Exception:
        return None


def execute_crop(
    grounding: Dict[str, Any],
    observations: List[Image.Image],
) -> Tuple[Optional[Image.Image], Optional[str]]:
    """Execute crop on the specified source image using relative coordinates."""
    bbox = grounding["bbox_2d"]
    source = grounding["source"]

    if source == "original_image":
        obs_idx = 0
    else:
        m = re.match(r"observation_(\d+)", source)
        if not m:
            return None, f"Invalid source: {source}"
        obs_idx = int(m.group(1))

    if obs_idx >= len(observations):
        return None, f"Source {source} out of range (have {len(observations)} observations)"

    image = observations[obs_idx]
    w, h = image.size

    # relative coords → absolute pixels
    abs_bbox = [
        max(0, int(bbox[0] * w)),
        max(0, int(bbox[1] * h)),
        min(w - 1, int(bbox[2] * w)),
        min(h - 1, int(bbox[3] * h)),
    ]

    if abs_bbox[0] >= abs_bbox[2] or abs_bbox[1] >= abs_bbox[3]:
        return None, f"Invalid bbox after conversion: {abs_bbox}"

    width_box = abs_bbox[2] - abs_bbox[0]
    height_box = abs_bbox[3] - abs_bbox[1]
    if width_box / height_box >= 200 or height_box / width_box >= 200:
        return None, f"Aspect ratio too extreme: {width_box}x{height_box}"

    cropped = crop_image(image, abs_bbox)
    return cropped, None


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------


def pil_to_data_url(img: Image.Image) -> str:
    buf = BytesIO()
    rgb = img.convert("RGB") if img.mode != "RGB" else img
    rgb.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def load_image_from_item(item: Dict, image_key: str) -> Optional[Image.Image]:
    raw = item.get(image_key)
    if raw is None:
        return None
    if isinstance(raw, Image.Image):
        return raw
    if isinstance(raw, dict) and "bytes" in raw:
        return Image.open(BytesIO(raw["bytes"]))
    if isinstance(raw, bytes):
        return Image.open(BytesIO(raw))
    if isinstance(raw, str) and os.path.isfile(raw):
        return Image.open(raw)
    return None


def process_image(image: Image.Image, max_pixels: int = 2_000_000, min_pixels: int = 40_000) -> Image.Image:
    w, h = image.size
    if w * h > max_pixels:
        scale = math.sqrt(max_pixels / (w * h))
        image = image.resize((int(w * scale), int(h * scale)), resample=Image.Resampling.NEAREST)
    w, h = image.size
    if w * h < min_pixels:
        scale = math.sqrt(min_pixels / (w * h))
        image = image.resize((int(w * scale), int(h * scale)), resample=Image.Resampling.NEAREST)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def extract_answer(text: str) -> str:
    matches = _ANSWER_RE.findall(text)
    if matches:
        return matches[-1].strip()
    return text.strip()


# ---------------------------------------------------------------------------
# Core multi-turn inference loop
# ---------------------------------------------------------------------------


def run_mini_o3_single(
    client: OpenAI,
    model: str,
    image: Image.Image,
    question: str,
    max_rounds: int = 12,
    temperature: float = 1.0,
    max_tokens: int = 8192,
) -> Dict[str, Any]:
    observations: List[Image.Image] = [image]
    turn_responses: List[str] = []
    turn_crops: List[Optional[str]] = []

    original_url = pil_to_data_url(process_image(image))

    messages = [
        {"role": "system", "content": MINI_O3_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": original_url}},
                {"type": "text", "text": question},
            ],
        },
    ]

    for turn in range(max_rounds):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.99,
        )
        text = response.choices[0].message.content or ""
        turn_responses.append(text)

        grounding = parse_grounding(text)

        if grounding is None:
            turn_crops.append(None)
            break

        cropped, error = execute_crop(grounding, observations)

        if error or cropped is None:
            turn_crops.append(None)
            logger.warning("[turn %d] crop failed: %s", turn, error)
            error_msg = f"ERROR occurs during grounding. Error Information: {error}.\n"
            messages.append({"role": "assistant", "content": text})
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": error_msg + ERROR_OBSERVATION_TEMPLATE},
                ],
            })
            continue

        observations.append(cropped)
        obs_index = len(observations) - 1
        cropped_url = pil_to_data_url(process_image(cropped))
        turn_crops.append(f"observation_{obs_index}")

        observation_text = TOOL_CALL_OBSERVATION_TEMPLATE.format(
            action_turn=turn, observation_turn=obs_index,
        )

        messages.append({"role": "assistant", "content": text})
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": cropped_url}},
                {"type": "text", "text": observation_text},
            ],
        })

    full_response = " ".join(turn_responses)
    predicted = extract_answer(full_response)

    return {
        "turn_responses": turn_responses,
        "turn_crops": turn_crops,
        "num_turns": len(turn_responses),
        "num_crops": sum(1 for c in turn_crops if c is not None),
        "model_response": full_response,
        "predicted": predicted,
        "observations": observations,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Mini-o3 inference on geo_edit datasets")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True, choices=sorted(DATASET_SPECS.keys()))
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--api_base", type=str, default="http://localhost:8000")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_rounds", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--sample_rate", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = os.path.basename(args.model_name_or_path.rstrip("/"))
    output_path = os.path.join(args.output_dir, f"{model_short}_{args.dataset_name}_{timestamp}.jsonl")
    stats_path = os.path.join(args.output_dir, f"{model_short}_{args.dataset_name}_{timestamp}_stats.json")
    img_dir = os.path.join(args.output_dir, f"{model_short}_{args.dataset_name}_{timestamp}_images")
    os.makedirs(img_dir, exist_ok=True)

    client = OpenAI(
        base_url=args.api_base.rstrip("/") + "/v1",
        api_key="none",
    )

    dataset_spec = get_dataset_spec(args.dataset_name)
    dataset = load_dataset("parquet", data_files=args.dataset_path)["train"]

    if args.sample_rate < 1.0:
        n = int(len(dataset) * args.sample_rate)
        dataset = dataset.shuffle(seed=args.seed).select(range(n))
        logger.info("Sampled %d examples", n)

    logger.info("Dataset: %s (%d examples)", args.dataset_name, len(dataset))
    logger.info("Model: %s at %s", args.model_name_or_path, args.api_base)

    total_turns = 0
    total_crops = 0
    num_with_crop = 0
    num_direct = 0

    for item in tqdm(dataset, desc="Mini-o3 Inference"):
        item_id = str(item[dataset_spec.id_key])

        image_key = dataset_spec.image_key or "image"
        image = load_image_from_item(item, image_key)
        if image is None:
            logger.warning("[%s] no image, skipping", item_id)
            continue

        question = dataset_spec.build_prompt(item, use_tools=True)
        answer_gt = dataset_spec.get_answer(item)

        try:
            result = run_mini_o3_single(
                client=client,
                model=args.model_name_or_path,
                image=image,
                question=question,
                max_rounds=args.max_rounds,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        except Exception as e:
            logger.error("[%s] inference failed: %s", item_id, e)
            continue

        total_turns += result["num_turns"]
        total_crops += result["num_crops"]
        if result["num_crops"] > 0:
            num_with_crop += 1
        else:
            num_direct += 1

        crop_paths = []
        for obs_idx, obs_img in enumerate(result["observations"]):
            if obs_idx == 0:
                continue
            p = os.path.join(img_dir, f"{item_id}_obs_{obs_idx}.png")
            obs_img.convert("RGB").save(p)
            crop_paths.append(p)

        record = {
            "id": item_id,
            "question": question,
            "ground_truth": str(answer_gt),
            "predicted": result["predicted"],
            "num_turns": result["num_turns"],
            "num_crops": result["num_crops"],
            "turn_responses": result["turn_responses"],
            "turn_crops": result["turn_crops"],
            "crop_image_paths": crop_paths,
        }

        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    total = num_with_crop + num_direct
    stats = {
        "total": total,
        "num_with_crop": num_with_crop,
        "num_direct": num_direct,
        "total_turns": total_turns,
        "total_crops": total_crops,
        "avg_turns": total_turns / max(total, 1),
        "avg_crops": total_crops / max(total, 1),
    }
    logger.info("Stats: %s", json.dumps(stats))

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
