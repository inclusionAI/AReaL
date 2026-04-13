#!/usr/bin/env python3
"""Convert Mini-o3-Coldstart-Dataset parquet to LLaMA Factory SFT (ShareGPT) format.

The source dataset is already split by turn: each row is one training sample
with a complete conversation up to a specific gpt response.  Multi-image rows
(e.g. round-2 data) contain both the original image and observation images.

Usage:
    python -m geo_edit.data_preprocess.convert_mini_o3_to_sft \
        --src_dir /storage/openpsi/data/Mini-o3-Coldstart-Dataset/data \
        --dst_dir /path/to/output

    python -m geo_edit.data_preprocess.convert_mini_o3_to_sft \
        --src_dir /storage/openpsi/data/Mini-o3-Coldstart-Dataset/data \
        --dst_dir /path/to/output \
        --enable_tools image_crop \
        --dataset_name mini_o3_sft
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from geo_edit.prompts.system_prompts import (
    DATASET_TASK_TYPES,
    DEFAULT_OUTPUT_FORMAT,
    build_tool_system_prompt,
    build_user_message,
)
from geo_edit.tool_definitions import ToolRouter, format_tool_declarations_text


# ---------------------------------------------------------------------------
# Grounding → image_crop mapping
# ---------------------------------------------------------------------------

_GROUNDING_RE = re.compile(r"<grounding>(.*?)</grounding>", re.DOTALL)
_OBSERVATION_TURN_RE = re.compile(
    r"After the above Action \d+, here is the.*?zoom-in image "
    r"\(Observation (\d+)\):\s*\n<image>\.\s*\n.*",
    re.DOTALL,
)


def _source_to_image_index(source: str) -> int:
    """Map grounding ``source`` field to image_index for image_crop.

    ``original_image`` → 0, ``observation_N`` → N.
    """
    if source == "original_image":
        return 0
    m = re.match(r"observation_(\d+)", source)
    if m:
        return int(m.group(1))
    return 0


def _convert_grounding_to_action(text: str) -> str:
    """Replace ``<grounding>{...}</grounding>`` with ``<action>{...}</action>``.

    Coordinate mapping: Mini-o3 bbox [0,1] float → image_crop [0,1000] int.
    """

    def _replace(match: re.Match) -> str:
        raw = match.group(1).strip()
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            return match.group(0)

        bbox = obj.get("bbox_2d", [0, 0, 1, 1])
        source = obj.get("source", "original_image")

        x1 = int(round(bbox[0] * 1000))
        y1 = int(round(bbox[1] * 1000))
        x2 = int(round(bbox[2] * 1000))
        y2 = int(round(bbox[3] * 1000))

        action_payload = {
            "name": "image_crop",
            "arguments": {
                "image_index": _source_to_image_index(source),
                "bounding_box": f"\\boxed{{{x1},{y1},{x2},{y2}}}",
            },
        }
        return f"<action>{json.dumps(action_payload)}</action>"

    return _GROUNDING_RE.sub(_replace, text)


# ---------------------------------------------------------------------------
# Conversation conversion
# ---------------------------------------------------------------------------


def _convert_observation_human(value: str) -> str:
    """Reformat Mini-o3 observation human turn to geo_edit style.

    ``After the above Action N ... (Observation M): <image>. Continue...``
    → ``Tool executed successfully. New image produced.\\nObservation M:\\n<image>``
    """
    m = _OBSERVATION_TURN_RE.match(value)
    if m:
        obs_idx = m.group(1)
        return (
            f"Tool executed successfully. New image produced.\n"
            f"Observation {obs_idx}:\n<image>"
        )
    return value


def convert_row(
    convs: List[Dict[str, str]],
    system_prompt: str,
    data_source: str,
) -> Optional[List[Dict[str, str]]]:
    """Convert a single row's conversations to geo_edit ShareGPT format.

    Returns the reformatted conversation list, or ``None`` on failure.
    """
    if len(convs) < 3:
        return None

    result: List[Dict[str, str]] = []

    first_human = convs[1] if len(convs) > 1 else None
    if not first_human or first_human.get("from") != "human":
        return None

    raw_question = first_human["value"]
    question_text = re.sub(r"^(<image>\s*\n?)+", "", raw_question).strip()
    question_text = re.sub(r"^(Question:\s*)+", "", question_text).strip()

    task_type = DATASET_TASK_TYPES.get(data_source, "visual question answering")
    first_user_value = build_user_message(
        question=question_text,
        num_images=1,
        task_type=task_type,
        output_format=DEFAULT_OUTPUT_FORMAT,
    )
    result.append({"from": "human", "value": first_user_value})

    for i in range(2, len(convs)):
        turn = convs[i]
        role = turn["from"]
        value = turn["value"]

        if role == "gpt":
            converted_value = _convert_grounding_to_action(value)
            result.append({"from": "gpt", "value": converted_value})

        elif role == "human":
            converted_value = _convert_observation_human(value)
            result.append({"from": "human", "value": converted_value})

        else:
            continue

    if len(result) < 2 or result[0]["from"] != "human":
        return None

    fixed: List[Dict[str, str]] = []
    for turn in result:
        if fixed and fixed[-1]["from"] == turn["from"]:
            fixed[-1]["value"] += "\n" + turn["value"]
        else:
            fixed.append(turn)

    return fixed


# ---------------------------------------------------------------------------
# Image extraction
# ---------------------------------------------------------------------------


def save_images(
    images,
    image_names,
    sample_id: str,
    dst_images_dir: str,
) -> List[str]:
    """Save images from parquet row to disk.

    Returns list of relative paths (``images/filename``).
    """
    rel_paths: List[str] = []
    for idx in range(len(image_names)):
        img_name = image_names[idx]
        filename = f"{sample_id}_{img_name}"
        dst_path = os.path.join(dst_images_dir, filename)

        if not os.path.exists(dst_path):
            img_data = images[idx]
            if isinstance(img_data, dict) and "bytes" in img_data:
                with open(dst_path, "wb") as f:
                    f.write(img_data["bytes"])
            elif isinstance(img_data, bytes):
                with open(dst_path, "wb") as f:
                    f.write(img_data)
            else:
                from PIL import Image

                if isinstance(img_data, Image.Image):
                    img_data.save(dst_path)
                else:
                    print(f"  [WARN] Unknown image type for {sample_id}[{idx}]: {type(img_data)}")
                    continue

        rel_paths.append(f"images/{filename}")

    return rel_paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Mini-o3-Coldstart-Dataset to LLaMA Factory SFT format"
    )
    parser.add_argument(
        "--src_dir",
        required=True,
        help="Directory containing parquet shards (train-*.parquet)",
    )
    parser.add_argument(
        "--dst_dir",
        required=True,
        help="Output directory for SFT data",
    )
    parser.add_argument(
        "--enable_tools",
        type=str,
        nargs="+",
        default=None,
        help="Override enabled tools (default: read from config.yaml)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="mini_o3_sft",
        help="Dataset name used in dataset_info.json",
    )
    args = parser.parse_args()

    src_dir = args.src_dir
    dst_dir = args.dst_dir
    dst_images = os.path.join(dst_dir, "images")
    os.makedirs(dst_images, exist_ok=True)

    # Build system prompt with tool definitions
    tool_router = ToolRouter(
        tool_mode="auto",
        enable_tools=args.enable_tools,
        skip_agent_init=True,
    )
    declarations = tool_router.get_available_declarations()
    tool_definitions_text = format_tool_declarations_text(declarations)
    system_prompt = build_tool_system_prompt(tool_definitions_text)

    print(f"Enabled tools: {[d['name'] for d in declarations]}")

    # Read all parquet shards
    parquet_files = sorted(glob.glob(os.path.join(src_dir, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {src_dir}")

    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} rows from {len(parquet_files)} parquet files")
    print(f"data_source distribution:\n{df['data_source'].value_counts().to_string()}")

    # Convert
    results: List[Dict[str, Any]] = []
    stats = Counter()

    for idx in tqdm(range(len(df)), desc="Converting", unit="row"):
        row = df.iloc[idx]
        sample_index = row["sample_index"]
        rollout_index = row["rollout_index"]
        data_source = row["data_source"]
        image_names = row["image_names"]
        images = row["images"]

        sample_id = f"{sample_index}_r{rollout_index}"

        # Parse conversations
        try:
            convs = json.loads(row["conversations"])
        except (json.JSONDecodeError, TypeError) as e:
            stats["bad_json"] += 1
            print(f"  [WARN] {sample_id}: bad conversations JSON: {e}")
            continue

        # Save images
        rel_paths = save_images(images, image_names, sample_id, dst_images)
        if not rel_paths:
            stats["no_images"] += 1
            continue

        # Convert conversations
        converted = convert_row(convs, system_prompt, data_source)
        if converted is None:
            stats["conversion_failed"] += 1
            continue

        # Count <image> tags to match with available images
        image_count = sum(t["value"].count("<image>") for t in converted)
        used_images = rel_paths[:image_count]

        results.append(
            {
                "conversations": converted,
                "images": used_images,
                "system": system_prompt,
            }
        )
        stats["kept"] += 1

    # Save train.json
    train_path = os.path.join(dst_dir, "train.json")
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save dataset_info.json
    dataset_name = args.dataset_name
    dataset_info = {
        dataset_name: {
            "file_name": "train.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "images": "images",
                "system": "system",
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
            },
        }
    }
    info_path = os.path.join(dst_dir, "dataset_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2)

    # Summary
    print(f"\n{'=' * 50}")
    print("Conversion Summary")
    print(f"{'=' * 50}")
    print(f"Total rows:              {len(df)}")
    print(f"Kept:                    {stats['kept']}")
    print(f"Skipped - bad JSON:      {stats['bad_json']}")
    print(f"Skipped - no images:     {stats['no_images']}")
    print(f"Skipped - conv failed:   {stats['conversion_failed']}")
    print(f"\nOutput: {train_path}")
    print(f"Images: {dst_images}")
    print(f"Dataset info: {info_path}")

    # Spot check
    if results:
        print(f"\n{'=' * 50}")
        print("Spot Check: First entry")
        print(f"{'=' * 50}")
        first = results[0]
        print(f"  Num turns: {len(first['conversations'])}")
        print(f"  Num images: {len(first['images'])}")
        print(f"  Images: {first['images']}")
        for j, turn in enumerate(first["conversations"]):
            preview = turn["value"][:150].replace("\n", "\\n")
            print(f"  Turn {j} [{turn['from']}]: {preview}...")

        # Also show a multi-image entry
        multi = next((r for r in results if len(r["images"]) >= 2), None)
        if multi:
            print(f"\nSpot Check: First multi-image entry")
            print(f"  Num turns: {len(multi['conversations'])}")
            print(f"  Num images: {len(multi['images'])}")
            print(f"  Images: {multi['images']}")
            for j, turn in enumerate(multi["conversations"]):
                preview = turn["value"][:150].replace("\n", "\\n")
                print(f"  Turn {j} [{turn['from']}]: {preview}...")


if __name__ == "__main__":
    main()
