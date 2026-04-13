"""Preprocess ReasonMap & ReasonMap Plus into verl-tool RL parquet format.

Usage:
    python preprocess_reasonmap_rl.py [--data_root /storage/openpsi/data] [--output_dir OUTPUT]
"""

from __future__ import annotations

import argparse
import io
import json
import random
import re
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset, Features, Image as HFImage, Sequence, Value
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


DEFAULT_DATA_ROOT = Path("/storage/openpsi/data")
DEFAULT_OUTPUT_DIR = Path("/storage/openpsi/data/reasonmap_rl")

REASONMAP_PLUS_TRAIN = "ReasonMap_plus/reasonmap_plus_train.parquet"
REASONMAP_PLUS_TEST = "ReasonMap_plus/reasonmap_plus_test.parquet"
REASONMAP_BASE_TRAIN = "ReasonMap/packaged/reasonmap_base_train_dataset.parquet"
REASONMAP_BASE_VAL = "ReasonMap/packaged/reasonmap_base_validation_dataset.parquet"

SHUFFLE_SEED = 42

TOOL_CALL_SYSTEM_PROMPT = """
You are an advanced AI agent capable of complex
reasoning and tool usage. You must strictly adhere
to the following protocol for every interaction:
1. Call appropriate tools based on the task;
2. Only persue one tool calling per action;
3. Reasoning Before Action: before selecting a tool,
you must analyze the user's request and determine
the necessary steps. Output your internal monologue
and logic inside <think> and </think> tags;
4. Tool Execution: If a tool is required, generate the
tool call immediately after your reasoning.
5. Reasoning After Action: Once you receive the
output from a tool, you must analyze the results to
determine if further actions are needed or if the task
is complete. Output this analysis inside <think> and </think> tags and then decide your next step, which could be calling another tool or providing the final answer.;
6. Final Output: When you have formulated your
conclusion, you must wrap your final answer in
<answer> and </answer> tags.
""".strip()

TOOL_DEFINITIONS_TEXT = """\
1. image_crop
   Crop a region from an image using bounding box coordinates.
   Arguments: {"image_index": int, "bounding_box": "x1,y1,x2,y2"}
   Coordinates are relative (0-1000 unified size). Returns cropped image.

2. image_label
   Add a text label at a position on an image.
   Arguments: {"image_index": int, "text": str, "position": "(x,y)"}
   Coordinates are relative (0-1000). Returns labeled image.

3. draw_line
   Draw a line segment on an image.
   Arguments: {"image_index": int, "coordinates": "x1,y1,x2,y2"}
   Coordinates are relative (0-1000). Returns modified image.

4. draw_path
   Draw a connected multi-point path on an image.
   Arguments: {"image_index": int, "points": "x1,y1,x2,y2,x3,y3,..."}
   Coordinates are relative (0-1000). At least 2 points required. Returns modified image.

5. bounding_box
   Draw a bounding box on an image.
   Arguments: {"image_index": int, "bounding_box": "x1,y1,x2,y2"}
   Coordinates are relative (0-1000). Returns image with box drawn.

6. image_highlight
   Highlight a rectangular region with yellow overlay on an image.
   Arguments: {"image_index": int, "bounding_box": "x1,y1,x2,y2"}
   Coordinates are relative (0-1000). Returns highlighted image.

7. text_spotting
   Text spotting with precise localization. Returns text content AND bounding box coordinates for each detected text region. Best for maps and annotated images.
   Arguments: {"image_index": int}
   Returns JSON with text and bbox for each detection.

8. map_text_ocr
   Text recognition optimized for maps. Extracts place names, road names, and landmarks while filtering out noise like pure numbers and scale markers.
   Arguments: {"image_index": int}
   Returns recognized map text."""

SYSTEM_PROMPT = (
    f"{TOOL_CALL_SYSTEM_PROMPT}\n\n"
    f"Available tools:\n{TOOL_DEFINITIONS_TEXT}\n\n"
    f"Use this format for tool calls:\n"
    f'<action>{{"name": "tool_name", "arguments": {{"param1": "value1"}}}}</action>'
)


OUTPUT_FORMATS = {
    "Counting1": "Provide the numeric answer inside <answer> and </answer> tags.",
    "Counting2": "Provide the numeric answer inside <answer> and </answer> tags.",
    "Counting3": "Provide the numeric answer inside <answer> and </answer> tags.",
    "TorF1": "Provide your answer (only yes or no) inside <answer> and </answer> tags.",
    "TorF2": "Provide your answer (only yes or no) inside <answer> and </answer> tags.",
    "reason_map": (
        "Provide the route plan inside <answer> and </answer> tags. "
        "Use the following format for each route section, separated by --:\n"
        "Route Name: <line name>\n"
        "Departure Stop: <station>\n"
        "Arrival Stop: <station>\n"
        "Number of Via Stops: <number>"
    ),
}
DEFAULT_OUTPUT_FORMAT = "Provide the final answer inside <answer> and </answer> tags."


USER_PROMPT_TEMPLATE = """\
Please answer the following {task_type} question:
Question: {question}
Please provide a complete step-by-step solution to
this problem. Your reasoning should:
1. Analyze the problem systematically
2. Check if the tool execution and answer are correct
3. If there are errors, explain what went wrong and
provide the correct reasoning
4. Provide the final answer
Use natural expressions like 'let me think' or 'hmm'
when helpful, but keep it concise. It's encouraged
to use self-reflection or verification especially in the
verifying tool output in the reasoning process.
Provide your detailed reasoning between <think>
and </think> tags, then give your final answer
between <answer> and </answer> tags.
Output format: {output_format}"""


MAX_SIDE = 4096
JPEG_QUALITY = 95

_BOXED_RE = re.compile(
    r'(?:[Pp]lease solve.*?\\boxed\{\}["\.]?\s*'
    r'|and put your answer.*?\\boxed\{\}["\.]?\s*'
    r'|put your answer.*?\\boxed\{\}["\.]?\s*)',
)


def decode_image_raw(img_field) -> Image.Image:
    if isinstance(img_field, dict):
        return Image.open(io.BytesIO(img_field["bytes"])).convert("RGB")
    if isinstance(img_field, str):
        import base64
        if img_field.startswith("data:image"):
            img_field = img_field.split("base64,", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(img_field))).convert("RGB")
    raise ValueError(f"Unsupported image field type: {type(img_field)}")


def resize_and_save(img: Image.Image, dest: Path, max_side: int = MAX_SIDE) -> Path:
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    img.save(str(dest), format="JPEG", quality=JPEG_QUALITY)
    return dest


def preprocess_images(data_root: Path, output_dir: Path):
    """Phase 1: extract one image per city, resize, save as JPEG."""
    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    city_to_path: dict[str, str] = {}

    for parquet_path in [
        data_root / REASONMAP_PLUS_TRAIN,
        data_root / REASONMAP_PLUS_TEST,
        data_root / REASONMAP_BASE_TRAIN,
        data_root / REASONMAP_BASE_VAL,
    ]:
        df = pd.read_parquet(parquet_path)
        for _, row in df.iterrows():
            city = str(row["city"])
            if city in city_to_path:
                continue
            dest = img_dir / f"{city}.jpg"
            if dest.exists():
                city_to_path[city] = str(dest)
                continue
            try:
                raw_img = decode_image_raw(row["image" if "image" in row else "figure"])
                resize_and_save(raw_img, dest)
                city_to_path[city] = str(dest)
                print(f"  {city}: {raw_img.size} -> {dest}")
            except Exception as e:
                print(f"  SKIP {city}: {e}")

    print(f"  {len(city_to_path)} unique city images saved")
    return city_to_path


def strip_boxed_instructions(question: str) -> str:
    q = _BOXED_RE.sub("", question).strip()
    return q if len(q) >= 10 else question


def build_reasonmap_plus_record(row: pd.Series, img_path: str) -> dict[str, Any]:
    qtype = str(row["type"])
    question = strip_boxed_instructions(str(row["question"]))

    output_format = OUTPUT_FORMATS.get(qtype, DEFAULT_OUTPUT_FORMAT)
    user_msg = "Observation 0:\n<image>\n" + USER_PROMPT_TEMPLATE.format(
        task_type="map reasoning",
        question=question,
        output_format=output_format,
    )

    image = Image.open(img_path).convert("RGB")

    return {
        "data_source": "reason_map_plus",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "images": [image],
        "reward_model": {
            "style": "rule",
            "ground_truth": str(row["answer"]),
            "extra": {
                "type": qtype,
                "station_1": "",
                "station_2": "",
                "metro_data": "",
            },
        },
        "extra_info": {
            "id": str(row.get("id", "")),
            "city": str(row.get("city", "")),
            "type": qtype,
        },
    }


def build_reasonmap_base_record(row: pd.Series, img_path: str) -> dict[str, Any]:
    question = str(row["question_long"])
    output_format = OUTPUT_FORMATS["reason_map"]
    user_msg = "Observation 0:\n<image>\n" + USER_PROMPT_TEMPLATE.format(
        task_type="map route planning",
        question=question,
        output_format=output_format,
    )

    image = Image.open(img_path).convert("RGB")

    metro_data_str = row.get("json", "")
    if isinstance(metro_data_str, dict):
        metro_data_str = json.dumps(metro_data_str, ensure_ascii=False)

    routes_str = row.get("routes", "")
    if isinstance(routes_str, dict):
        routes_str = json.dumps(routes_str, ensure_ascii=False)

    return {
        "data_source": "reason_map",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "images": [image],
        "reward_model": {
            "style": "rule",
            "ground_truth": routes_str,
            "extra": {
                "type": "",
                "station_1": str(row.get("station_1", "")),
                "station_2": str(row.get("station_2", "")),
                "metro_data": metro_data_str,
            },
        },
        "extra_info": {
            "id": str(row.get("id", "")),
            "city": str(row.get("city", "")),
            "type": "route_planning",
        },
    }


def build_features() -> Features:
    return Features(
        {
            "data_source": Value("string"),
            "prompt": [
                {
                    "role": Value("string"),
                    "content": Value("string"),
                }
            ],
            "images": Sequence(HFImage()),
            "reward_model": {
                "style": Value("string"),
                "ground_truth": Value("string"),
                "extra": {
                    "type": Value("string"),
                    "station_1": Value("string"),
                    "station_2": Value("string"),
                    "metro_data": Value("string"),
                },
            },
            "extra_info": {
                "id": Value("string"),
                "city": Value("string"),
                "type": Value("string"),
            },
        }
    )


def save_parquet(records: list[dict], output_path: Path, features: Features) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    dataset = Dataset.from_list(df.to_dict(orient="records"), features=features)
    dataset.to_parquet(str(output_path))
    print(f"  Saved {len(records)} records → {output_path}")


def process_dataset(df, build_fn, tag, city_to_path):
    records, skipped = [], []
    for idx, row in df.iterrows():
        city = str(row["city"])
        img_path = city_to_path.get(city)
        if img_path is None:
            skipped.append({"id": str(row.get("id", idx)), "error": f"no image for city={city}"})
            continue
        try:
            records.append(build_fn(row, img_path))
        except Exception as e:
            skipped.append({"id": str(row.get("id", idx)), "error": str(e)})
    print(f"  {tag}: {len(records)} ok, {len(skipped)} skipped")
    return records, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    features = build_features()

    print("=== Phase 1: Image preprocessing ===")
    city_to_path = preprocess_images(data_root, output_dir)

    print("=== Phase 2: Building parquets ===")

    print("ReasonMap Plus")
    rmp_train, _ = process_dataset(
        pd.read_parquet(data_root / REASONMAP_PLUS_TRAIN),
        build_reasonmap_plus_record, "rmp_train", city_to_path,
    )
    rmp_test, _ = process_dataset(
        pd.read_parquet(data_root / REASONMAP_PLUS_TEST),
        build_reasonmap_plus_record, "rmp_test", city_to_path,
    )

    print("ReasonMap Base")
    rm_train, _ = process_dataset(
        pd.read_parquet(data_root / REASONMAP_BASE_TRAIN),
        build_reasonmap_base_record, "rm_train", city_to_path,
    )
    rm_test, _ = process_dataset(
        pd.read_parquet(data_root / REASONMAP_BASE_VAL),
        build_reasonmap_base_record, "rm_test", city_to_path,
    )

    save_parquet(rmp_train, output_dir / "reasonmap_plus_train.parquet", features)
    save_parquet(rmp_test, output_dir / "reasonmap_plus_test.parquet", features)
    save_parquet(rm_train, output_dir / "reasonmap_base_train.parquet", features)
    save_parquet(rm_test, output_dir / "reasonmap_base_test.parquet", features)

    combined_train = rmp_train + rm_train
    combined_test = rmp_test + rm_test
    random.Random(SHUFFLE_SEED).shuffle(combined_train)
    random.Random(SHUFFLE_SEED).shuffle(combined_test)
    save_parquet(combined_train, output_dir / "combined_train.parquet", features)
    save_parquet(combined_test, output_dir / "combined_test.parquet", features)

    print(f"\nReasonMap Plus : train={len(rmp_train)} test={len(rmp_test)}")
    print(f"ReasonMap Base : train={len(rm_train)}  test={len(rm_test)}")
    print(f"Combined       : train={len(combined_train)}  test={len(combined_test)}")
    print(f"Output dir     : {output_dir}")


if __name__ == "__main__":
    main()
