"""Preprocess mixed RL datasets into verl-tool parquet format.

Converts MapQA, MapTrace, VisualProbe, DeepEyes into the same schema as
preprocess_reasonmap_rl.py and merges with existing reasonmap RL data.

Usage:
    python preprocess_mixed_rl.py [--output_dir OUTPUT]
"""
from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
from datasets import Dataset, Features, Image as HFImage, Sequence, Value
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

SEED = 42
OUT_DIR = Path("/storage/openpsi/data/mixed_rl")

EXISTING_TRAIN = "/storage/openpsi/data/reasonmap_rl/combined_train_rl_only.parquet"
EXISTING_VAL = "/storage/openpsi/data/reasonmap_rl/combined_test_10pct.parquet"

NEW_TRAIN = {
    "mm_mapqa": "/storage/openpsi/data/lcy_image_edit/mm_mapqa_rl_2k.parquet",
    "map_trace": "/storage/openpsi/data/lcy_image_edit/maptrace_rl_2k.parquet",
    "visual_probe": "/storage/openpsi/data/lcy_image_edit/visualprobe_rl_1k.parquet",
    "deep_eyes": "/storage/openpsi/data/lcy_image_edit/deepeyes_rl_1500.parquet",
}

NEW_VAL = {
    "visual_probe_easy": "/storage/openpsi/data/lcy_image_edit/visualprobe_easy_val.parquet",
    "visual_probe_medium": "/storage/openpsi/data/lcy_image_edit/visualprobe_medium_val.parquet",
    "visual_probe_hard": "/storage/openpsi/data/lcy_image_edit/visualprobe_hard_val.parquet",
    "map_trace": "/storage/openpsi/data/lcy_image_edit/maptrace_val_285.parquet",
}

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

TOOL_DEFINITIONS_TEXT = None


def _get_tool_definitions_text() -> str:
    global TOOL_DEFINITIONS_TEXT
    if TOOL_DEFINITIONS_TEXT is not None:
        return TOOL_DEFINITIONS_TEXT
    from geo_edit.tool_definitions.router import (
        TOOL_CATEGORIES,
        _TOOL_REGISTRY,
        format_tool_declarations_text,
    )
    target = set(TOOL_CATEGORIES["general"]) | set(TOOL_CATEGORIES["map"])
    decls = [_TOOL_REGISTRY[n][0] for n in sorted(target) if n in _TOOL_REGISTRY]
    TOOL_DEFINITIONS_TEXT = format_tool_declarations_text(decls)
    return TOOL_DEFINITIONS_TEXT

def _build_system_prompt() -> str:
    return (
        f"{TOOL_CALL_SYSTEM_PROMPT}\n\n"
        f"Available tools:\n{_get_tool_definitions_text()}\n\n"
        f"Use this format for tool calls:\n"
        f'<action>{{"name": "tool_name", "arguments": {{"param1": "value1"}}}}</action>'
    )

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

DEFAULT_OUTPUT_FORMAT = "Provide the final answer inside <answer> and </answer> tags."

FEATURES = Features({
    "data_source": Value("string"),
    "prompt": [{"role": Value("string"), "content": Value("string")}],
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
})


def decode_image(img_field) -> Image.Image:
    if isinstance(img_field, Image.Image):
        return img_field.convert("RGB")
    if isinstance(img_field, dict) and "bytes" in img_field:
        return Image.open(io.BytesIO(img_field["bytes"])).convert("RGB")
    if isinstance(img_field, bytes):
        return Image.open(io.BytesIO(img_field)).convert("RGB")
    if isinstance(img_field, str):
        return Image.open(img_field).convert("RGB")
    raise ValueError(f"Unsupported image type: {type(img_field)}")


def make_record(data_source: str, question: str, answer: str,
                image: Image.Image, task_type: str = "visual question answering",
                output_format: str = DEFAULT_OUTPUT_FORMAT,
                extra_id: str = "", extra_type: str = "") -> dict:
    user_msg = "Observation 0:\n<image>\n" + USER_PROMPT_TEMPLATE.format(
        task_type=task_type,
        question=question,
        output_format=output_format,
    )
    return {
        "data_source": data_source,
        "prompt": [
            {"role": "system", "content": _build_system_prompt()},
            {"role": "user", "content": user_msg},
        ],
        "images": [image],
        "reward_model": {
            "style": "rule",
            "ground_truth": str(answer),
            "extra": {"type": extra_type, "station_1": "", "station_2": "", "metro_data": ""},
        },
        "extra_info": {"id": extra_id, "city": "", "type": extra_type},
    }


def convert_mapqa(path: str) -> list[dict]:
    df = pq.read_table(path).to_pandas()
    records = []
    for _, row in df.iterrows():
        try:
            img = decode_image(row["image"])
            records.append(make_record(
                data_source="mm_mapqa",
                question=str(row["question"]),
                answer=str(row["answer"]),
                image=img,
                task_type="map question answering",
                extra_id=str(row["id"]),
            ))
        except Exception as e:
            print(f"  SKIP mapqa {row['id']}: {e}")
    return records


def convert_maptrace(path: str) -> list[dict]:
    df = pq.read_table(path).to_pandas()
    records = []
    output_format = (
        "Provide the list of (x, y) coordinate tuples "
        "inside <answer> and </answer> tags."
    )
    for _, row in df.iterrows():
        try:
            img = decode_image(row["image"])
            records.append(make_record(
                data_source="map_trace",
                question=str(row["input"]),
                answer=str(row["label"]),
                image=img,
                task_type="map path tracing",
                output_format=output_format,
                extra_id=str(row.get("id", "")),
                extra_type="map_trace",
            ))
        except Exception as e:
            print(f"  SKIP maptrace {row.get('id','?')}: {e}")
    return records


def convert_visualprobe_or_deepeyes(path: str, data_source: str) -> list[dict]:
    df = pq.read_table(path).to_pandas()
    records = []
    for _, row in df.iterrows():
        try:
            img_field = row["images"]
            if hasattr(img_field, '__iter__') and not isinstance(img_field, str):
                img_path = str(img_field[0])
            else:
                img_path = str(img_field)
            if not img_path.startswith("/"):
                img_path = f"/storage/openpsi/data/{img_path}"
            img = Image.open(img_path).convert("RGB")
            question = str(row["problem"]).replace("<image>\n", "").replace("<image>", "")
            records.append(make_record(
                data_source=data_source,
                question=question,
                answer=str(row["solution"]),
                image=img,
                extra_id=str(row.get("doc_id", "")),
            ))
        except Exception as e:
            print(f"  SKIP {data_source} {row.get('doc_id','?')}: {e}")
    return records


def save_parquet(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.from_list(records, features=FEATURES)
    dataset.to_parquet(str(output_path))
    print(f"  Saved {len(records)} records -> {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=str(OUT_DIR))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    random.seed(SEED)

    # --- Train ---
    print("=== Building train set ===")

    existing_train = pq.read_table(EXISTING_TRAIN)
    existing_records = existing_train.to_pydict()
    n_existing = existing_train.num_rows
    print(f"  Existing train: {n_existing}")

    new_records: list[dict] = []

    print("  Converting mm_mapqa...")
    new_records.extend(convert_mapqa(NEW_TRAIN["mm_mapqa"]))

    print("  Converting map_trace...")
    new_records.extend(convert_maptrace(NEW_TRAIN["map_trace"]))

    print("  Converting visual_probe...")
    new_records.extend(convert_visualprobe_or_deepeyes(NEW_TRAIN["visual_probe"], "visual_probe"))

    print("  Converting deep_eyes...")
    new_records.extend(convert_visualprobe_or_deepeyes(NEW_TRAIN["deep_eyes"], "deep_eyes"))

    print(f"  New records: {len(new_records)}")

    save_parquet(new_records, output_dir / "new_train.parquet")

    # Save existing as-is (already in correct format), combine paths in shell script
    print(f"\n  Train total: {n_existing} (existing) + {len(new_records)} (new) = {n_existing + len(new_records)}")

    # --- Val ---
    print("\n=== Building val set ===")

    new_val_records: list[dict] = []

    for ds_name, path in NEW_VAL.items():
        print(f"  Converting {ds_name}...")
        if "map_trace" in ds_name:
            new_val_records.extend(convert_maptrace(path))
        else:
            new_val_records.extend(convert_visualprobe_or_deepeyes(path, ds_name))

    print(f"  New val records: {len(new_val_records)}")
    save_parquet(new_val_records, output_dir / "new_val.parquet")

    existing_val = pq.read_table(EXISTING_VAL)
    n_existing_val = existing_val.num_rows
    print(f"\n  Val total: {n_existing_val} (existing) + {len(new_val_records)} (new) = {n_existing_val + len(new_val_records)}")

    print(f"\nOutput dir: {output_dir}")
    print("Use these in training script:")
    print(f"  train_data=\"[{EXISTING_TRAIN},{output_dir}/new_train.parquet]\"")
    print(f"  val_data=\"[{EXISTING_VAL},{output_dir}/new_val.parquet]\"")


if __name__ == "__main__":
    main()
