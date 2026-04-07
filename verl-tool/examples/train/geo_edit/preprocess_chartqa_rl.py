from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset, Features, Image as HFImage, Sequence, Value
from PIL import Image


SPLIT_INFO_PATH = Path(
    "/storage/openpsi/data/lcy_image_edit/chartqa_sft_data_1third/split_info.json"
)
SFT_TRAIN_PATH = Path(
    "/storage/openpsi/data/lcy_image_edit/chartqa_sft_data_1third/train.json"
)
CHARTQA_DATA_DIR = Path("/storage/openpsi/data/lcy_image_edit/chartqa_augmented_data")

OUTPUT_DIR = Path("/storage/openpsi/models/lcy_image_edit/rl_workspace/data")
TRAIN_OUTPUT_PATH = OUTPUT_DIR / "chartqa_rl_train.parquet"
VAL_OUTPUT_PATH = OUTPUT_DIR / "chartqa_rl_val.parquet"

SHUFFLE_SEED = 42
TRAIN_RATIO = 0.9
QUESTION_PREFIX_RE = re.compile(r"^(Question:\s*)+")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_removed_task_ids(path: Path) -> list[str]:
    split_info = load_json(path)
    removed_task_ids = split_info.get("removed_task_ids")
    if not isinstance(removed_task_ids, list):
        raise ValueError(f"Expected 'removed_task_ids' list in {path}")
    if len(removed_task_ids) != 1438:
        raise ValueError(
            f"Expected 1438 removed task ids, found {len(removed_task_ids)}"
        )
    return [str(task_id) for task_id in removed_task_ids]


def load_sft_alignment(path: Path) -> tuple[str, str]:
    train_data = load_json(path)
    if not isinstance(train_data, list) or not train_data:
        raise ValueError(f"Expected non-empty list in {path}")

    first_item = train_data[0]
    system_prompt = first_item["system"]
    first_user_message = first_item["conversations"][0]["value"]

    if not first_user_message.startswith("Observation 0:\n<image>\n"):
        raise ValueError(
            "Unexpected SFT first user message format; expected Observation 0 image prefix"
        )

    return system_prompt, first_user_message


def clean_question(question: str) -> str:
    cleaned = question.lstrip("\n")
    cleaned = cleaned.strip()
    cleaned = QUESTION_PREFIX_RE.sub("Question: ", cleaned)
    if not cleaned.startswith("Question: "):
        cleaned = f"Question: {cleaned}"
    return cleaned


def load_pil_image(image_path: Path) -> Image.Image:
    with Image.open(image_path) as image:
        image.load()
        return image.copy()


def build_record(task_id: str, system_prompt: str) -> dict[str, Any]:
    task_dir = CHARTQA_DATA_DIR / task_id
    meta_info_path = task_dir / "meta_info.jsonl"
    image_path = task_dir / "input_image.png"

    if not meta_info_path.exists():
        raise FileNotFoundError(f"missing meta_info.jsonl: {meta_info_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"missing input_image.png: {image_path}")

    with meta_info_path.open("r", encoding="utf-8") as f:
        first_line = f.readline().strip()

    if not first_line:
        raise ValueError(f"empty meta_info.jsonl first line: {meta_info_path}")

    meta = json.loads(first_line)
    question = clean_question(str(meta["question"]))
    answer = str(meta["answer"]).strip()
    user_message = f"Observation 0:\n<image>\n{question}"

    return {
        "data_source": "chartqa_rl",
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "images": [load_pil_image(image_path)],
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "task_id": task_id,
            "answer": answer,
            "question": question,
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
            },
            "extra_info": {
                "task_id": Value("string"),
                "answer": Value("string"),
                "question": Value("string"),
            },
        }
    )


def save_split(
    records: list[dict[str, Any]], output_path: Path, features: Features
) -> None:
    df = pd.DataFrame(records)
    serializable_records = df.to_dict(orient="records")
    dataset = Dataset.from_list(serializable_records, features=features)
    dataset.to_parquet(str(output_path))


def print_summary(
    total_records: int,
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    skipped: list[dict[str, str]],
    sft_first_user_message: str,
) -> None:
    print(f"Loaded total records: {total_records}")
    print(f"Train records: {len(train_records)}")
    print(f"Val records: {len(val_records)}")
    print(f"Skipped records: {len(skipped)}")

    if train_records:
        sample = train_records[0]
        print("Sample first record prompt:")
        print(json.dumps(sample["prompt"], ensure_ascii=False, indent=2))
        print("Sample first record reward_model:")
        print(json.dumps(sample["reward_model"], ensure_ascii=False, indent=2))
        print("SFT reference first user message:")
        print(sft_first_user_message)

    if skipped:
        reason_counts = Counter(item["reason"] for item in skipped)
        print("Skipped record reasons:")
        for reason, count in sorted(reason_counts.items()):
            print(f"  - {reason}: {count}")
        print("Skipped record details:")
        for item in skipped:
            print(json.dumps(item, ensure_ascii=False))


def main() -> None:
    removed_task_ids = load_removed_task_ids(SPLIT_INFO_PATH)
    system_prompt, sft_first_user_message = load_sft_alignment(SFT_TRAIN_PATH)

    records: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for task_id in removed_task_ids:
        try:
            records.append(build_record(task_id=task_id, system_prompt=system_prompt))
        except Exception as exc:  # noqa: BLE001
            skipped.append({"task_id": task_id, "reason": str(exc)})

    if not records:
        raise RuntimeError("No ChartQA RL records were created")

    random.Random(SHUFFLE_SEED).shuffle(records)
    train_size = int(len(records) * TRAIN_RATIO)
    train_records = records[:train_size]
    val_records = records[train_size:]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    features = build_features()
    save_split(train_records, TRAIN_OUTPUT_PATH, features)
    save_split(val_records, VAL_OUTPUT_PATH, features)

    print_summary(
        total_records=len(records),
        train_records=train_records,
        val_records=val_records,
        skipped=skipped,
        sft_first_user_message=sft_first_user_message,
    )
    print(f"Wrote train parquet: {TRAIN_OUTPUT_PATH}")
    print(f"Wrote val parquet: {VAL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
