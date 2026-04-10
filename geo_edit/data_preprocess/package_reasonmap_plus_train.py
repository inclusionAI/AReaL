"""Package ReasonMap-Plus train split to parquet format (local only).

Reads from a local parquet file and the maps/ directory, excluding 'planning'
type questions (open-ended, no ground truth answer).

Usage:
    python geo_edit/data_preprocess/package_reasonmap_plus_train.py \
        --data_dir /storage/openpsi/data/ReasonMap_plus
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm
from datasets import Dataset, Features, Value, load_dataset, Image as HFImage

# Question types to keep (planning excluded — open-ended with no reference answer)
KEEP_TYPES = {"Counting1", "Counting2", "Counting3", "TorF1", "TorF2"}


def package_reasonmap_plus(data_dir: Path, out_dir: Path, split: str = "train") -> Path:
    out_parquet = out_dir / f"reasonmap_plus_{split}.parquet"

    parquet_path = data_dir / "data" / f"{split}.parquet"
    print(f"Loading local parquet: {parquet_path}")
    ds = load_dataset("parquet", data_files=str(parquet_path))["train"]

    items = [item for item in ds if item["type"] in KEEP_TYPES]
    print(f"Kept {len(items)} items from {len(ds)} (planning excluded)")

    examples = []
    missing = 0
    print(f"Packaging {len(items)} items with map images...")

    for i, item in enumerate(tqdm(items, desc="Processing")):
        fig_rel = item["figure"].lstrip("./")
        image_path = data_dir / fig_rel

        if not image_path.exists():
            missing += 1
            if missing <= 5:
                print(f"Warning: Image not found: {image_path}")
            continue

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        examples.append({
            "id": i,
            "country": item["country"],
            "city": item["city"],
            "station_1": item["station_1"],
            "station_2": item["station_2"],
            "question": item["question"],
            "answer": item["answer"],
            "type": item["type"],
            "difficulty_question": item.get("difficulty_question", ""),
            "difficulty_city": item["difficulty_city"],
            "city_line_count": item["city_line_count"],
            "city_transfer_count": item["city_transfer_count"],
            "image": {"bytes": image_bytes, "path": None},
        })

    if missing > 0:
        print(f"Warning: {missing} entries skipped due to missing images.")

    features = Features({
        "id": Value("int64"),
        "country": Value("string"),
        "city": Value("string"),
        "station_1": Value("string"),
        "station_2": Value("string"),
        "question": Value("string"),
        "answer": Value("int64"),
        "type": Value("string"),
        "difficulty_question": Value("string"),
        "difficulty_city": Value("string"),
        "city_line_count": Value("int64"),
        "city_transfer_count": Value("int64"),
        "image": HFImage(),
    })

    dataset = Dataset.from_list(examples, features=features)
    dataset.to_parquet(str(out_parquet))

    print(f"Saved parquet: {out_parquet}")
    print(f"Total examples: {len(examples)}")

    return out_parquet


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package ReasonMap-Plus train split from local parquet."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory containing data/ and maps/ (e.g. /storage/openpsi/data/ReasonMap_plus).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. Defaults to data_dir.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split to package: train or test (default: train).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else data_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    package_reasonmap_plus(data_dir, out_dir, split=args.split)


if __name__ == "__main__":
    main()
