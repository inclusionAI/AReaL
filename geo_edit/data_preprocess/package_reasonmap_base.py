"""Package FSCCS/ReasonMap (base) dataset to parquet format.

The ``figure`` column is a base64-encoded PNG; we decode it and store as
an HF Image column.  Only ``question_long`` is kept as the canonical
question field.

Usage:
    python geo_edit/data_preprocess/package_reasonmap_base.py \
        --out_dir /path/to/output
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from tqdm import tqdm
from datasets import Dataset, Features, Image as HFImage, Value, load_dataset


def package_reasonmap_base(input_dir: str | None, out_dir: Path, split: str = "validation") -> Path:
    out_parquet = out_dir / f"reasonmap_base_{split}_dataset.parquet"

    if input_dir:
        pattern = str(Path(input_dir) / f"*{split}*.parquet")
        print(f"Loading from local path: {pattern}")
        ds = load_dataset("parquet", data_files=pattern, split="train")
    else:
        print(f"Loading FSCCS/ReasonMap ({split} split)...")
        ds = load_dataset("FSCCS/ReasonMap", split=split)
    print(f"  total items: {len(ds)}")

    examples = []
    skipped = 0

    for i, item in enumerate(tqdm(ds, desc="Processing")):
        figure = item.get("figure", "")
        if not figure:
            skipped += 1
            if skipped <= 5:
                print(f"Warning: empty figure at index {i}")
            continue

        try:
            image_bytes = base64.b64decode(figure)
        except Exception:
            skipped += 1
            if skipped <= 5:
                print(f"Warning: failed to decode figure at index {i}")
            continue

        routes = item.get("routes", {})
        metro_json = item.get("json", {})

        examples.append({
            "id": i,
            "country": item.get("country", ""),
            "city": item.get("city", ""),
            "station_1": item.get("station_1", ""),
            "station_2": item.get("station_2", ""),
            "question_long": item.get("question_long", ""),
            "difficulty_question": item.get("difficulty_question", ""),
            "difficulty_city": item.get("difficulty_city", ""),
            "city_line_count": item.get("city_line_count", 0),
            "city_transfer_count": item.get("city_transfer_count", 0),
            "question_transfer_count": item.get("question_transfer_count", 0),
            "routes": json.dumps(routes, ensure_ascii=False) if isinstance(routes, dict) else str(routes),
            "json": json.dumps(metro_json, ensure_ascii=False) if isinstance(metro_json, dict) else str(metro_json),
            "image": {"bytes": image_bytes, "path": None},
        })

    if skipped:
        print(f"Warning: {skipped} entries skipped (empty/invalid figure).")

    features = Features({
        "id": Value("int64"),
        "country": Value("string"),
        "city": Value("string"),
        "station_1": Value("string"),
        "station_2": Value("string"),
        "question_long": Value("string"),
        "difficulty_question": Value("string"),
        "difficulty_city": Value("string"),
        "city_line_count": Value("int64"),
        "city_transfer_count": Value("int64"),
        "question_transfer_count": Value("int64"),
        "routes": Value("string"),
        "json": Value("string"),
        "image": HFImage(),
    })

    dataset = Dataset.from_list(examples, features=features)
    dataset.to_parquet(str(out_parquet))

    print(f"Saved parquet: {out_parquet}")
    print(f"Total examples: {len(examples)}")

    return out_parquet


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package FSCCS/ReasonMap (base) dataset to parquet.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Local directory containing dataset files. If not set, downloads from HuggingFace.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. Defaults to current directory.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="HuggingFace split to load (default: validation).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve() if args.out_dir else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)

    package_reasonmap_base(args.input_dir, out_dir, split=args.split)


if __name__ == "__main__":
    main()
