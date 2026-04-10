"""Package Farhadsakhodi/MapTrace dataset to parquet format.

Fields: ``image`` (HF Image), ``input`` (str), ``label`` (str).

Usage:
    python geo_edit/data_preprocess/package_maptrace.py \
        --input_dir /path/to/local/data \
        --out_dir /path/to/output \
        --split train
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm
from datasets import Dataset, Features, Image as HFImage, Value, load_dataset


def package_maptrace(input_dir: str | None, out_dir: Path, split: str = "train") -> Path:
    out_parquet = out_dir / f"maptrace_{split}_dataset.parquet"

    if input_dir:
        pattern = str(Path(input_dir) / f"*{split}*.parquet")
        print(f"Loading from local path: {pattern}")
        ds = load_dataset("parquet", data_files=pattern, split="train")
    else:
        print(f"Loading Farhadsakhodi/MapTrace ({split} split)...")
        ds = load_dataset("Farhadsakhodi/MapTrace", split=split)

    print(f"  total items: {len(ds)}")

    examples = []
    skipped = 0

    for i, item in enumerate(tqdm(ds, desc="Processing")):
        img = item.get("image")
        if img is None:
            skipped += 1
            continue

        label = item.get("label", "")
        input_text = item.get("input", "")

        if not input_text:
            skipped += 1
            continue

        examples.append({
            "id": i,
            "input": input_text,
            "label": label,
            "image": img,
        })

    if skipped:
        print(f"Warning: {skipped} entries skipped (missing image/input).")

    features = Features({
        "id": Value("int64"),
        "input": Value("string"),
        "label": Value("string"),
        "image": HFImage(),
    })

    dataset = Dataset.from_list(examples, features=features)
    dataset.to_parquet(str(out_parquet))

    print(f"Saved parquet: {out_parquet}")
    print(f"Total examples: {len(examples)}")

    return out_parquet


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package MapTrace dataset to parquet.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Local directory containing dataset parquet files. If not set, downloads from HuggingFace.",
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
        default="train",
        help="Dataset split to load (default: train).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve() if args.out_dir else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)

    package_maptrace(args.input_dir, out_dir, split=args.split)


if __name__ == "__main__":
    main()
