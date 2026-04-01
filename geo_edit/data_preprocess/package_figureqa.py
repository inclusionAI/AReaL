"""Package FigureQA dataset to parquet format.

Downloads the dataset from HuggingFace, flattens the per-image QA pairs
into individual rows, adds an id column, and saves as parquet.

Source: https://huggingface.co/datasets/vikhyatk/figureqa

Each original row has an image and a list of ~10-12 Yes/No QA pairs.
This script flattens them so each row is one (image, question, answer) triple.

Usage:
    python geo_edit/data_preprocess/package_figureqa.py --out_dir /path/to/output
"""
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset, load_dataset


def package_figureqa(out_dir: Path, split: str = "train") -> Path:
    """Package FigureQA dataset into parquet format."""
    out_parquet = out_dir / f"figureqa_{split}_dataset.parquet"

    print(f"Loading FigureQA dataset ({split} split) from HuggingFace...")
    ds = load_dataset("vikhyatk/figureqa")[split]

    # Flatten: each image has multiple QA pairs
    print("Flattening QA pairs...")
    rows = []
    global_id = 0
    for idx in range(len(ds)):
        item = ds[idx]
        image = item["image"]
        for qa in item["qa"]:
            rows.append({
                "id": global_id,
                "image_idx": idx,
                "image": image,
                "question": qa["question"],
                "answer": qa["answer"],
            })
            global_id += 1

    flat_ds = Dataset.from_list(rows)

    # Save to parquet
    flat_ds.to_parquet(str(out_parquet))
    print(f"Saved parquet: {out_parquet}")
    print(f"Original images: {len(ds)}")
    print(f"Total QA pairs (flattened rows): {len(flat_ds)}")

    return out_parquet


def main() -> None:
    parser = argparse.ArgumentParser(description="Package FigureQA dataset to parquet.")
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
        help="Dataset split to package (default: train).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve() if args.out_dir else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)

    package_figureqa(out_dir, args.split)


if __name__ == "__main__":
    main()
