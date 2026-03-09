"""Package ChartQA dataset to parquet format.

Downloads the dataset from HuggingFace, adds an id column, and saves as parquet.

Usage:
    python geo_edit/data_preprocess/package_chartqa.py --out_dir /path/to/output
"""
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset


def package_chartqa(out_dir: Path, split: str = "test") -> Path:
    """Package ChartQA dataset into parquet format."""
    out_parquet = out_dir / f"chartqa_{split}_dataset.parquet"

    print(f"Loading ChartQA dataset ({split} split) from HuggingFace...")
    ds = load_dataset("lmms-lab/ChartQA")[split]

    # Add id column
    ds = ds.add_column("id", list(range(len(ds))))

    # Save to parquet
    ds.to_parquet(str(out_parquet))
    print(f"Saved parquet: {out_parquet}")
    print(f"Total examples: {len(ds)}")

    return out_parquet


def main() -> None:
    parser = argparse.ArgumentParser(description="Package ChartQA dataset to parquet.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. Defaults to current directory.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to package (default: test).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve() if args.out_dir else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)

    package_chartqa(out_dir, args.split)


if __name__ == "__main__":
    main()
