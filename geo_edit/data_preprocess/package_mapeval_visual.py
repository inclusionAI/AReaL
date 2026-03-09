"""Package MapEval-Visual dataset to parquet format.

Downloads the dataset and images from HuggingFace and packages them into a parquet file.

Usage:
    python geo_edit/data_preprocess/package_mapeval_visual.py --out_dir /path/to/output
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasets import Dataset, Features, Image as HFImage, Sequence, Value, load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm


def package_mapeval_visual(out_dir: Path) -> Path:
    """Package MapEval-Visual dataset with images into parquet format."""
    out_parquet = out_dir / "mapeval_visual_dataset.parquet"

    # Load dataset metadata from HuggingFace
    print("Loading MapEval-Visual dataset from HuggingFace...")
    ds = load_dataset("MapEval/MapEval-Visual")["test"]

    examples = []
    print("Downloading images and packaging dataset...")

    for i, item in enumerate(tqdm(ds, desc="Processing")):
        # Download image from HuggingFace repo
        context = item["context"]  # e.g., "Vdata/ID_308.PNG"
        try:
            image_path = hf_hub_download(
                repo_id="MapEval/MapEval-Visual",
                filename=context,
                repo_type="dataset",
            )
            with open(image_path, "rb") as f:
                image_bytes = f.read()
        except Exception as e:
            print(f"Warning: Failed to download image {context}: {e}")
            continue

        # Extract ID from context (e.g., "Vdata/ID_308.PNG" -> "ID_308")
        image_id = os.path.splitext(os.path.basename(context))[0]

        examples.append(
            {
                "id": i,
                "image_id": image_id,
                "question": item["question"],
                "options": item["options"],
                "answer": item["answer"],
                "classification": item["classification"],
                "url": item["url"],
                "image": {"bytes": image_bytes, "path": None},
            }
        )

    # Define features
    features = Features(
        {
            "id": Value("int64"),
            "image_id": Value("string"),
            "question": Value("string"),
            "options": Sequence(Value("string")),
            "answer": Value("int64"),
            "classification": Value("string"),
            "url": Value("string"),
            "image": HFImage(),
        }
    )

    # Create dataset and save to parquet
    dataset = Dataset.from_list(examples, features=features)
    dataset.to_parquet(str(out_parquet))
    print(f"Saved parquet: {out_parquet}")
    print(f"Total examples: {len(examples)}")

    return out_parquet


def main() -> None:
    parser = argparse.ArgumentParser(description="Package MapEval-Visual dataset to parquet.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. Defaults to current directory.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve() if args.out_dir else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)

    package_mapeval_visual(out_dir)


if __name__ == "__main__":
    main()
