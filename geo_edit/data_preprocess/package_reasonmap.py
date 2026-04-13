"""Package ReasonMap dataset to parquet format.

Merges FSCCS/ReasonMap-Plus and FSCCS/ReasonMap-Train into a single parquet,
excluding 'planning' type questions (open-ended, no ground truth answer).

The figure field in both datasets is a relative path (e.g. ./maps/portugal/lisboa.png).
You must provide --maps_dir pointing to the directory containing the maps/ folder.

Usage:
    python geo_edit/data_preprocess/package_reasonmap.py \
        --maps_dir /path/to/reasonmap_images \
        --out_dir /path/to/output
"""
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset, Features, Value, load_dataset
from datasets import Image as HFImage
from tqdm import tqdm

# Question types to keep (planning excluded — open-ended with no reference answer)
KEEP_TYPES = {"Counting1", "Counting2", "Counting3", "TorF1", "TorF2"}


def _load_and_filter(hf_path: str, split: str):
    """Load a HuggingFace dataset split and filter out planning questions."""
    ds = load_dataset(hf_path)[split]
    return [item for item in ds if item["type"] in KEEP_TYPES]


def package_reasonmap(out_dir: Path, maps_dir: Path) -> Path:
    """Package merged ReasonMap dataset into parquet format."""
    out_parquet = out_dir / "reasonmap_dataset.parquet"

    # Load both sources
    print("Loading ReasonMap-Train (train split)...")
    train_items = _load_and_filter("FSCCS/ReasonMap-Train", "train")
    print(f"  kept {len(train_items)} items (planning excluded)")

    print("Loading ReasonMap-Plus (test split)...")
    test_items = _load_and_filter("FSCCS/ReasonMap-Plus", "test")
    print(f"  kept {len(test_items)} items (planning excluded)")

    # Also grab ReasonMap-Plus train split
    print("Loading ReasonMap-Plus (train split)...")
    plus_train_items = _load_and_filter("FSCCS/ReasonMap-Plus", "train")
    print(f"  kept {len(plus_train_items)} items (planning excluded)")

    all_items = train_items + plus_train_items + test_items

    examples = []
    missing = 0
    print(f"Packaging {len(all_items)} items with map images...")

    for i, item in enumerate(tqdm(all_items, desc="Processing")):
        fig_rel = item["figure"].lstrip("./")
        image_path = maps_dir / fig_rel

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
        description="Package merged ReasonMap (Plus + Train) dataset to parquet."
    )
    parser.add_argument(
        "--maps_dir",
        type=str,
        required=True,
        help="Directory containing the maps/ folder with subway map images.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. Defaults to current directory.",
    )
    args = parser.parse_args()

    maps_dir = Path(args.maps_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)

    package_reasonmap(out_dir, maps_dir)


if __name__ == "__main__":
    main()
