import os
import argparse
from pathlib import Path

from tqdm import tqdm
from datasets import load_dataset, Dataset, Features, Value, Sequence, Image as HFImage


def package_mapeval_visual(out_dir: Path) -> Path:
    """Package MapEval-Visual dataset with images into parquet format."""
    out_parquet = out_dir / "mapeval_visual.parquet"

    dataset_root = Path("/storage/openpsi/data/lcy_image_edit/MapEval-Visual")

    print("Loading MapEval-Visual dataset from local disk...")
    ds = load_dataset(str(dataset_root))["test"]

    examples = []
    print("Reading local images and packaging dataset...")

    for i, item in enumerate(tqdm(ds, desc="Processing")):
        context = item["context"]  # e.g. "Vdata/ID_308.PNG"
        image_path = dataset_root / context

        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            continue

        with open(image_path, "rb") as f:
            image_bytes = f.read()

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
        default="/storage/openpsi/data/lcy_image_edit/",
        help="Output directory.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    package_mapeval_visual(out_dir)


if __name__ == "__main__":
    main()