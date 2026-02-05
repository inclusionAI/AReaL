from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset, Features, Image as HFImage, Value


def _load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def package_shortest_path(dataset_dir: Path, out_dir: Path) -> None:
    dataset_jsonl = dataset_dir / "dataset.jsonl"
    if not dataset_jsonl.exists():
        raise FileNotFoundError(f"dataset.jsonl not found: {dataset_jsonl}")

    out_dir.mkdir(parents=True, exist_ok=True)

    conditions = {
        "text": "shortest_path_text.parquet",
        "image": "shortest_path_image.parquet",
        "image_text": "shortest_path_image_text.parquet",
    }

    records_by_cond = {k: [] for k in conditions}

    for rec in _load_jsonl(dataset_jsonl):
        cond = rec.get("condition")
        if cond not in records_by_cond:
            continue
        image_rel = rec.get("image_path")
        if not image_rel:
            continue
        image_path = dataset_dir / image_rel
        with image_path.open("rb") as imf:
            b = imf.read()

        answer = rec.get("ground_truth", {}).get("path", "")
        records_by_cond[cond].append(
            {
                "case_id": int(rec.get("case_id", 0)),
                "base_graph_id": int(rec.get("base_graph_id", 0)),
                "level_nodes": int(rec.get("level_nodes", 0)),
                "prompt": rec.get("prompt", ""),
                "answer": str(answer),
                "image": {"bytes": b, "path": None},
            }
        )

    features = Features(
        {
            "case_id": Value("int64"),
            "base_graph_id": Value("int64"),
            "level_nodes": Value("int64"),
            "prompt": Value("string"),
            "answer": Value("string"),
            "image": HFImage(),
        }
    )

    for cond, filename in conditions.items():
        out_path = out_dir / filename
        ds = Dataset.from_list(records_by_cond[cond], features=features)
        ds.to_parquet(str(out_path))
        print(f"Saved parquet: {out_path} ({len(ds)})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Package shortest-path dataset to separate parquet files.")
    parser.add_argument("--dataset_dir", type=str, default="vlm_sp_unique_dataset", help="Dataset directory containing dataset.jsonl and images.")
    parser.add_argument("--out_dir", type=str, default="vlm_sp_unique_dataset", help="Output directory for parquet files.")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    package_shortest_path(dataset_dir, out_dir)


if __name__ == "__main__":
    main()
