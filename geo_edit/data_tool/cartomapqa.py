from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

from datasets import Dataset, Features, Image as HFImage, Sequence, Value


def _default_cartomapqa_root() -> Path:
    here = Path(__file__).resolve()
    repo_root = here.parents[3]
    return repo_root / "CartoMapQA"


def package_mapnavigation(cartomapqa_root: Path, out_dir: Path) -> Path:
    json_dir = cartomapqa_root / "Dataset" / "CartoMapQA" / "MapNavigation" / "_routes"
    image_root = cartomapqa_root / "Dataset" / "CartoMapQA" / "MapNavigation" / "_maps"
    image_ext = ".png"
    out_parquet = out_dir / "MapNavigation_dataset.parquet"

    examples = []
    json_files = sorted(glob.glob(os.path.join(str(json_dir), "*.json")))

    for i, fp in enumerate(json_files):
        with open(fp, "r", encoding="utf-8") as f:
            obj = json.load(f)

        base = os.path.splitext(os.path.basename(fp))[0]
        image_path = image_root / f"{base}{image_ext}"
        with open(image_path, "rb") as imf:
            b = imf.read()

        obj["id"] = i
        obj["image"] = {"bytes": b, "path": None}
        examples.append(obj)

    features = Features(
        {
            "origin": Sequence(Value("float64")),
            "destination": Sequence(Value("float64")),
            "facing": Value("string"),
            "travel_mode": Value("string"),
            "route_directions": Value("string"),
            "route_node_id": Sequence(Value("int64")),
            "area": Value("string"),
            "zoom_level": Value("int64"),
            "id": Value("int64"),
            "image": HFImage(),
        }
    )

    ds = Dataset.from_list(examples, features=features)
    ds.to_parquet(str(out_parquet))
    print("Saved parquet:", out_parquet)
    return out_parquet


def _stmf_features(task_type: str) -> Features:
    if task_type == "presence":
        answer_feature = Value("string")
    elif task_type == "counting":
        answer_feature = Value("int64")
    elif task_type == "name_listing":
        answer_feature = Sequence(Value("string"))
    else:
        raise ValueError(f"Unknown STMF task type: {task_type}")

    return Features(
        {
            "id": Value("int64"),
            "question_key": Value("string"),
            "mf_type": Value("string"),
            "correct_answer": answer_feature,
            "poi_type": Value("string"),
            "city": Value("string"),
            "zoom_level": Value("int64"),
            "image_file": Value("string"),
            "image": HFImage(),
        }
    )


def _coerce_stmf_answer(task_type: str, answer):
    if task_type == "presence":
        return str(answer)
    if task_type == "counting":
        return int(answer)
    if task_type == "name_listing":
        return [str(x) for x in answer]
    raise ValueError(f"Unknown STMF task type: {task_type}")


def package_stmf(task_type: str, cartomapqa_root: Path, out_dir: Path) -> Path:
    base_dir = cartomapqa_root / "Dataset" / "CartoMapQA" / "MapFeatureUnderstanding"
    json_path = base_dir / "STMF_task" / f"{task_type}.json"
    image_root = base_dir / "_Maps"
    out_parquet = out_dir / f"STMF_{task_type}_dataset.parquet"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples = []
    for key in sorted(data.keys(), key=lambda k: int(k.split("_")[1])):
        obj = data[key]
        question_id = int(key.split("_")[1])
        image_path = image_root / obj["City"] / obj["Image"]
        with open(image_path, "rb") as imf:
            b = imf.read()

        correct_answer = _coerce_stmf_answer(task_type, obj["Correct answer"])
        examples.append(
            {
                "id": question_id,
                "question_key": key,
                "mf_type": obj["MF type"],
                "correct_answer": correct_answer,
                "poi_type": obj.get("POI type", ""),
                "city": obj.get("City", ""),
                "zoom_level": int(obj.get("Zoom level", 0)),
                "image_file": obj.get("Image", ""),
                "image": {"bytes": b, "path": None},
            }
        )

    features = _stmf_features(task_type)
    ds = Dataset.from_list(examples, features=features)
    ds.to_parquet(str(out_parquet))
    print("Saved parquet:", out_parquet)
    return out_parquet


def main() -> None:
    parser = argparse.ArgumentParser(description="Package CartoMapQA datasets to parquet.")
    parser.add_argument(
        "--task",
        type=str,
        default="stmf_all",
        choices=[
            "mapnavigation",
            "stmf_presence",
            "stmf_counting",
            "stmf_name_listing",
            "stmf_all",
        ],
    )
    parser.add_argument("--cartomapqa_root", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    cartomapqa_root = (
        Path(args.cartomapqa_root).resolve()
        if args.cartomapqa_root
        else _default_cartomapqa_root()
    )
    if not cartomapqa_root.exists():
        raise FileNotFoundError(f"CartoMapQA root not found: {cartomapqa_root}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else cartomapqa_root
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.task == "mapnavigation":
        package_mapnavigation(cartomapqa_root, out_dir)
        return

    if args.task == "stmf_all":
        for stmf_type in ("presence", "counting", "name_listing"):
            package_stmf(stmf_type, cartomapqa_root, out_dir)
        return

    if args.task.startswith("stmf_"):
        stmf_type = args.task.replace("stmf_", "", 1)
        package_stmf(stmf_type, cartomapqa_root, out_dir)
        return

    raise ValueError(f"Unknown task: {args.task}")


if __name__ == "__main__":
    main()
