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


def _path_to_annotated(image_path: str) -> str:
    p = image_path.replace("images_original", "images_annotated")
    p = p.replace("_orig.png", "_ann.png")
    return p


def _path_to_original(image_path: str) -> str:
    p = image_path.replace("images_annotated", "images_original")
    p = p.replace("_ann.png", "_orig.png")
    return p


def _answer_str(rec: dict) -> str:
    path = rec.get("ground_truth", {}).get("path", [])
    if isinstance(path, list):
        return ",".join(str(x) for x in path)
    return str(path)


def _connectivity_text(rec: dict) -> str:
    edges = rec.get("edges", [])
    pairs = []
    for e in edges:
        u = e.get("u")
        v = e.get("v")
        if u is None or v is None:
            continue
        a, b = (u, v) if u < v else (v, u)
        pairs.append((a, b))
    pairs = sorted(set(pairs))
    return ", ".join([f"({u},{v})" for u, v in pairs])


def _prompt_text(rec: dict) -> str:
    q = rec.get("query", {})
    s = q.get("source", "")
    t = q.get("target", "")
    edge_text = rec.get("edge_list_text", "")
    conn_text = _connectivity_text(rec)
    return (
        f"You are given a graph image with node labels. "
        f"Connected node pairs (undirected) are: {conn_text}. "
        f"The edge distances are provided as text in the format d(U,V)=W: {edge_text}. "
        f"Find the shortest path from {s} to {t}. "
        f"Output only the node sequence separated by ','."
    )


def _prompt_image(rec: dict) -> str:
    q = rec.get("query", {})
    s = q.get("source", "")
    t = q.get("target", "")
    conn_text = _connectivity_text(rec)
    return (
        f"You are given an annotated graph image. "
        f"Connected node pairs (undirected) are: {conn_text}. "
        f"Read edge distances from the image labels. "
        f"Find the shortest path from {s} to {t}. "
        f"Output only the node sequence separated by ','."
    )


def _prompt_image_text(rec: dict) -> str:
    q = rec.get("query", {})
    s = q.get("source", "")
    t = q.get("target", "")
    edge_text = rec.get("edge_list_text", "")
    conn_text = _connectivity_text(rec)
    return (
        f"You are given an annotated graph image and additional edge-distance text. "
        f"Connected node pairs (undirected) are: {conn_text}. "
        f"The distance text uses format d(U,V)=W: {edge_text}. "
        f"Find the shortest path from {s} to {t}. "
        f"Output only the node sequence separated by ','."
    )


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
        if cond not in ("text", "image"):
            continue
        case_id = int(rec.get("case_id", 0))
        base_graph_id = int(rec.get("base_graph_id", 0))
        level_nodes = int(rec.get("level_nodes", 0))
        answer = _answer_str(rec)

        if cond == "text":
            image_rel = _path_to_original(rec.get("image_path", ""))
            if image_rel:
                image_path = dataset_dir / image_rel
                if image_path.exists():
                    with image_path.open("rb") as imf:
                        b = imf.read()
                    records_by_cond["text"].append(
                        {
                            "case_id": case_id,
                            "base_graph_id": base_graph_id,
                            "level_nodes": level_nodes,
                            "prompt": _prompt_text(rec),
                            "answer": answer,
                            "image": {"bytes": b, "path": None},
                        }
                    )

            ann_rel = _path_to_annotated(rec.get("image_path", ""))
            if ann_rel:
                ann_path = dataset_dir / ann_rel
                if ann_path.exists():
                    with ann_path.open("rb") as imf:
                        b = imf.read()
                    records_by_cond["image_text"].append(
                        {
                            "case_id": case_id,
                            "base_graph_id": base_graph_id,
                            "level_nodes": level_nodes,
                            "prompt": _prompt_image_text(rec),
                            "answer": answer,
                            "image": {"bytes": b, "path": None},
                        }
                    )

        if cond == "image":
            image_rel = _path_to_annotated(rec.get("image_path", ""))
            if not image_rel:
                continue
            image_path = dataset_dir / image_rel
            if not image_path.exists():
                continue
            with image_path.open("rb") as imf:
                b = imf.read()
            records_by_cond["image"].append(
                {
                    "case_id": case_id,
                    "base_graph_id": base_graph_id,
                    "level_nodes": level_nodes,
                    "prompt": _prompt_image(rec),
                    "answer": answer,
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
