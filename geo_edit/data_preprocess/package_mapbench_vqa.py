from __future__ import annotations

import argparse
import json
import pickle
from io import BytesIO
from pathlib import Path

from datasets import Dataset, Features, Image as HFImage, Value


def _serialize_graph(pkl_path: Path) -> str:
    import networkx as nx

    with open(pkl_path, "rb") as f:
        G = pickle.load(f)
    data = nx.node_link_data(G)
    return json.dumps(data, default=str)


def package_mapbench_vqa(
    input_path: str | None,
    pkl_dir: str | None,
    out_path: str,
) -> None:
    from datasets import load_dataset

    if input_path:
        ds = load_dataset("parquet", data_files=input_path, split="train")
    else:
        ds = load_dataset("shuoxing/MapBench_VQA", split="test")

    pkl_root = Path(pkl_dir) if pkl_dir else None
    graph_cache: dict[str, str] = {}

    rows = []
    missing_graphs = set()
    for map_idx, item in enumerate(ds):
        image = item["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        buf = BytesIO()
        image.save(buf, format="JPEG", quality=95)
        image_bytes = buf.getvalue()

        queries = item["queries"]
        image_id = item["image_id"]
        map_class = item["map_class"]

        graph_json = ""
        if pkl_root:
            cache_key = f"{map_class}/{image_id}"
            if cache_key not in graph_cache:
                pkl_path = pkl_root / map_class / f"{image_id}.pkl"
                if pkl_path.exists():
                    graph_cache[cache_key] = _serialize_graph(pkl_path)
                else:
                    graph_cache[cache_key] = ""
                    missing_graphs.add(str(pkl_path))
            graph_json = graph_cache[cache_key]

        for idx, q in enumerate(queries):
            rows.append(
                {
                    "id": f"{image_id}_q{idx}",
                    "image": {"bytes": image_bytes, "path": None},
                    "start": q["start"],
                    "destination": q["destination"],
                    "answer": q["gpt_answer"],
                    "map_class": map_class,
                    "image_id": image_id,
                    "graph_json": graph_json,
                }
            )

        if (map_idx + 1) % 10 == 0:
            print(f"  processed {map_idx + 1}/{len(ds)} maps, {len(rows)} rows so far")

    if missing_graphs:
        print(f"WARNING: {len(missing_graphs)} pkl files not found, graph_json will be empty for those maps.")
        for p in sorted(missing_graphs)[:5]:
            print(f"  missing: {p}")

    features = Features(
        {
            "id": Value("string"),
            "image": HFImage(),
            "start": Value("string"),
            "destination": Value("string"),
            "answer": Value("string"),
            "map_class": Value("string"),
            "image_id": Value("string"),
            "graph_json": Value("string"),
        }
    )

    flat_ds = Dataset.from_list(rows, features=features)
    flat_ds.to_parquet(out_path)
    graphs_loaded = sum(1 for v in graph_cache.values() if v)
    print(f"Saved {len(rows)} rows (from {len(ds)} maps, {graphs_loaded} graphs) to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flatten shuoxing/MapBench_VQA (1 map -> N queries) to parquet with embedded MSSG graphs.",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        help="Local parquet/dataset path. If omitted, downloads from HuggingFace.",
    )
    parser.add_argument(
        "--pkl_dir",
        type=str,
        default=None,
        help="Path to taco-group/MapBench pkl/ directory containing MSSG graphs. "
        "If omitted, graph_json column will be empty (eval will skip graph-based metrics).",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default=None,
        help="Output parquet path.",
    )
    args = parser.parse_args()

    if args.out_path:
        out_path = args.out_path
    else:
        here = Path(__file__).resolve()
        repo_root = here.parents[3]
        out_dir = repo_root / "MapBench" / "MapBench_parquet"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / "MapBench_VQA.parquet")

    package_mapbench_vqa(args.input_path, args.pkl_dir, out_path)


if __name__ == "__main__":
    main()
