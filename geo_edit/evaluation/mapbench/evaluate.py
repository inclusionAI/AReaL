from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List

from geo_edit.evaluation.mapbench.extractors import extract_navigation_steps
from geo_edit.evaluation.mapbench.metrics import (
    load_graph_from_json,
    path_eval,
)
from geo_edit.utils.io_utils import iter_meta_info_files, load_records


def _get_output_text(record: dict) -> str:
    output_text = record.get("output_text", "")
    if isinstance(output_text, list):
        return output_text[-1] if output_text else ""
    return str(output_text)


def evaluate_mapbench(records: List[dict]) -> Dict:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    total = 0
    success_count = 0
    path_score_sum = 0.0
    failure_counts: Dict[int, int] = defaultdict(int)
    per_class: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"total": 0, "success": 0, "path_score_sum": 0.0}
    )
    details = []

    for r in records:
        total += 1
        text = _get_output_text(r)
        nav_steps = extract_navigation_steps(text)
        meta = r.get("meta_info_extra", {}) or {}
        start = r.get("start", meta.get("start", ""))
        destination = r.get("destination", meta.get("destination", ""))
        map_class = r.get("map_class", meta.get("map_class", "unknown"))
        graph_json = r.get("graph_json", meta.get("graph_json", ""))
        record_id = r.get("_id", "")

        entry = {
            "id": record_id,
            "map_class": map_class,
            "start": start,
            "destination": destination,
            "nav_steps": nav_steps,
            "success": 0,
            "path_score": 0.0,
            "failure_flag": 0,
        }

        try:
            G = load_graph_from_json(graph_json)
            flag, score = path_eval(G, nav_steps, start, destination, model)
        except Exception as e:
            flag, score = -5, 0.0
            entry["error"] = str(e)

        entry["failure_flag"] = flag
        if flag == 1:
            entry["success"] = 1
            entry["path_score"] = score
            success_count += 1
            path_score_sum += score
            per_class[map_class]["success"] += 1
            per_class[map_class]["path_score_sum"] += score
        else:
            failure_counts[flag] += 1

        per_class[map_class]["total"] += 1
        details.append(entry)

    summary: Dict[str, Any] = {
        "total": total,
        "success_count": success_count,
        "success_rate": success_count / total if total else 0.0,
        "avg_path_score": path_score_sum / success_count if success_count else 0.0,
        "failure_breakdown": dict(failure_counts),
    }

    per_class_summary = {}
    for cls, data in sorted(per_class.items()):
        n, s = data["total"], data["success"]
        per_class_summary[cls] = {
            "total": n,
            "success_count": s,
            "success_rate": s / n if n else 0.0,
            "avg_path_score": data["path_score_sum"] / s if s else 0.0,
        }
    summary["per_map_class"] = per_class_summary

    return {"summary": summary, "details": details}


def _load_records_from_result_path(result_path: str) -> List[dict]:
    records = []
    for meta_path in iter_meta_info_files(result_path):
        record_id = os.path.basename(os.path.dirname(meta_path))
        for record in load_records(meta_path):
            record["_id"] = record_id
            records.append(record)
    return records


def _write_outputs(output_path: str, result: Dict) -> None:
    os.makedirs(output_path, exist_ok=True)

    with open(
        os.path.join(output_path, "eval_result.jsonl"), "w", encoding="utf-8"
    ) as f:
        for item in result["details"]:
            f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")

    with open(os.path.join(output_path, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(result["summary"], f, indent=2, ensure_ascii=False, default=str)

    _print_summary(result["summary"])


def _print_summary(summary: dict) -> None:
    print(f"Total: {summary['total']}")
    print(f"Success: {summary['success_count']} ({summary['success_rate']:.3f})")
    print(f"Avg Path Score: {summary['avg_path_score']:.3f}")
    if summary.get("failure_breakdown"):
        print(f"Failures: {summary['failure_breakdown']}")
    print()
    for cls, data in summary.get("per_map_class", {}).items():
        print(
            f"  {cls}: success={data['success_rate']:.3f} ({data['success_count']}/{data['total']}), "
            f"path_score={data['avg_path_score']:.3f}"
        )


def main():
    parser = argparse.ArgumentParser(description="MapBench VQA evaluation.")
    parser.add_argument("--result_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    records = _load_records_from_result_path(args.result_path)
    print(f"Loaded {len(records)} records")

    result = evaluate_mapbench(records)
    _write_outputs(args.output_path, result)
    print(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    main()
