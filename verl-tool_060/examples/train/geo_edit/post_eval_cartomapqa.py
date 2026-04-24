#!/usr/bin/env python3
"""Post-evaluation for CartoMapQA results from run_ood_eval.sh.

Reads the trajectory jsonl files and computes paper-aligned metrics
using geo_edit's extractors and metrics.

Usage:
    python post_eval_cartomapqa.py --results_dir /path/to/results/RUN_NAME
    python post_eval_cartomapqa.py --results_dir /path/to/results/RUN_NAME --tasks carto_rle carto_srn
"""

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List

from geo_edit.evaluation.cartomapqa.extractors import extract_structured
from geo_edit.evaluation.cartomapqa.metrics import (
    binary_prf1,
    exact_match_accuracy,
    mml_match,
    name_listing_prf1,
    normalize_route,
    regression_metrics,
    route_eval,
)


TASK_MAP = {
    "carto_mfs": "cartomapqa_mfs",
    "carto_mml": "cartomapqa_mml",
    "carto_mtmf": "cartomapqa_mtmf",
    "carto_rle": "cartomapqa_rle",
    "carto_srn": "cartomapqa_srn",
    "carto_stmf_counting": "cartomapqa_stmf_counting",
    "carto_stmf_name_listing": "cartomapqa_stmf_name_listing",
    "carto_stmf_presence": "cartomapqa_stmf_presence",
    "mapeval_visual": "mapeval_visual",
}


def load_records(jsonl_path: str) -> List[dict]:
    records = []
    with open(jsonl_path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def eval_mfs(records):
    true_labels, pred_labels = [], []
    for r in records:
        pred = extract_structured("cartomapqa_mfs", r["output"])
        gt = str(r["gts"]).strip().upper()
        true_labels.append(gt)
        pred_labels.append(pred if pred else "?")
    return exact_match_accuracy(true_labels, pred_labels)


def eval_stmf_presence(records):
    true_labels, pred_labels = [], []
    for r in records:
        pred = extract_structured("cartomapqa_stmf_presence", r["output"])
        gt = str(r["gts"]).strip().lower()
        true_labels.append(gt)
        pred_labels.append(pred if pred else "unknown")
    return binary_prf1(true_labels, pred_labels)


def eval_stmf_counting(records):
    gt_vals, pred_vals = [], []
    for r in records:
        pred = extract_structured("cartomapqa_stmf_counting", r["output"])
        try:
            gt = int(str(r["gts"]).strip())
        except (ValueError, TypeError):
            gt = 0
        gt_vals.append(gt)
        pred_vals.append(pred if pred is not None else 0)
    metrics = regression_metrics(gt_vals, pred_vals)
    correct = sum(1 for g, p in zip(gt_vals, pred_vals) if g == p)
    exact_tol1 = sum(1 for g, p in zip(gt_vals, pred_vals) if abs(g - p) <= 1)
    metrics["accuracy_exact"] = correct / len(gt_vals) if gt_vals else 0.0
    metrics["accuracy_tol1"] = exact_tol1 / len(gt_vals) if gt_vals else 0.0
    return metrics


def eval_stmf_name_listing(records):
    all_prec, all_rec, all_f1 = [], [], []
    for r in records:
        pred_names = extract_structured("cartomapqa_stmf_name_listing", r["output"])
        gt = str(r["gts"]).strip()
        gt_names = [n.strip() for n in gt.split("\n") if n.strip()]
        if not pred_names:
            pred_names = []
        m = name_listing_prf1(gt_names, pred_names)
        all_prec.append(m["precision"])
        all_rec.append(m["recall"])
        all_f1.append(m["f1"])
    n = len(all_prec)
    return {
        "avg_precision": sum(all_prec) / n if n else 0.0,
        "avg_recall": sum(all_rec) / n if n else 0.0,
        "avg_f1": sum(all_f1) / n if n else 0.0,
    }


def eval_mml(records):
    total = correct = 0
    for r in records:
        pred_data = extract_structured("cartomapqa_mml", r["output"])
        gt = str(r["gts"]).strip()
        try:
            gt_data = json.loads(gt)
        except (ValueError, TypeError):
            gt_data = {}
        total += 1
        if pred_data and isinstance(gt_data, dict):
            if mml_match(gt_data.get("road_1", ""), gt_data.get("road_2", ""),
                         pred_data.get("road_1", ""), pred_data.get("road_2", "")):
                correct += 1
    return {"accuracy": correct / total if total else 0.0, "correct": correct, "total": total}


def eval_rle(records):
    groups = defaultdict(lambda: {"gt": [], "pred": []})
    for r in records:
        pred_data = extract_structured("cartomapqa_rle", r["output"])
        gt_str = str(r["gts"]).strip()
        is_feet = "ft" in gt_str.lower() or "feet" in gt_str.lower()
        unit = "feet" if is_feet else "meters"
        gt_match = re.search(r"([-+]?\d[\d,]*(?:\.\d+)?)", gt_str)
        if gt_match and pred_data:
            gt_val = float(gt_match.group(1).replace(",", ""))
            pred_val = pred_data["value"]
            groups[unit]["gt"].append(gt_val)
            groups[unit]["pred"].append(pred_val)

    result = {}
    all_gt, all_pred = [], []
    for unit, data in sorted(groups.items()):
        result[unit] = regression_metrics(data["gt"], data["pred"])
        result[unit]["count"] = len(data["gt"])
        all_gt.extend(data["gt"])
        all_pred.extend(data["pred"])
    if all_gt:
        result["overall"] = regression_metrics(all_gt, all_pred)
        result["overall"]["count"] = len(all_gt)
    return result


def eval_srn(records):
    results = []
    for r in records:
        pred_route = extract_structured("cartomapqa_srn", r["output"])
        gt_str = str(r["gts"]).strip()
        gt_route = [item.strip() for item in gt_str.replace("[", "").replace("]", "").split(",")]
        if not pred_route:
            pred_route = []
        gt_norm = normalize_route(gt_route)
        pred_norm = normalize_route(pred_route)
        is_success, correct_steps = route_eval(gt_norm, pred_norm)
        step_acc = max(0.0, (correct_steps - 1) / (len(gt_norm) - 1)) if len(gt_norm) > 1 else 0.0
        results.append({"is_success": is_success, "step_accuracy": step_acc})

    n = len(results)
    return {
        "shortest_path_success_rate": sum(1 for r in results if r["is_success"]) / n if n else 0.0,
        "avg_step_accuracy": sum(r["step_accuracy"] for r in results) / n if n else 0.0,
        "total": n,
    }


def eval_mtmf(records):
    count_gt, count_pred = [], []
    all_prec, all_rec, all_f1 = [], [], []
    for r in records:
        pred_data = extract_structured("cartomapqa_mtmf", r["output"])
        gt_str = str(r["gts"]).strip()
        try:
            gt_data = json.loads(gt_str)
        except (ValueError, TypeError):
            gt_data = {}
        if not pred_data:
            pred_data = {}
        r_names_gt, r_names_pred = [], []
        for poi_type, gt_info in gt_data.items():
            pred_info = pred_data.get(poi_type, {})
            count_gt.append(gt_info.get("true_count", gt_info.get("count", 0)))
            count_pred.append(pred_info.get("count", 0))
            r_names_gt.extend(gt_info.get("true_names", gt_info.get("names", [])))
            r_names_pred.extend([n for n in pred_info.get("names", []) if n.strip()])
        m = name_listing_prf1(r_names_gt, r_names_pred)
        all_prec.append(m["precision"])
        all_rec.append(m["recall"])
        all_f1.append(m["f1"])

    n = len(all_prec)
    return {
        "counting": regression_metrics(count_gt, count_pred),
        "name_listing": {
            "avg_precision": sum(all_prec) / n if n else 0.0,
            "avg_recall": sum(all_rec) / n if n else 0.0,
            "avg_f1": sum(all_f1) / n if n else 0.0,
        },
    }


def eval_mapeval_visual(records):
    correct = total = 0
    for r in records:
        gt = str(r["gts"]).strip()
        pred = str(r.get("output", "")).strip()
        answer_m = re.search(r"<answer>(.*?)</answer>", pred, re.DOTALL | re.IGNORECASE)
        if answer_m:
            pred_ans = answer_m.group(1).strip()
        else:
            pred_ans = pred
        pred_norm = re.sub(r"[^\d]", "", pred_ans)
        gt_norm = re.sub(r"[^\d]", "", gt)
        total += 1
        if pred_norm == gt_norm:
            correct += 1
    return {"accuracy": correct / total if total else 0.0, "correct": correct, "total": total}


EVALUATORS = {
    "cartomapqa_mfs": eval_mfs,
    "cartomapqa_mml": eval_mml,
    "cartomapqa_mtmf": eval_mtmf,
    "cartomapqa_rle": eval_rle,
    "cartomapqa_srn": eval_srn,
    "cartomapqa_stmf_counting": eval_stmf_counting,
    "cartomapqa_stmf_name_listing": eval_stmf_name_listing,
    "cartomapqa_stmf_presence": eval_stmf_presence,
    "mapeval_visual": eval_mapeval_visual,
}


def format_metrics(metrics, indent=0):
    lines = []
    prefix = "  " * indent
    for k, v in metrics.items():
        if isinstance(v, dict):
            lines.append(f"{prefix}{k}:")
            lines.append(format_metrics(v, indent + 1))
        elif isinstance(v, float):
            lines.append(f"{prefix}{k}: {v:.4f}")
        else:
            lines.append(f"{prefix}{k}: {v}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    all_results = {}

    for dir_name in sorted(os.listdir(args.results_dir)):
        clean_name = dir_name.replace("ood_", "").replace("notool_", "")
        if clean_name not in TASK_MAP:
            continue
        if args.tasks and clean_name not in args.tasks:
            continue

        task_name = TASK_MAP[clean_name]
        jsonl_path = os.path.join(args.results_dir, dir_name, "0.jsonl")
        if not os.path.exists(jsonl_path):
            print(f"SKIP {dir_name}: no 0.jsonl")
            continue

        records = load_records(jsonl_path)
        evaluator = EVALUATORS.get(task_name)
        if not evaluator:
            print(f"SKIP {dir_name}: no evaluator for {task_name}")
            continue

        print(f"\n{'='*60}")
        print(f"  {clean_name} ({task_name}): {len(records)} samples")
        print(f"{'='*60}")

        metrics = evaluator(records)
        all_results[clean_name] = metrics
        print(format_metrics(metrics))

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nSaved to {args.output}")

    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    for task, metrics in all_results.items():
        if isinstance(metrics, dict):
            acc = metrics.get("accuracy", metrics.get("accuracy_exact", metrics.get("avg_f1", None)))
            if acc is not None:
                print(f"  {task}: {acc:.4f}")
            elif "overall" in metrics:
                overall = metrics["overall"]
                if "rmse" in overall:
                    print(f"  {task}: RMSE={overall['rmse']:.4f}, MAE={overall['mae']:.4f}")
                elif "accuracy" in overall:
                    print(f"  {task}: {overall['accuracy']:.4f}")
            elif "shortest_path_success_rate" in metrics:
                print(f"  {task}: success={metrics['shortest_path_success_rate']:.4f}, step_acc={metrics['avg_step_accuracy']:.4f}")
            elif "counting" in metrics:
                print(f"  {task}: counting_RMSE={metrics['counting'].get('rmse',0):.4f}, naming_F1={metrics['name_listing'].get('avg_f1',0):.4f}")


if __name__ == "__main__":
    main()
