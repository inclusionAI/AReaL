"""Unified CartoMapQA evaluation CLI.

Usage:
    python -m geo_edit.evaluation.cartomapqa.evaluate \
        --task cartomapqa_stmf_counting \
        --result_path /path/to/inference_output \
        --output_path /path/to/eval_output

    # With LLM judge fallback for extraction failures:
    python -m geo_edit.evaluation.cartomapqa.evaluate \
        --task cartomapqa_rle \
        --result_path ./output/rle \
        --output_path ./eval/rle \
        --use_judge --api_key $OPENAI_API_KEY

    # SRN requires topology data:
    python -m geo_edit.evaluation.cartomapqa.evaluate \
        --task cartomapqa_srn \
        --result_path ./output/srn \
        --output_path ./eval/srn \
        --road_conj_path ./Dataset/road_conjunction \
        --road_dict_path ./Dataset/road_dict
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from geo_edit.datasets.task_registry import get_dataset_spec
from geo_edit.evaluation.cartomapqa.extractors import extract_structured
from geo_edit.evaluation.cartomapqa.metrics import (
    binary_prf1,
    check_valid_route,
    exact_match_accuracy,
    mml_match,
    name_listing_prf1,
    normalize_route,
    regression_metrics,
    route_eval,
    srn_metrics,
)
from geo_edit.utils.io_utils import iter_meta_info_files, load_records
from geo_edit.utils.text_utils import get_final_prediction


SUPPORTED_TASKS = [
    "cartomapqa_mfs",
    "cartomapqa_stmf_presence",
    "cartomapqa_stmf_counting",
    "cartomapqa_stmf_name_listing",
    "cartomapqa_mtmf",
    "cartomapqa_rle",
    "cartomapqa_mml",
    "cartomapqa_srn",
]


def _get_output_text(record: dict) -> str:
    output_text = record.get("output_text", "")
    if isinstance(output_text, list):
        return output_text[-1] if output_text else ""
    return str(output_text)


def _try_judge_extract(
    question: str,
    output_text: str,
    task_name: str,
    judge: Any,
) -> Optional[str]:
    prompt = (
        f"Extract the final answer from the model's response.\n\n"
        f"Question: {question}\n"
        f"Model response: {output_text}\n\n"
        f"Return ONLY the extracted answer, nothing else."
    )
    try:
        if judge.api_mode == "chat":
            resp = judge.client.chat.completions.create(
                model=judge.model,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content.strip()
        resp = judge.client.responses.create(
            model=judge.model,
            input=[
                {"role": "user", "content": [{"type": "input_text", "text": prompt}]}
            ],
        )
        return (resp.output_text or "").strip() or None
    except Exception:
        return None


def _extract_with_fallback(
    task_name: str,
    text: str,
    record: dict,
    judge: Any = None,
) -> Any:
    result = extract_structured(task_name, text)
    if result is None and judge is not None:
        raw = _try_judge_extract(record.get("question", ""), text, task_name, judge)
        if raw:
            result = extract_structured(task_name, raw)
    return result


def _llm_judge_correctness(record: dict, output_text: str, judge: Any) -> bool:
    question = record.get("question", "")
    gt = record.get("answer", "")
    if isinstance(gt, list):
        gt = ", ".join(str(x) for x in gt)
    else:
        gt = str(gt)
    try:
        score = judge.judge_correctness(question, gt, output_text)
        return score == "1"
    except Exception:
        return False


def _parallel_judge(
    items: List[tuple],
    judge: Any,
    max_workers: int = 64,
) -> set:
    corrected: set = set()

    def _judge_one(item: tuple):
        idx, record, text = item
        if _llm_judge_correctness(record, text, judge):
            return idx
        return None

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_judge_one, it): it[0] for it in items}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                corrected.add(result)
    return corrected


# ── Per-task evaluation logic ────────────────────────────────────────────


def _evaluate_mfs(records: List[dict], **kwargs) -> Dict:
    judge = kwargs.get("judge")
    true_labels, pred_labels = [], []
    details = []
    for r in records:
        text = _get_output_text(r)
        pred = _extract_with_fallback("cartomapqa_mfs", text, r, judge)
        gt = str(r.get("answer", "")).strip().upper()
        if pred is None:
            pred = "?"
        true_labels.append(gt)
        pred_labels.append(pred)
        details.append(
            {
                "id": r["_id"],
                "gt": gt,
                "pred": pred,
                "result": 1.0 if gt == pred else 0.0,
            }
        )
    summary = exact_match_accuracy(true_labels, pred_labels)
    return {"summary": summary, "details": details}


def _evaluate_stmf_presence(records: List[dict], **kwargs) -> Dict:
    judge = kwargs.get("judge")
    true_labels, pred_labels = [], []
    details = []
    for r in records:
        text = _get_output_text(r)
        pred = _extract_with_fallback("cartomapqa_stmf_presence", text, r, judge)
        gt = str(r.get("answer", "")).strip().lower()
        if pred is None:
            pred = "unknown"
        true_labels.append(gt)
        pred_labels.append(pred)
        details.append(
            {
                "id": r["_id"],
                "gt": gt,
                "pred": pred,
                "result": 1.0 if gt == pred else 0.0,
            }
        )
    summary = binary_prf1(true_labels, pred_labels)
    return {"summary": summary, "details": details}


def _evaluate_stmf_counting(records: List[dict], **kwargs) -> Dict:
    judge = kwargs.get("judge")
    gt_vals, pred_vals = [], []
    details = []
    for r in records:
        text = _get_output_text(r)
        pred = _extract_with_fallback("cartomapqa_stmf_counting", text, r, judge)
        try:
            gt = int(r.get("answer", 0))
        except (ValueError, TypeError):
            gt = 0
        if pred is None:
            pred = 0
        gt_vals.append(gt)
        pred_vals.append(pred)
        details.append(
            {
                "id": r["_id"],
                "gt": gt,
                "pred": pred,
                "result": 1.0 if gt == pred else 0.0,
            }
        )
    summary = regression_metrics(gt_vals, pred_vals)
    correct = sum(1 for g, p in zip(gt_vals, pred_vals) if g == p)
    summary["accuracy"] = correct / len(gt_vals) if gt_vals else 0.0
    return {"summary": summary, "details": details}


def _evaluate_stmf_name_listing(records: List[dict], **kwargs) -> Dict:
    judge = kwargs.get("judge")
    all_prec, all_rec, all_f1 = [], [], []
    details = []
    for r in records:
        text = _get_output_text(r)
        pred_names = _extract_with_fallback(
            "cartomapqa_stmf_name_listing", text, r, judge
        )
        gt_raw = r.get("answer", [])
        gt_names = gt_raw if isinstance(gt_raw, list) else [str(gt_raw)]
        gt_names = [n.strip() for n in gt_names if n.strip()]
        m = name_listing_prf1(gt_names, pred_names)
        all_prec.append(m["precision"])
        all_rec.append(m["recall"])
        all_f1.append(m["f1"])
        details.append({"id": r["_id"], "gt": gt_names, "pred": pred_names, **m})
    n = len(all_prec)
    summary = {
        "avg_precision": sum(all_prec) / n if n else 0.0,
        "avg_recall": sum(all_rec) / n if n else 0.0,
        "avg_f1": sum(all_f1) / n if n else 0.0,
    }
    return {"summary": summary, "details": details}


def _evaluate_mtmf(records: List[dict], **kwargs) -> Dict:
    judge = kwargs.get("judge")
    count_gt, count_pred = [], []
    all_prec, all_rec, all_f1 = [], [], []
    details = []

    for r in records:
        text = _get_output_text(r)
        pred_data = _extract_with_fallback("cartomapqa_mtmf", text, r, judge)
        gt_raw = r.get("answer", {})
        if isinstance(gt_raw, str):
            try:
                gt_raw = json.loads(gt_raw)
            except (json.JSONDecodeError, TypeError):
                gt_raw = {}

        if pred_data is None:
            pred_data = {}

        r_count_gt, r_count_pred = [], []
        r_names_gt, r_names_pred = [], []
        for poi_type, gt_info in gt_raw.items():
            pred_info = pred_data.get(poi_type, {})
            r_count_gt.append(gt_info.get("true_count", gt_info.get("count", 0)))
            r_count_pred.append(pred_info.get("count", 0))
            r_names_gt.extend(gt_info.get("true_names", gt_info.get("names", [])))
            pred_names = [n for n in pred_info.get("names", []) if n.strip()]
            r_names_pred.extend(pred_names)

        count_gt.extend(r_count_gt)
        count_pred.extend(r_count_pred)
        m = name_listing_prf1(r_names_gt, r_names_pred)
        all_prec.append(m["precision"])
        all_rec.append(m["recall"])
        all_f1.append(m["f1"])
        details.append({"id": r["_id"]})

    n = len(all_prec)
    summary = {
        "counting": regression_metrics(count_gt, count_pred),
        "name_listing": {
            "avg_precision": sum(all_prec) / n if n else 0.0,
            "avg_recall": sum(all_rec) / n if n else 0.0,
            "avg_f1": sum(all_f1) / n if n else 0.0,
        },
    }
    return {"summary": summary, "details": details}


def _evaluate_rle(records: List[dict], **kwargs) -> Dict:
    judge = kwargs.get("judge")
    groups: Dict[str, Dict[str, List]] = defaultdict(lambda: {"gt": [], "pred": []})
    details = []

    for r in records:
        text = _get_output_text(r)
        pred_data = _extract_with_fallback("cartomapqa_rle", text, r, judge)
        gt_raw = r.get("answer", "")
        meta = r.get("meta_info_extra", {}) or {}
        difficulty = meta.get("difficulty", meta.get("Difficulty", "Simple"))
        measure = meta.get("measure", meta.get("Measure", ""))
        if not measure:
            rid = str(r.get("_id", ""))
            if rid.endswith("_feet"):
                measure = "feet"
            elif rid.endswith("_meters"):
                measure = "meters"
            else:
                measure = "unknown"

        gt_val = None
        if isinstance(gt_raw, (int, float)):
            gt_val = float(gt_raw)
        elif isinstance(gt_raw, str):
            m = _extract_rle_value(gt_raw)
            gt_val = m

        pred_val = pred_data["value"] if pred_data else None

        if gt_val is not None and pred_val is not None:
            key = f"{difficulty}_{measure}"
            groups[key]["gt"].append(gt_val)
            groups[key]["pred"].append(pred_val)

        details.append(
            {
                "id": r["_id"],
                "gt": gt_val,
                "pred": pred_val,
                "difficulty": difficulty,
                "measure": measure,
            }
        )

    summary: Dict[str, Any] = {}
    all_gt, all_pred = [], []
    for group_key, data in sorted(groups.items()):
        summary[group_key] = regression_metrics(data["gt"], data["pred"])
        all_gt.extend(data["gt"])
        all_pred.extend(data["pred"])
    if all_gt:
        summary["overall"] = regression_metrics(all_gt, all_pred)
    return {"summary": summary, "details": details}


def _extract_rle_value(text: str) -> Optional[float]:
    import re

    m = re.search(r"([-+]?\d[\d,]*(?:\.\d+)?)", text)
    if m:
        return float(m.group(1).replace(",", ""))
    return None


def _evaluate_mml(records: List[dict], **kwargs) -> Dict:
    judge = kwargs.get("judge")
    details = []
    zoom_groups: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "total": 0}
    )
    color_groups: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "total": 0}
    )
    total = correct = 0

    for r in records:
        text = _get_output_text(r)
        pred_data = _extract_with_fallback("cartomapqa_mml", text, r, judge)
        gt_raw = r.get("answer", {})
        if isinstance(gt_raw, str):
            try:
                gt_raw = json.loads(gt_raw)
            except (json.JSONDecodeError, TypeError):
                gt_raw = {}
        meta = r.get("meta_info_extra", {})
        zoom = str(meta.get("zoom_level", "unknown"))
        color = meta.get("Marker_color", meta.get("marker_color", "unknown"))

        is_correct = False
        if pred_data and isinstance(gt_raw, dict):
            is_correct = mml_match(
                gt_raw.get("road_1", ""),
                gt_raw.get("road_2", ""),
                pred_data.get("road_1", ""),
                pred_data.get("road_2", ""),
            )

        total += 1
        if is_correct:
            correct += 1
        zoom_groups[zoom]["total"] += 1
        color_groups[color]["total"] += 1
        if is_correct:
            zoom_groups[zoom]["correct"] += 1
            color_groups[color]["correct"] += 1

        details.append(
            {
                "id": r["_id"],
                "gt": gt_raw,
                "pred": pred_data,
                "result": 1.0 if is_correct else 0.0,
                "zoom_level": zoom,
                "marker_color": color,
            }
        )

    summary: Dict[str, Any] = {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
    }
    summary["per_zoom"] = {
        z: {"accuracy": d["correct"] / d["total"] if d["total"] else 0.0, **d}
        for z, d in sorted(zoom_groups.items())
    }
    summary["per_color"] = {
        c: {"accuracy": d["correct"] / d["total"] if d["total"] else 0.0, **d}
        for c, d in sorted(color_groups.items())
    }
    return {"summary": summary, "details": details}


def _evaluate_srn(records: List[dict], **kwargs) -> Dict:
    judge = kwargs.get("judge")
    road_conj_path = kwargs.get("road_conj_path")
    road_dict_path = kwargs.get("road_dict_path")
    results = []
    details = []

    for r in records:
        text = _get_output_text(r)
        pred_route = _extract_with_fallback("cartomapqa_srn", text, r, judge)
        gt_raw = r.get("answer", "")
        meta = r.get("meta_info_extra", {})
        zoom_level = meta.get("zoom_level")
        record_id = r["_id"]

        if isinstance(gt_raw, str):
            gt_raw = gt_raw.replace("[", "").replace("]", "")
            gt_route = [item.strip() for item in gt_raw.split(",")]
        elif isinstance(gt_raw, list):
            gt_route = [str(item).strip() for item in gt_raw]
        else:
            gt_route = []

        if pred_route is None:
            pred_route = []

        gt_norm = normalize_route(gt_route)
        pred_norm = normalize_route(pred_route)

        is_success, correct_step_count = route_eval(gt_norm, pred_norm)
        step_accuracy = (
            (correct_step_count - 1) / (len(gt_norm) - 1) if len(gt_norm) > 1 else 0.0
        )
        step_accuracy = max(0.0, step_accuracy)

        is_connected = False
        if is_success:
            is_connected = True
        elif road_conj_path and road_dict_path:
            conj_file = os.path.join(road_conj_path, f"{record_id}.json")
            rd_file = os.path.join(road_dict_path, f"{record_id}.json")
            if os.path.exists(conj_file) and os.path.exists(rd_file):
                try:
                    with open(conj_file) as f:
                        conjunction = json.load(f)
                    with open(rd_file) as f:
                        road_dict = json.load(f)
                    origin = meta.get("origin", [])
                    dest = meta.get("destination", [])
                    is_connected = check_valid_route(
                        origin, dest, pred_norm, conjunction, road_dict
                    )
                except Exception:
                    is_connected = False

        entry = {
            "is_success": is_success,
            "step_accuracy": step_accuracy,
            "is_connected": is_connected,
            "zoom_level": zoom_level,
        }
        results.append(entry)
        details.append({"id": record_id, **entry, "gt": gt_norm, "pred": pred_norm})

    summary = srn_metrics(results)
    return {"summary": summary, "details": details}


EVALUATORS = {
    "cartomapqa_mfs": _evaluate_mfs,
    "cartomapqa_stmf_presence": _evaluate_stmf_presence,
    "cartomapqa_stmf_counting": _evaluate_stmf_counting,
    "cartomapqa_stmf_name_listing": _evaluate_stmf_name_listing,
    "cartomapqa_mtmf": _evaluate_mtmf,
    "cartomapqa_rle": _evaluate_rle,
    "cartomapqa_mml": _evaluate_mml,
    "cartomapqa_srn": _evaluate_srn,
}


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
    eval_path = os.path.join(output_path, "eval_result.jsonl")
    with open(eval_path, "w", encoding="utf-8") as f:
        for item in result["details"]:
            f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")

    summary_path = os.path.join(output_path, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result["summary"], f, indent=2, ensure_ascii=False, default=str)

    summary_txt_path = os.path.join(output_path, "summary.txt")
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write(_format_summary(result["summary"]))

    print(_format_summary(result["summary"]))


def _format_summary(summary: dict, indent: int = 0) -> str:
    lines = []
    prefix = "  " * indent
    for k, v in summary.items():
        if isinstance(v, dict):
            lines.append(f"{prefix}{k}:")
            lines.append(_format_summary(v, indent + 1))
        elif isinstance(v, float):
            lines.append(f"{prefix}{k}={v:.4f}")
        else:
            lines.append(f"{prefix}{k}={v}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Unified CartoMapQA evaluation.")
    parser.add_argument("--task", type=str, required=True, choices=SUPPORTED_TASKS)
    parser.add_argument("--result_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--road_conj_path", type=str, default=None)
    parser.add_argument("--road_dict_path", type=str, default=None)
    parser.add_argument("--use_judge", action="store_true")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--api_base", type=str, default=None)
    parser.add_argument("--judge_model", type=str, default="gpt-4o-mini")
    args = parser.parse_args()

    evaluator = EVALUATORS[args.task]
    records = _load_records_from_result_path(args.result_path)
    print(f"Loaded {len(records)} records for {args.task}")

    kwargs = {}
    if args.use_judge:
        from geo_edit.evaluation.openai_as_judge import OpenAIJudge

        kwargs["judge"] = OpenAIJudge(
            api_key=args.api_key,
            model=args.judge_model,
            api_base=args.api_base,
        )
    if args.task == "cartomapqa_srn":
        kwargs["road_conj_path"] = args.road_conj_path
        kwargs["road_dict_path"] = args.road_dict_path

    result = evaluator(records, **kwargs)
    _write_outputs(args.output_path, result)
    print(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    main()
