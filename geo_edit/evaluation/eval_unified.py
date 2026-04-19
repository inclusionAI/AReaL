"""Unified evaluation script for all datasets.

Supports: visual_probe, map_trace, reason_map, reason_map_plus, mm_mapqa
Rule-based evaluation first, then optional LLM judge fallback for incorrect answers.

Usage:
    python -m geo_edit.evaluation.eval_unified \
        --dataset_name visual_probe \
        --result_path /path/to/inference_output \
        --output_path /path/to/eval_output

    # With LLM judge fallback
    python -m geo_edit.evaluation.eval_unified \
        --dataset_name reason_map \
        --result_path /path/to/inference_output \
        --output_path /path/to/eval_output \
        --use_judge
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from geo_edit.utils.io_utils import iter_meta_info_files, load_records
from geo_edit.utils.stats import get_output_tokens_total, get_input_tokens_total
from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"</think>\s*(.*)", re.DOTALL | re.IGNORECASE)
_YES_VARIANTS = frozenset({"yes", "yeah", "yep", "yup", "true", "correct"})
_NO_VARIANTS = frozenset({"no", "nope", "nah", "false", "incorrect"})
_NA_VARIANTS = frozenset({"n/a", "na", "none", "not available", "not applicable"})

SUPPORTED_DATASETS = [
    "visual_probe", "map_trace", "reason_map", "reason_map_plus", "mm_mapqa",
]


def _extract_prediction(text: str) -> str:
    m = _ANSWER_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _THINK_RE.search(text)
    if m and m.group(1).strip():
        return m.group(1).strip()
    return text.strip()


def _normalize(text: str) -> str:
    text = re.sub(r"\\boxed\{(.*?)\}", r"\1", text)
    return text.strip().lower().rstrip(".")


def _parse_number(s: str):
    try:
        return float(s.strip().replace(",", ""))
    except (ValueError, TypeError):
        return None


def _parse_items_as_set(text: str):
    items = [i.strip() for i in text.split(",") if i.strip()]
    return frozenset(items) if items else None


def _parse_range(text: str):
    m = re.match(r"^([\d.]+)%?\s*[-–]\s*([\d.]+)%?$", text.strip())
    if m:
        try:
            return (float(m.group(1)), float(m.group(2)))
        except ValueError:
            return None
    return None


def _rule_score_generic(prediction: str, ground_truth: str) -> float:
    pred = _normalize(prediction)
    gt = _normalize(ground_truth)
    if pred == gt:
        return 1.0
    if gt in _YES_VARIANTS and pred in _YES_VARIANTS:
        return 1.0
    if gt in _NO_VARIANTS and pred in _NO_VARIANTS:
        return 1.0
    if gt in _NA_VARIANTS and pred in _NA_VARIANTS:
        return 1.0
    if "," in gt:
        gt_set, pred_set = _parse_items_as_set(gt), _parse_items_as_set(pred)
        if gt_set and pred_set and gt_set == pred_set:
            return 1.0
    gt_num, pred_num = _parse_number(gt), _parse_number(pred)
    if gt_num is not None and pred_num is not None and abs(gt_num - pred_num) < 1e-6:
        return 1.0
    gt_range, pred_range = _parse_range(gt), _parse_range(pred)
    if gt_range is not None and pred_range is not None and gt_range == pred_range:
        return 1.0
    return 0.0


def _rule_score_map_trace(output_str: str, ground_truth: str) -> dict:
    from geo_edit.evaluation.map_trace_verifier import map_trace_score
    try:
        ndtw, is_success, reason = map_trace_score(output_str, ground_truth)
    except (ValueError, TypeError):
        return {"score": 0.0, "ndtw": None, "reason": "parse_error"}
    return {"score": 1.0 if is_success and ndtw <= 1.0 else 0.0, "ndtw": ndtw if is_success else None, "reason": reason}


def _rule_score_reason_map(output_str: str, record: dict) -> dict:
    from geo_edit.evaluation.reason_map_verifier import reason_map_score

    station_1 = record.get("station_1", "") or record.get("meta_info_extra", {}).get("station_1", "")
    station_2 = record.get("station_2", "") or record.get("meta_info_extra", {}).get("station_2", "")
    metro_raw = record.get("metro_data") or record.get("meta_info_extra", {}).get("metro_data", {})
    if isinstance(metro_raw, str):
        if metro_raw == "None":
            metro_raw = {}
        else:
            try:
                metro_raw = json.loads(metro_raw)
            except (json.JSONDecodeError, TypeError):
                metro_raw = {}
    metro_data = metro_raw if isinstance(metro_raw, dict) else {}

    if not station_1 or not station_2 or not metro_data:
        gt = str(record.get("answer", ""))
        prediction = _extract_prediction(output_str)
        score = _rule_score_generic(prediction, gt)
        return {"score": score, "reason": "fallback_generic"}

    try:
        score, reason = reason_map_score(output_str, str(station_1), str(station_2), metro_data)
    except Exception as e:
        return {"score": 0.0, "reason": f"error: {e}"}
    return {"score": score, "reason": reason}


def _rule_score_reason_map_plus(prediction: str, ground_truth: str, qtype: str) -> float:
    pred = _normalize(prediction)
    gt_str = str(ground_truth).strip()

    if qtype.startswith("TorF"):
        try:
            gt_norm = "yes" if int(gt_str) == 1 else "no"
        except (ValueError, TypeError):
            gt_norm = gt_str.lower()
        return 1.0 if pred == gt_norm else 0.0

    if qtype == "Counting1":
        mapping = {"a": "0", "b": "1", "c": "2", "d": "3"}
        pred_mapped = mapping.get(pred, pred)
        gt_mapped = mapping.get(gt_str.lower(), gt_str.lower())
        return 1.0 if pred_mapped == gt_mapped else 0.0

    return _rule_score_generic(prediction, ground_truth)


def _get_output_str(record: dict) -> str:
    output = record.get("output_text", "")
    if isinstance(output, list):
        output = output[-1] if output else ""
    return str(output)


def evaluate_record(record: dict, record_id: str, dataset_name: str) -> dict:
    output_str = _get_output_str(record)
    prediction = _extract_prediction(output_str)
    ground_truth = str(record.get("answer", record.get("label", "")))

    result = {
        "id": record_id,
        "prediction": prediction,
        "ground_truth": ground_truth,
        "judge_called": False,
        "judge_overturned": False,
    }

    if dataset_name == "map_trace":
        mr = _rule_score_map_trace(output_str, ground_truth)
        result["score"] = mr["score"]
        result["ndtw"] = mr.get("ndtw")
        result["reason"] = mr.get("reason")
        return result

    if dataset_name == "reason_map":
        mr = _rule_score_reason_map(output_str, record)
        result["score"] = mr["score"]
        result["reason"] = mr.get("reason")
        result["ground_truth"] = ground_truth or str(record.get("routes", ""))
        return result

    if dataset_name == "reason_map_plus":
        qtype = record.get("type", "") or record.get("meta_info_extra", {}).get("type", "")
        result["score"] = _rule_score_reason_map_plus(prediction, ground_truth, qtype)
        result["qtype"] = qtype
        return result

    result["score"] = _rule_score_generic(prediction, ground_truth)
    return result


def run_llm_judge(judge, results: list[dict], records_map: dict) -> list[dict]:
    failed = [(i, r) for i, r in enumerate(results) if r["score"] == 0.0 and r["prediction"]]
    if not failed:
        return results

    logger.info(f"Running LLM judge on {len(failed)} failed records...")

    def _judge_one(idx, result):
        rec = records_map.get(result["id"], {})
        question = str(rec.get("question", rec.get("question_long", "")))
        try:
            score = judge.judge_correctness(
                question=question,
                ground_truth=result["ground_truth"],
                prediction=result["prediction"],
            )
            is_correct = score == 1.0 if isinstance(score, (int, float)) else str(score).strip() == "1"
        except Exception as e:
            logger.warning(f"Judge failed for {result['id']}: {e}")
            is_correct = False
        return idx, is_correct

    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(_judge_one, i, r): (i, r) for i, r in failed}
        for future in as_completed(futures):
            idx, is_correct = future.result()
            results[idx]["judge_called"] = True
            if is_correct:
                results[idx]["score"] = 1.0
                results[idx]["judge_overturned"] = True

    overturned = sum(1 for r in results if r.get("judge_overturned"))
    logger.info(f"LLM judge overturned {overturned}/{len(failed)} to correct")
    return results


def main():
    parser = argparse.ArgumentParser(description="Unified evaluation for all datasets.")
    parser.add_argument("--dataset_name", type=str, required=True, choices=SUPPORTED_DATASETS)
    parser.add_argument("--result_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--use_judge", action="store_true")
    parser.add_argument("--judge_model", type=str, default="gpt-5-mini-2025-08-07")
    parser.add_argument("--judge_api_key", type=str, default=None)
    parser.add_argument("--judge_api_base", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    all_results = []
    records_map = {}

    for meta_path in iter_meta_info_files(args.result_path):
        record_id = os.path.basename(os.path.dirname(meta_path))
        for record in load_records(meta_path):
            records_map[record_id] = record
            result = evaluate_record(record, record_id, args.dataset_name)
            all_results.append(result)

    if args.use_judge and args.dataset_name != "map_trace":
        api_key = args.judge_api_key or os.environ.get("JUDGE_API_KEY")
        api_base = args.judge_api_base or os.environ.get("JUDGE_API_BASE")
        if api_key:
            from geo_edit.evaluation.openai_as_judge import OpenAIJudge
            judge = OpenAIJudge(api_key=api_key, model=args.judge_model, api_base=api_base)
            all_results = run_llm_judge(judge, all_results, records_map)
        else:
            logger.warning("--use_judge set but no JUDGE_API_KEY found, skipping LLM judge")

    total = len(all_results)
    correct = sum(1 for r in all_results if r["score"] > 0)
    judge_called = sum(1 for r in all_results if r.get("judge_called"))
    judge_overturned = sum(1 for r in all_results if r.get("judge_overturned"))
    accuracy = correct / total if total > 0 else 0.0

    with open(os.path.join(args.output_path, "eval_result.jsonl"), "w") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    lines = [
        f"Dataset: {args.dataset_name}",
        f"Total: {total}",
        f"Correct: {correct}",
        f"Accuracy: {accuracy:.4f} ({correct}/{total})",
    ]

    if judge_called:
        lines.append(f"LLM Judge called: {judge_called}")
        lines.append(f"LLM Judge overturned: {judge_overturned}")
        rule_correct = correct - judge_overturned
        lines.append(f"Rule-only accuracy: {rule_correct/total:.4f} ({rule_correct}/{total})")

    if args.dataset_name == "map_trace":
        ndtw_vals = [r["ndtw"] for r in all_results if r.get("ndtw") is not None]
        if ndtw_vals:
            ndtw_vals.sort()
            lines.append(f"Avg NDTW: {sum(ndtw_vals)/len(ndtw_vals):.4f}")
            lines.append(f"Median NDTW: {ndtw_vals[len(ndtw_vals)//2]:.4f}")

    if args.dataset_name == "reason_map_plus":
        qtype_stats = defaultdict(lambda: {"n": 0, "correct": 0})
        for r in all_results:
            qt = r.get("qtype", "unknown")
            qtype_stats[qt]["n"] += 1
            if r["score"] > 0:
                qtype_stats[qt]["correct"] += 1
        lines.append("\nPer Question Type:")
        for qt in sorted(qtype_stats):
            s = qtype_stats[qt]
            lines.append(f"  {qt}: {s['correct']}/{s['n']} ({s['correct']/s['n']:.4f})" if s["n"] else f"  {qt}: 0/0")

    summary = "\n".join(lines)
    with open(os.path.join(args.output_path, "summary.txt"), "w") as f:
        f.write(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
