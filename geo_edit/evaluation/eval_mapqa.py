"""Evaluation script for MapQA visual question answering task.

Extracts predictions from <answer> tags (with </think> fallback),
compares against ground truth using normalized string matching.
Reports overall accuracy and per-question-type breakdown.

Usage:
    python -m geo_edit.evaluation.eval_mapqa \
        --result_path /path/to/inference_output \
        --output_path /path/to/eval_output
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

from geo_edit.utils.io_utils import iter_meta_info_files, load_records
from geo_edit.utils.stats import get_output_tokens_total, get_input_tokens_total
from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"</think>\s*(.*)", re.DOTALL | re.IGNORECASE)


def _extract_prediction(text: str) -> str:
    m = _ANSWER_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _THINK_RE.search(text)
    if m and m.group(1).strip():
        return m.group(1).strip()
    return text.strip()


def _normalize(text: str) -> str:
    return text.strip().lower().rstrip(".")


def _score(prediction: str, ground_truth: str) -> float:
    pred = _normalize(prediction)
    gt = _normalize(ground_truth)
    if pred == gt:
        return 1.0
    try:
        if abs(float(pred) - float(gt)) < 1e-6:
            return 1.0
    except (ValueError, TypeError):
        pass
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MapQA predictions.")
    parser.add_argument("--result_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    eval_output_path = os.path.join(args.output_path, "eval_result.jsonl")
    summary_path = os.path.join(args.output_path, "summary.txt")

    total = 0
    correct = 0
    no_answer = 0
    output_tokens_sum = input_tokens_sum = 0.0
    output_tokens_count = input_tokens_count = 0

    qtype_stats: dict[str, dict] = defaultdict(lambda: {"total": 0, "correct": 0})

    with open(eval_output_path, "w", encoding="utf-8") as out_f:
        for meta_path in iter_meta_info_files(args.result_path):
            record_id = os.path.basename(os.path.dirname(meta_path))

            for record in load_records(meta_path):
                output_text = record.get("output_text", "")
                if isinstance(output_text, list):
                    output_text = output_text[-1] if output_text else ""
                output_str = str(output_text)

                ground_truth = str(record.get("answer", ""))
                qtype = str(record.get("meta_info_extra", {}).get("question_type", "unknown"))

                prediction = _extract_prediction(output_str)
                if not prediction:
                    no_answer += 1

                score = _score(prediction, ground_truth)
                total += 1
                if score > 0:
                    correct += 1

                qtype_stats[qtype]["total"] += 1
                if score > 0:
                    qtype_stats[qtype]["correct"] += 1

                ot = get_output_tokens_total(record)
                if ot is not None:
                    output_tokens_sum += ot
                    output_tokens_count += 1
                it = get_input_tokens_total(record)
                if it is not None:
                    input_tokens_sum += it
                    input_tokens_count += 1

                eval_item = {
                    "id": record_id,
                    "result": score,
                    "prediction": prediction,
                    "ground_truth": ground_truth,
                    "question_type": qtype,
                }
                out_f.write(json.dumps(eval_item, ensure_ascii=False) + "\n")

    accuracy = correct / total if total > 0 else 0.0

    lines = [
        f"Total: {total}",
        f"Correct: {correct}",
        f"No answer: {no_answer}",
        f"Accuracy: {accuracy:.4f} ({correct}/{total})",
        "",
    ]

    if output_tokens_count > 0:
        lines.append(f"Avg output tokens: {output_tokens_sum / output_tokens_count:.1f}")
    if input_tokens_count > 0:
        lines.append(f"Avg input tokens: {input_tokens_sum / input_tokens_count:.1f}")
    lines.append("")

    lines.append("=== Per Question Type ===")
    for qtype in sorted(qtype_stats.keys()):
        s = qtype_stats[qtype]
        acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
        lines.append(f"  {qtype}: {acc:.4f} ({s['correct']}/{s['total']})")

    summary = "\n".join(lines)
    with open(summary_path, "w") as f:
        f.write(summary + "\n")

    print(summary)


if __name__ == "__main__":
    main()
