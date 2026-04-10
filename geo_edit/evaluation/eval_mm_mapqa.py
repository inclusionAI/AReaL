"""Exact-match evaluator for mm_mapqa.

Answers are typically short: Yes/No, numbers, place names.
Uses normalized string matching with a fallback to OpenAI judge for ambiguous cases.

Usage:
    python -m geo_edit.evaluation.eval_mm_mapqa \
        --result_path /path/to/output/mm_mapqa_thinking \
        --output_path /path/to/eval/mm_mapqa_thinking
"""
from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from geo_edit.utils.io_utils import iter_meta_info_files, load_records
from geo_edit.utils.stats import (
    get_input_tokens_total,
    get_output_tokens_total,
    get_total_tokens,
)
from geo_edit.utils.text_utils import get_final_prediction


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.lower().strip()
    text = text.rstrip(".")
    text = re.sub(r"\s+", " ", text)
    return text


def exact_match(prediction: str, ground_truth: str) -> bool:
    return normalize(prediction) == normalize(ground_truth)


def relaxed_match(prediction: str, ground_truth: str) -> bool:
    pred_n = normalize(prediction)
    gt_n = normalize(ground_truth)

    if pred_n == gt_n:
        return True

    YES_VARIANTS = {"yes", "true", "correct"}
    NO_VARIANTS = {"no", "false", "incorrect"}
    if gt_n in YES_VARIANTS and pred_n in YES_VARIANTS:
        return True
    if gt_n in NO_VARIANTS and pred_n in NO_VARIANTS:
        return True

    return False


def evaluate_record(
    record: dict,
    record_id: str,
    extract_mode: str = "split",
    judge=None,
) -> dict:
    question = record["question"]
    ground_truth = record["answer"]
    if isinstance(ground_truth, list):
        ground_truth = "\n".join(str(g) for g in ground_truth)
    ground_truth = str(ground_truth)

    output_text = record["output_text"]
    if isinstance(output_text, list):
        predict_str_list = [str(x) for x in output_text]
    else:
        predict_str_list = [str(output_text)]

    prediction = get_final_prediction(predict_str_list, extract_mode)

    if relaxed_match(prediction, ground_truth):
        result = 1.0
    elif judge is not None:
        try:
            score_str = judge.judge_correctness(question, ground_truth, prediction)
            result = 1.0 if score_str == "1" else 0.0
        except Exception:
            result = {"is_filter": True, "info": "judge_error"}
    else:
        result = 0.0

    return {
        "id": record_id,
        "question": question,
        "image_path": record.get("image_path"),
        "total_steps": record.get("total_steps"),
        "function_call_each_count": record.get("function_call_each_count"),
        "function_call_total_count": record.get("function_call_total_count"),
        "tokens_used_total": record.get("tokens_used_total"),
        "tokens_output_total": record.get("tokens_output_total"),
        "tokens_input_total": record.get("tokens_input_total"),
        "output_text": output_text,
        "ground_truth": ground_truth,
        "prediction": prediction,
        "result": result,
    }


def main():
    parser = argparse.ArgumentParser(description="Exact-match evaluator for mm_mapqa.")
    parser.add_argument("--result_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--extract_answer_tags",
        type=str,
        default="split",
        choices=["split", "strict"],
    )
    parser.add_argument(
        "--use_judge_fallback",
        action="store_true",
        help="Fall back to OpenAI judge for non-exact matches.",
    )
    parser.add_argument("--judge_api_key", type=str, default=None)
    parser.add_argument("--judge_model", type=str, default="gpt-5-mini-2025-08-07")
    parser.add_argument("--judge_api_base", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    judge = None
    if args.use_judge_fallback and args.judge_api_key:
        from geo_edit.evaluation.openai_as_judge import OpenAIJudge
        judge = OpenAIJudge(
            api_key=args.judge_api_key,
            model=args.judge_model,
            api_base=args.judge_api_base,
        )

    eval_output_path = os.path.join(args.output_path, "eval_result.jsonl")
    summary_path = os.path.join(args.output_path, "summary.txt")

    total = 0
    correct = 0
    filtered = 0
    eval_results: List[dict] = []
    output_tokens_sum = 0.0
    input_tokens_sum = 0.0
    total_tokens_sum = 0.0
    output_tokens_count = 0
    input_tokens_count = 0
    total_tokens_count = 0

    with open(eval_output_path, "w", encoding="utf-8") as out_f:
        for meta_path in iter_meta_info_files(args.result_path):
            record_id = os.path.basename(os.path.dirname(meta_path))
            for record in load_records(meta_path):
                eval_item = evaluate_record(
                    record, record_id, args.extract_answer_tags, judge
                )
                eval_results.append(eval_item)
                result = eval_item["result"]
                is_filter = isinstance(result, dict) and result.get("is_filter")
                if is_filter:
                    filtered += 1
                else:
                    total += 1
                    if result == 1.0:
                        correct += 1
                out_f.write(json.dumps(eval_item, ensure_ascii=False) + "\n")

                output_total = get_output_tokens_total(eval_item)
                if output_total is not None:
                    output_tokens_sum += output_total
                    output_tokens_count += 1
                input_total = get_input_tokens_total(eval_item)
                if input_total is not None:
                    input_tokens_sum += input_total
                    input_tokens_count += 1
                total_total = get_total_tokens(eval_item)
                if total_total is not None:
                    total_tokens_sum += float(total_total)
                    total_tokens_count += 1

    accuracy = correct / total if total else 0.0
    avg_output = output_tokens_sum / output_tokens_count if output_tokens_count else 0.0
    avg_input = input_tokens_sum / input_tokens_count if input_tokens_count else 0.0
    avg_total = total_tokens_sum / total_tokens_count if total_tokens_count else 0.0

    error_ids = [r["id"] for r in eval_results if r["result"] == 0.0]
    correct_ids = [r["id"] for r in eval_results if r["result"] == 1.0]

    error_ids_path = os.path.join(args.output_path, "error_ids.json")
    correct_ids_path = os.path.join(args.output_path, "correct_ids.json")
    with open(error_ids_path, "w") as f:
        json.dump(error_ids, f)
    with open(correct_ids_path, "w") as f:
        json.dump(correct_ids, f)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"evaluated={total}\n")
        f.write(f"correct={correct}\n")
        f.write(f"filtered={filtered}\n")
        f.write(f"accuracy={accuracy:.6f}\n")
        f.write(f"error_count={len(error_ids)}\n")
        f.write(f"total_output_tokens={output_tokens_sum:.0f}\n")
        f.write(f"total_input_tokens={input_tokens_sum:.0f}\n")
        f.write(f"total_tokens={total_tokens_sum:.0f}\n")
        f.write(f"avg_output_tokens={avg_output:.2f}\n")
        f.write(f"avg_input_tokens={avg_input:.2f}\n")
        f.write(f"avg_total_tokens={avg_total:.2f}\n")

    print(f"Accuracy: {correct}/{total} = {accuracy:.4f}")
    print(f"Filtered: {filtered}")
    print(f"Error IDs: {len(error_ids)} saved to {error_ids_path}")
    print(f"Avg output tokens: {avg_output:.2f}")
    print(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    main()
