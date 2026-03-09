"""Evaluation script for ChartQA dataset.

Uses relaxed accuracy metric: for numerical answers, allows 5% tolerance.

Usage:
    python geo_edit/evaluation/eval_chartqa.py \
        --result_path /path/to/inference_output \
        --output_path /path/to/eval_output
"""

import argparse
import json
import os
import re
from collections import defaultdict

from geo_edit.utils.io_utils import iter_meta_info_files, load_records
from geo_edit.utils.stats import get_output_tokens_total, get_input_tokens_total


def extract_answer(text: str) -> str:
    """Extract answer from <answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    answer = answer.lower().strip()
    # Remove common punctuation
    answer = re.sub(r"[,\$%]", "", answer)
    return answer


def is_numeric(s: str) -> bool:
    """Check if string is numeric."""
    try:
        float(s)
        return True
    except ValueError:
        return False


def relaxed_match(pred: str, gold: str, tolerance: float = 0.05) -> bool:
    """Check if prediction matches gold answer with relaxed accuracy.

    For numerical answers, allows a tolerance (default 5%).
    For text answers, uses exact match after normalization.
    """
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)

    # Exact match after normalization
    if pred_norm == gold_norm:
        return True

    # Try numerical comparison with tolerance
    if is_numeric(pred_norm) and is_numeric(gold_norm):
        pred_val = float(pred_norm)
        gold_val = float(gold_norm)
        if gold_val == 0:
            return pred_val == 0
        return abs(pred_val - gold_val) / abs(gold_val) <= tolerance

    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ChartQA predictions.")
    parser.add_argument("--result_path", type=str, required=True, help="Directory containing inference results.")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to save evaluation results.")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    eval_output_path = os.path.join(args.output_path, "eval_result.jsonl")
    summary_path = os.path.join(args.output_path, "summary.txt")

    results_by_type = defaultdict(lambda: {"correct": 0, "total": 0})
    eval_results = []
    output_tokens_sum = input_tokens_sum = 0.0
    output_tokens_count = input_tokens_count = 0

    with open(eval_output_path, "w", encoding="utf-8") as out_f:
        for meta_path in iter_meta_info_files(args.result_path):
            record_id = os.path.basename(os.path.dirname(meta_path))

            for record in load_records(meta_path):
                # type may be in meta_info_extra or directly in record
                qtype = record.get("type") or record.get("meta_info_extra", {}).get("type", "unknown")
                ground_truth = str(record.get("answer", ""))

                # Extract predicted answer
                output_text = record.get("output_text", "")
                if isinstance(output_text, list):
                    output_text = output_text[-1] if output_text else ""
                predicted = extract_answer(str(output_text))

                is_correct = relaxed_match(predicted, ground_truth)
                results_by_type[qtype]["total"] += 1
                if is_correct:
                    results_by_type[qtype]["correct"] += 1

                eval_item = {
                    "id": record_id,
                    "type": qtype,
                    "ground_truth": ground_truth,
                    "prediction": predicted,
                    "result": 1.0 if is_correct else 0.0,
                    "output_text": output_text,
                }
                eval_results.append(eval_item)
                out_f.write(json.dumps(eval_item, ensure_ascii=False) + "\n")

                # Token statistics
                output_total = get_output_tokens_total(record)
                if output_total is not None:
                    output_tokens_sum += output_total
                    output_tokens_count += 1
                input_total = get_input_tokens_total(record)
                if input_total is not None:
                    input_tokens_sum += input_total
                    input_tokens_count += 1

    # Calculate overall statistics
    overall_correct = sum(s["correct"] for s in results_by_type.values())
    overall_total = sum(s["total"] for s in results_by_type.values())
    overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0.0

    # Write summary
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("ChartQA Results (Relaxed Accuracy)\n")
        f.write("=" * 50 + "\n")

        for qtype in sorted(results_by_type.keys()):
            stats = results_by_type[qtype]
            accuracy = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
            f.write(f"{qtype:20s}: {stats['correct']:4d}/{stats['total']:4d} ({accuracy:5.1f}%)\n")

        f.write("-" * 50 + "\n")
        f.write(f"{'Overall':20s}: {overall_correct:4d}/{overall_total:4d} ({overall_accuracy * 100:5.1f}%)\n")
        f.write("\n")
        f.write(f"evaluated={overall_total}\n")
        f.write(f"correct={overall_correct}\n")
        f.write(f"accuracy={overall_accuracy:.6f}\n")

        avg_output = output_tokens_sum / output_tokens_count if output_tokens_count else 0.0
        avg_input = input_tokens_sum / input_tokens_count if input_tokens_count else 0.0
        f.write(f"total_output_tokens={output_tokens_sum:.0f}\n")
        f.write(f"total_input_tokens={input_tokens_sum:.0f}\n")
        f.write(f"avg_output_tokens={avg_output:.2f}\n")
        f.write(f"avg_input_tokens={avg_input:.2f}\n")

    # Print summary to console
    print("ChartQA Results (Relaxed Accuracy)")
    print("=" * 50)
    for qtype in sorted(results_by_type.keys()):
        stats = results_by_type[qtype]
        accuracy = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"{qtype:20s}: {stats['correct']:4d}/{stats['total']:4d} ({accuracy:5.1f}%)")
    print("-" * 50)
    print(f"{'Overall':20s}: {overall_correct:4d}/{overall_total:4d} ({overall_accuracy * 100:5.1f}%)")
    print(f"\nResults saved to: {args.output_path}")


if __name__ == "__main__":
    main()
