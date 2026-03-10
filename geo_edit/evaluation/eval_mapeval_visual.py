"""Evaluation script for MapEval-Visual dataset.

Usage:
    python geo_edit/evaluation/eval_mapeval_visual.py \
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
    return ""


def parse_integer_answer(answer_text: str) -> int:
    """Parse integer from answer text."""
    numbers = re.findall(r"\d+", answer_text)
    if numbers:
        return int(numbers[0])
    return -1


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MapEval-Visual predictions.")
    parser.add_argument("--result_path", type=str, required=True, help="Directory containing inference results.")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to save evaluation results.")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    eval_output_path = os.path.join(args.output_path, "eval_result.jsonl")
    summary_path = os.path.join(args.output_path, "summary.txt")

    results_by_classification = defaultdict(lambda: {"correct": 0, "total": 0})
    eval_results = []
    output_tokens_sum = input_tokens_sum = 0.0
    output_tokens_count = input_tokens_count = 0

    with open(eval_output_path, "w", encoding="utf-8") as out_f:
        for meta_path in iter_meta_info_files(args.result_path):
            record_id = os.path.basename(os.path.dirname(meta_path))

            for record in load_records(meta_path):
                # classification may be in meta_info_extra or directly in record
                classification = record.get("classification") or record.get("meta_info_extra", {}).get("classification", "unknown")
                ground_truth = int(record.get("answer", -1))

                # Extract predicted answer
                output_text = record.get("output_text", "")
                if isinstance(output_text, list):
                    output_text = output_text[-1] if output_text else ""
                output_str = str(output_text)
                # Try to extract from <answer> tags first, fallback to raw text
                extracted = extract_answer(output_str)
                predicted = parse_integer_answer(extracted if extracted else output_str)

                is_correct = predicted == ground_truth
                results_by_classification[classification]["total"] += 1
                if is_correct:
                    results_by_classification[classification]["correct"] += 1

                eval_item = {
                    "id": record_id,
                    "question": record.get("question", ""),
                    "image_path": record.get("image_path"),
                    "classification": classification,
                    "ground_truth": str(ground_truth),
                    "prediction": str(predicted),
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
    overall_correct = sum(s["correct"] for s in results_by_classification.values())
    overall_total = sum(s["total"] for s in results_by_classification.values())
    overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0.0

    # Write summary
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("MapEval-Visual Results\n")
        f.write("=" * 50 + "\n")

        for classification in sorted(results_by_classification.keys()):
            stats = results_by_classification[classification]
            accuracy = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
            f.write(f"{classification:20s}: {stats['correct']:4d}/{stats['total']:4d} ({accuracy:5.1f}%)\n")

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
    print("MapEval-Visual Results")
    print("=" * 50)
    for classification in sorted(results_by_classification.keys()):
        stats = results_by_classification[classification]
        accuracy = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"{classification:20s}: {stats['correct']:4d}/{stats['total']:4d} ({accuracy:5.1f}%)")
    print("-" * 50)
    print(f"{'Overall':20s}: {overall_correct:4d}/{overall_total:4d} ({overall_accuracy * 100:5.1f}%)")
    print(f"\nResults saved to: {args.output_path}")


if __name__ == "__main__":
    main()
