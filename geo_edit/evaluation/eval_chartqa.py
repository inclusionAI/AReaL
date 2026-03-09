"""Evaluation script for ChartQA dataset.

Uses relaxed accuracy metric: for numerical answers, allows 5% tolerance.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


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


def evaluate_chartqa(output_dir: Path):
    """Evaluate ChartQA predictions."""
    results_by_type = defaultdict(lambda: {"correct": 0, "total": 0})

    # Read global meta info
    meta_file = output_dir / "global_meta_info.jsonl"

    with open(meta_file) as f:
        for line in f:
            data = json.loads(line)

            qtype = data.get("meta_info_extra", {}).get("type", "unknown")
            ground_truth = str(data["answer"])

            # Extract predicted answer
            prediction_text = data.get("output_text", "")
            predicted = extract_answer(prediction_text)

            results_by_type[qtype]["total"] += 1
            if relaxed_match(predicted, ground_truth):
                results_by_type[qtype]["correct"] += 1

    # Print results
    print("ChartQA Results (Relaxed Accuracy)")
    print("=" * 50)

    overall_correct = 0
    overall_total = 0

    for qtype in sorted(results_by_type.keys()):
        stats = results_by_type[qtype]
        accuracy = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"{qtype:20s}: {stats['correct']:4d}/{stats['total']:4d} ({accuracy:5.1f}%)")
        overall_correct += stats["correct"]
        overall_total += stats["total"]

    print("-" * 50)
    overall_accuracy = overall_correct / overall_total * 100 if overall_total > 0 else 0
    print(f"{'Overall':20s}: {overall_correct:4d}/{overall_total:4d} ({overall_accuracy:5.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ChartQA predictions.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory containing inference results.")
    args = parser.parse_args()

    evaluate_chartqa(Path(args.output_dir))
