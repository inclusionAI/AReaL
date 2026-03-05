"""Evaluation script for VisWorld-Eval dataset."""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict


def extract_answer(text: str) -> str:
    """Extract answer from <answer> tags."""
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def parse_integer_answer(answer_text: str) -> int:
    """Parse integer from answer text."""
    # Try to extract first integer
    numbers = re.findall(r'\d+', answer_text)
    if numbers:
        return int(numbers[0])
    return -1


def evaluate_visworld(output_dir: Path):
    """Evaluate VisWorld-Eval predictions."""
    results_by_category = defaultdict(lambda: {"correct": 0, "total": 0})

    # Read global meta info
    meta_file = output_dir / "global_meta_info.jsonl"

    with open(meta_file) as f:
        for line in f:
            data = json.loads(line)

            category = data.get("meta_info_extra", {}).get("category", "unknown")
            ground_truth = int(data["answer"])

            # Extract predicted answer
            prediction_text = data.get("output_text", "")
            predicted = parse_integer_answer(extract_answer(prediction_text))

            results_by_category[category]["total"] += 1
            if predicted == ground_truth:
                results_by_category[category]["correct"] += 1

    # Print results
    print("VisWorld-Eval Results")
    print("=" * 50)

    overall_correct = 0
    overall_total = 0

    for category in sorted(results_by_category.keys()):
        stats = results_by_category[category]
        accuracy = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"{category:15s}: {stats['correct']:4d}/{stats['total']:4d} ({accuracy:5.1f}%)")
        overall_correct += stats["correct"]
        overall_total += stats["total"]

    print("-" * 50)
    overall_accuracy = overall_correct / overall_total * 100 if overall_total > 0 else 0
    print(f"{'Overall':15s}: {overall_correct:4d}/{overall_total:4d} ({overall_accuracy:5.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    evaluate_visworld(Path(args.output_dir))
