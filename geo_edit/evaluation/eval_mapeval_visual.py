"""Evaluation script for MapEval-Visual dataset."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# python geo_edit/evaluation/eval_mapeval_visual.py --output_dir /path/to/inference_output
def extract_answer(text: str) -> str:
    """Extract answer from <answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def parse_integer_answer(answer_text: str) -> int:
    """Parse integer from answer text."""
    # Try to extract first integer
    numbers = re.findall(r"\d+", answer_text)
    if numbers:
        return int(numbers[0])
    return -1


def evaluate_mapeval_visual(output_dir: Path):
    """Evaluate MapEval-Visual predictions."""
    results_by_classification = defaultdict(lambda: {"correct": 0, "total": 0})

    # Read global meta info
    meta_file = output_dir / "global_meta_info.jsonl"

    with open(meta_file) as f:
        for line in f:
            data = json.loads(line)

            classification = data.get("meta_info_extra", {}).get("classification", "unknown")
            ground_truth = int(data["answer"])

            # Extract predicted answer
            prediction_text = data.get("output_text", "")
            predicted = parse_integer_answer(extract_answer(prediction_text))

            results_by_classification[classification]["total"] += 1
            if predicted == ground_truth:
                results_by_classification[classification]["correct"] += 1

    # Print results
    print("MapEval-Visual Results")
    print("=" * 50)

    overall_correct = 0
    overall_total = 0

    for classification in sorted(results_by_classification.keys()):
        stats = results_by_classification[classification]
        accuracy = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"{classification:20s}: {stats['correct']:4d}/{stats['total']:4d} ({accuracy:5.1f}%)")
        overall_correct += stats["correct"]
        overall_total += stats["total"]

    print("-" * 50)
    overall_accuracy = overall_correct / overall_total * 100 if overall_total > 0 else 0
    print(f"{'Overall':20s}: {overall_correct:4d}/{overall_total:4d} ({overall_accuracy:5.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MapEval-Visual predictions.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory containing inference results.")
    args = parser.parse_args()

    evaluate_mapeval_visual(Path(args.output_dir))
