"""Evaluation script for VisWorld-Eval maze task.

Scores maze path predictions using wall-collision detection on the maze image.

Usage:
    python geo_edit/evaluation/eval_visworld_maze.py \
        --result_path /path/to/inference_output \
        --output_path /path/to/eval_output
"""

import argparse
import json
import os
from collections import defaultdict

from geo_edit.evaluation.maze_verifier import wall_judge
from geo_edit.utils.io_utils import iter_meta_info_files, load_records
from geo_edit.utils.stats import get_output_tokens_total, get_input_tokens_total
from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VisWorld-Eval maze predictions.")
    parser.add_argument("--result_path", type=str, required=True, help="Directory containing inference results.")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to save evaluation results.")
    parser.add_argument("--maze_size", type=str, default="5", help="Maze size for scoring threshold (default: 5).")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    eval_output_path = os.path.join(args.output_path, "eval_result.jsonl")
    summary_path = os.path.join(args.output_path, "summary.txt")

    total = 0
    perfect = 0
    score_sum = 0.0
    filtered = 0
    eval_results = []
    output_tokens_sum = input_tokens_sum = 0.0
    output_tokens_count = input_tokens_count = 0

    with open(eval_output_path, "w", encoding="utf-8") as out_f:
        for meta_path in iter_meta_info_files(args.result_path):
            record_id = os.path.basename(os.path.dirname(meta_path))

            for record in load_records(meta_path):
                output_text = record.get("output_text", "")
                if isinstance(output_text, list):
                    output_text = output_text[-1] if output_text else ""
                output_str = str(output_text)

                ground_truth = str(record.get("answer", ""))
                image_path = record.get("image_path")

                if not image_path or not os.path.exists(image_path):
                    filtered += 1
                    logger.warning("Missing image for id=%s: %s", record_id, image_path)
                    eval_item = {
                        "id": record_id,
                        "image_path": image_path,
                        "ground_truth": ground_truth,
                        "score": 0.0,
                        "result": 0.0,
                        "error": "missing_image",
                    }
                    eval_results.append(eval_item)
                    out_f.write(json.dumps(eval_item, ensure_ascii=False) + "\n")
                    continue

                try:
                    score, _ = wall_judge(output_str, image_path, ground_truth, maze_size=args.maze_size)
                except Exception as e:
                    filtered += 1
                    logger.warning("wall_judge error for id=%s: %s", record_id, e)
                    eval_item = {
                        "id": record_id,
                        "image_path": image_path,
                        "ground_truth": ground_truth,
                        "score": 0.0,
                        "result": 0.0,
                        "error": str(e),
                    }
                    eval_results.append(eval_item)
                    out_f.write(json.dumps(eval_item, ensure_ascii=False) + "\n")
                    continue

                total += 1
                score_sum += score
                if score >= 1.0:
                    perfect += 1

                eval_item = {
                    "id": record_id,
                    "question": record.get("question", ""),
                    "image_path": image_path,
                    "ground_truth": ground_truth,
                    "output_text": output_text,
                    "score": score,
                    "result": score,
                    "total_steps": record.get("total_steps"),
                    "function_call_total_count": record.get("function_call_total_count"),
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

    mean_score = score_sum / total if total > 0 else 0.0
    perfect_rate = perfect / total if total > 0 else 0.0

    # Write summary
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("VisWorld-Eval Maze Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"evaluated={total}\n")
        f.write(f"filtered={filtered}\n")
        f.write(f"perfect={perfect}\n")
        f.write(f"perfect_rate={perfect_rate:.6f}\n")
        f.write(f"mean_score={mean_score:.6f}\n")
        f.write("\n")
        avg_output = output_tokens_sum / output_tokens_count if output_tokens_count else 0.0
        avg_input = input_tokens_sum / input_tokens_count if input_tokens_count else 0.0
        f.write(f"total_output_tokens={output_tokens_sum:.0f}\n")
        f.write(f"total_input_tokens={input_tokens_sum:.0f}\n")
        f.write(f"avg_output_tokens={avg_output:.2f}\n")
        f.write(f"avg_input_tokens={avg_input:.2f}\n")

    # Print summary to console
    print("VisWorld-Eval Maze Results")
    print("=" * 50)
    print(f"Evaluated: {total}")
    print(f"Filtered:  {filtered}")
    print(f"Perfect:   {perfect}/{total} ({perfect_rate * 100:.1f}%)")
    print(f"Mean Score: {mean_score:.4f}")
    print(f"\nResults saved to: {args.output_path}")


if __name__ == "__main__":
    main()
