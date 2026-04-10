"""Evaluation script for MapTrace route-tracing task.

Computes NDTW distance and Success Rate.

Usage:
    python -m geo_edit.evaluation.eval_map_trace \
        --result_path /path/to/inference_output \
        --output_path /path/to/eval_output
"""
from __future__ import annotations

import argparse
import json
import os

from geo_edit.evaluation.map_trace_verifier import map_trace_score
from geo_edit.utils.io_utils import iter_meta_info_files, load_records
from geo_edit.utils.stats import get_output_tokens_total, get_input_tokens_total
from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)


def _get_field(record: dict, key: str, default: str = "") -> str:
    val = record.get(key)
    if val is not None:
        return str(val)
    return str(record.get("meta_info_extra", {}).get(key, default))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MapTrace predictions.")
    parser.add_argument("--result_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--ndtw_threshold", type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    eval_output_path = os.path.join(args.output_path, "eval_result.jsonl")
    summary_path = os.path.join(args.output_path, "summary.txt")

    total = 0
    success_count = 0
    accepted_count = 0
    ndtw_sum = 0.0
    ndtw_values: list[float] = []
    output_tokens_sum = input_tokens_sum = 0.0
    output_tokens_count = input_tokens_count = 0

    with open(eval_output_path, "w", encoding="utf-8") as out_f:
        for meta_path in iter_meta_info_files(args.result_path):
            record_id = os.path.basename(os.path.dirname(meta_path))

            for record in load_records(meta_path):
                total += 1
                output_text = record.get("output_text", "")
                if isinstance(output_text, list):
                    output_text = output_text[-1] if output_text else ""
                output_str = str(output_text)

                ground_truth = record.get("answer", "")

                ndtw, is_success, reason = map_trace_score(
                    output_str, str(ground_truth), args.ndtw_threshold,
                )

                if is_success:
                    success_count += 1
                    ndtw_sum += ndtw
                    ndtw_values.append(ndtw)

                is_accepted = is_success and ndtw <= args.ndtw_threshold
                if is_accepted:
                    accepted_count += 1

                eval_item = {
                    "id": record_id,
                    "ndtw": ndtw if is_success else None,
                    "success": is_success,
                    "accepted": is_accepted,
                    "reason": reason,
                    "result": 1.0 if is_accepted else 0.0,
                    "output_text": output_text,
                    "ground_truth": ground_truth,
                    "total_steps": record.get("total_steps"),
                    "function_call_total_count": record.get("function_call_total_count"),
                }
                out_f.write(json.dumps(eval_item, ensure_ascii=False) + "\n")

                output_total = get_output_tokens_total(record)
                if output_total is not None:
                    output_tokens_sum += output_total
                    output_tokens_count += 1
                input_total = get_input_tokens_total(record)
                if input_total is not None:
                    input_tokens_sum += input_total
                    input_tokens_count += 1

    sr = success_count / total if total > 0 else 0.0
    avg_ndtw = ndtw_sum / success_count if success_count > 0 else float("inf")
    accept_rate = accepted_count / total if total > 0 else 0.0

    ndtw_values.sort()
    median_ndtw = (
        ndtw_values[len(ndtw_values) // 2] if ndtw_values else float("inf")
    )

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("MapTrace Evaluation Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"total={total}\n")
        f.write(f"success_count={success_count}\n")
        f.write(f"success_rate={sr:.6f}\n")
        f.write(f"avg_ndtw={avg_ndtw:.6f}\n")
        f.write(f"median_ndtw={median_ndtw:.6f}\n")
        f.write(f"ndtw_threshold={args.ndtw_threshold}\n")
        f.write(f"accepted_count={accepted_count}\n")
        f.write(f"accept_rate={accept_rate:.6f}\n")
        f.write("\n")
        avg_out = output_tokens_sum / output_tokens_count if output_tokens_count else 0.0
        avg_in = input_tokens_sum / input_tokens_count if input_tokens_count else 0.0
        f.write(f"total_output_tokens={output_tokens_sum:.0f}\n")
        f.write(f"total_input_tokens={input_tokens_sum:.0f}\n")
        f.write(f"avg_output_tokens={avg_out:.2f}\n")
        f.write(f"avg_input_tokens={avg_in:.2f}\n")

    print("MapTrace Evaluation Results")
    print("=" * 50)
    print(f"Total:        {total}")
    print(f"Success Rate: {success_count}/{total} ({sr * 100:.1f}%)")
    print(f"Avg NDTW:     {avg_ndtw:.4f}")
    print(f"Median NDTW:  {median_ndtw:.4f}")
    print(f"Accepted (NDTW<={args.ndtw_threshold}): {accepted_count}/{total} ({accept_rate * 100:.1f}%)")
    print(f"\nResults saved to: {args.output_path}")


if __name__ == "__main__":
    main()
