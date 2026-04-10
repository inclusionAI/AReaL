"""Evaluation script for ReasonMap (base) route planning task.

Rule-based verification of metro route predictions against the network
topology.  Computes accuracy, weighted accuracy (by city difficulty),
and per-city breakdowns.

Usage:
    python -m geo_edit.evaluation.eval_reason_map \
        --result_path /path/to/inference_output \
        --output_path /path/to/eval_output
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

from geo_edit.evaluation.reason_map_verifier import reason_map_score
from geo_edit.utils.io_utils import iter_meta_info_files, load_records
from geo_edit.utils.stats import get_output_tokens_total, get_input_tokens_total
from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)

DIFFICULTY_WEIGHTS = {"easy": 1.0, "middle": 1.5, "hard": 2.0}


def _parse_metro_data(record: dict) -> dict:
    """Extract metro_data from record, checking both top-level and meta_info_extra."""
    raw = record.get("metro_data") or record.get("meta_info_extra", {}).get("metro_data", {})
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
    return raw if isinstance(raw, dict) else {}


def _get_field(record: dict, key: str, default: str = "") -> str:
    """Get a field from record top-level or meta_info_extra."""
    val = record.get(key)
    if val is not None:
        return str(val)
    return str(record.get("meta_info_extra", {}).get(key, default))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ReasonMap route predictions.")
    parser.add_argument("--result_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    eval_output_path = os.path.join(args.output_path, "eval_result.jsonl")
    summary_path = os.path.join(args.output_path, "summary.txt")

    total = 0
    correct = 0
    filtered = 0
    weighted_correct_sum = 0.0
    weight_total = 0.0
    output_tokens_sum = input_tokens_sum = 0.0
    output_tokens_count = input_tokens_count = 0

    city_stats: dict[str, dict] = defaultdict(lambda: {
        "total": 0, "correct": 0, "w_correct": 0.0, "w_total": 0.0,
    })

    with open(eval_output_path, "w", encoding="utf-8") as out_f:
        for meta_path in iter_meta_info_files(args.result_path):
            record_id = os.path.basename(os.path.dirname(meta_path))

            for record in load_records(meta_path):
                output_text = record.get("output_text", "")
                if isinstance(output_text, list):
                    output_text = output_text[-1] if output_text else ""
                output_str = str(output_text)

                station_1 = _get_field(record, "station_1")
                station_2 = _get_field(record, "station_2")
                metro_data = _parse_metro_data(record)
                difficulty_city = _get_field(record, "difficulty_city", "easy")
                difficulty_question = _get_field(record, "difficulty_question", "easy")
                city = _get_field(record, "city")
                country = _get_field(record, "country")

                if not station_1 or not station_2 or not metro_data:
                    filtered += 1
                    eval_item = {
                        "id": record_id,
                        "result": 0.0,
                        "error": "missing_station_or_metro_data",
                        "country": country,
                        "city": city,
                        "station_1": station_1,
                        "station_2": station_2,
                        "difficulty_city": difficulty_city,
                        "difficulty_question": difficulty_question,
                    }
                    out_f.write(json.dumps(eval_item, ensure_ascii=False) + "\n")
                    continue

                try:
                    score, reason = reason_map_score(
                        output_str, station_1, station_2, metro_data,
                    )
                except Exception as e:
                    filtered += 1
                    logger.warning("reason_map_score error for id=%s: %s", record_id, e)
                    eval_item = {
                        "id": record_id,
                        "result": 0.0,
                        "error": str(e),
                        "country": country,
                        "city": city,
                        "difficulty_city": difficulty_city,
                        "difficulty_question": difficulty_question,
                    }
                    out_f.write(json.dumps(eval_item, ensure_ascii=False) + "\n")
                    continue

                total += 1
                is_correct = score >= 1.0
                if is_correct:
                    correct += 1

                w = DIFFICULTY_WEIGHTS.get(difficulty_city, 1.0)
                weight_total += w
                if is_correct:
                    weighted_correct_sum += w

                cs = city_stats[city]
                cs["total"] += 1
                cs["correct"] += int(is_correct)
                cs["w_total"] += w
                cs["w_correct"] += w if is_correct else 0.0

                eval_item = {
                    "id": record_id,
                    "question": record.get("question", ""),
                    "output_text": output_text,
                    "ground_truth": record.get("answer", ""),
                    "score": score,
                    "result": score,
                    "reason": reason,
                    "country": country,
                    "city": city,
                    "station_1": station_1,
                    "station_2": station_2,
                    "difficulty_city": difficulty_city,
                    "difficulty_question": difficulty_question,
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

    accuracy = correct / total if total > 0 else 0.0
    weighted_acc = weighted_correct_sum / weight_total if weight_total > 0 else 0.0

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("ReasonMap Route Evaluation Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"evaluated={total}\n")
        f.write(f"filtered={filtered}\n")
        f.write(f"correct={correct}\n")
        f.write(f"accuracy={accuracy:.6f}\n")
        f.write(f"weighted_accuracy={weighted_acc:.6f}\n")
        f.write("\n")
        avg_out = output_tokens_sum / output_tokens_count if output_tokens_count else 0.0
        avg_in = input_tokens_sum / input_tokens_count if input_tokens_count else 0.0
        f.write(f"total_output_tokens={output_tokens_sum:.0f}\n")
        f.write(f"total_input_tokens={input_tokens_sum:.0f}\n")
        f.write(f"avg_output_tokens={avg_out:.2f}\n")
        f.write(f"avg_input_tokens={avg_in:.2f}\n")
        f.write("\n--- Per-city breakdown ---\n")
        for city_name in sorted(city_stats):
            cs = city_stats[city_name]
            ct, cc = cs["total"], cs["correct"]
            ca = cc / ct if ct > 0 else 0.0
            cwa = cs["w_correct"] / cs["w_total"] if cs["w_total"] > 0 else 0.0
            f.write(f"{city_name}: {cc}/{ct} ({ca:.1%}) weighted_acc={cwa:.4f}\n")

    print("ReasonMap Route Evaluation Results")
    print("=" * 50)
    print(f"Evaluated: {total}")
    print(f"Filtered:  {filtered}")
    print(f"Correct:   {correct}/{total} ({accuracy * 100:.1f}%)")
    print(f"Weighted Accuracy: {weighted_acc:.4f}")
    print(f"\nResults saved to: {args.output_path}")


if __name__ == "__main__":
    main()
