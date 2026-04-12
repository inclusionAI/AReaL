"""Evaluate ReasonMap Plus metro QA predictions.

Usage:
    python -m geo_edit.evaluation.eval_reason_map_plus \
        --result_path /storage/openpsi/data/lcy_image_edit/reasonmap_plus_test_0412/qwen3vl8b-thinking-reasonmap-lr3e-6/   \
        --output_path /storage/openpsi/data/lcy_image_edit/reasonmap_plus_test_0412/qwen3vl8b-thinking-reasonmap-lr3e-6_eval/
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

from geo_edit.evaluation.reason_map_plus_verifier import (
    VALID_TYPES,
    reason_map_plus_score,
)
from geo_edit.utils.io_utils import iter_meta_info_files, load_records
from geo_edit.utils.logger import setup_logger
from geo_edit.utils.stats import (
    compute_tool_combination_statistics,
    get_input_tokens_total,
    get_output_tokens_total,
    get_total_tokens,
)

logger = setup_logger(__name__)

DIFFICULTY_WEIGHTS = {"easy": 1.0, "middle": 1.5, "hard": 2.0}


def _get_field(record: dict, key: str, default: str = "") -> str:
    val = record.get(key)
    if val is not None:
        return str(val)
    return str(record.get("meta_info_extra", {}).get(key, default))


_LETTER_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}
_YESNO_TO_INT = {"yes": 1, "no": 0, "true": 1, "false": 0}


def _normalize_ground_truth(raw, question_type: str) -> int | None:
    """Convert ground truth to int, accepting both integer and string formats.

    Handles: int/str-int directly, A/B/C/D for Counting1, Yes/No for TorF.
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return int(raw)
    s = str(raw).strip()
    try:
        return int(s)
    except ValueError:
        pass
    upper = s.upper()
    if question_type == "Counting1" and upper in _LETTER_TO_INDEX:
        return _LETTER_TO_INDEX[upper]
    lower = s.lower()
    if question_type in ("TorF1", "TorF2") and lower in _YESNO_TO_INT:
        return _YESNO_TO_INT[lower]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ReasonMap Plus predictions.")
    parser.add_argument("--result_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--extract_answer_tags",
        type=str,
        default="split",
        choices=["split", "strict", "none"],
        help="How to extract from <answer> tags before boxed parsing (default: split).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    eval_output_path = os.path.join(args.output_path, "eval_result.jsonl")
    summary_path = os.path.join(args.output_path, "summary.txt")

    total = correct = filtered = 0
    weighted_correct_sum = 0.0
    weight_total = 0.0

    output_tokens_sum = input_tokens_sum = total_tokens_sum = 0.0
    output_tokens_count = input_tokens_count = total_tokens_count = 0

    type_stats: dict[str, dict] = defaultdict(
        lambda: {"total": 0, "correct": 0, "w_correct": 0.0, "w_total": 0.0}
    )
    city_stats: dict[str, dict] = defaultdict(
        lambda: {"total": 0, "correct": 0, "w_correct": 0.0, "w_total": 0.0}
    )
    difficulty_stats: dict[str, dict] = defaultdict(lambda: {"total": 0, "correct": 0})

    eval_results = []

    with open(eval_output_path, "w", encoding="utf-8") as out_f:
        for meta_path in iter_meta_info_files(args.result_path):
            record_id = os.path.basename(os.path.dirname(meta_path))

            for record in load_records(meta_path):
                output_text = record.get("output_text", "")
                if isinstance(output_text, list):
                    output_text = output_text[-1] if output_text else ""
                output_str = str(output_text)

                question_type = _get_field(record, "type")
                city = _get_field(record, "city")
                country = _get_field(record, "country")
                difficulty_city = _get_field(record, "difficulty_city", "easy")

                gt_raw = record.get("answer")
                if gt_raw is None:
                    gt_raw = record.get("meta_info_extra", {}).get("answer")

                if question_type not in VALID_TYPES:
                    filtered += 1
                    eval_item = {
                        "id": record_id,
                        "result": 0.0,
                        "error": f"invalid_type:{question_type}",
                        "type": question_type,
                        "country": country,
                        "city": city,
                    }
                    out_f.write(json.dumps(eval_item, ensure_ascii=False) + "\n")
                    continue

                ground_truth = _normalize_ground_truth(gt_raw, question_type)
                if ground_truth is None:
                    filtered += 1
                    logger.warning(
                        "Unparseable ground_truth for id=%s: %s", record_id, gt_raw
                    )
                    eval_item = {
                        "id": record_id,
                        "result": 0.0,
                        "error": f"invalid_gt:{gt_raw}",
                        "type": question_type,
                        "country": country,
                        "city": city,
                    }
                    out_f.write(json.dumps(eval_item, ensure_ascii=False) + "\n")
                    continue

                try:
                    score, reason, extracted = reason_map_plus_score(
                        output_str, ground_truth, question_type
                    )
                except Exception as e:
                    filtered += 1
                    logger.warning("Scoring error for id=%s: %s", record_id, e)
                    eval_item = {
                        "id": record_id,
                        "result": 0.0,
                        "error": str(e),
                        "type": question_type,
                        "country": country,
                        "city": city,
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

                ts = type_stats[question_type]
                ts["total"] += 1
                ts["correct"] += int(is_correct)
                ts["w_total"] += w
                ts["w_correct"] += w if is_correct else 0.0

                cs = city_stats[city]
                cs["total"] += 1
                cs["correct"] += int(is_correct)
                cs["w_total"] += w
                cs["w_correct"] += w if is_correct else 0.0

                ds = difficulty_stats[difficulty_city]
                ds["total"] += 1
                ds["correct"] += int(is_correct)

                eval_item = {
                    "id": record_id,
                    "question": record.get("question", ""),
                    "output_text": output_text,
                    "ground_truth": ground_truth,
                    "prediction": extracted,
                    "score": score,
                    "result": score,
                    "reason": reason,
                    "type": question_type,
                    "country": country,
                    "city": city,
                    "difficulty_city": difficulty_city,
                    "total_steps": record.get("total_steps"),
                    "function_call_each_count": record.get("function_call_each_count"),
                    "function_call_total_count": record.get(
                        "function_call_total_count"
                    ),
                    "function_call_per_step": record.get("function_call_per_step"),
                    "tokens_used_total": record.get("tokens_used_total"),
                    "tokens_used_per_step": record.get("tokens_used_per_step"),
                    "tokens_output_total": record.get("tokens_output_total"),
                    "tokens_input_total": record.get("tokens_input_total"),
                    "tokens_input_per_step": record.get("tokens_input_per_step"),
                    "tokens_total_per_step": record.get("tokens_total_per_step"),
                }
                eval_results.append(eval_item)
                out_f.write(json.dumps(eval_item, ensure_ascii=False) + "\n")

                output_total = get_output_tokens_total(eval_item)
                if output_total is not None:
                    output_tokens_sum += output_total
                    output_tokens_count += 1
                input_total = get_input_tokens_total(eval_item)
                if input_total is not None:
                    input_tokens_sum += input_total
                    input_tokens_count += 1
                tt = get_total_tokens(eval_item)
                if tt is not None:
                    total_tokens_sum += float(tt)
                    total_tokens_count += 1

    accuracy = correct / total if total > 0 else 0.0
    weighted_acc = weighted_correct_sum / weight_total if weight_total > 0 else 0.0
    tool_stats_text = compute_tool_combination_statistics(eval_results)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("ReasonMap Plus Evaluation Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"evaluated={total}\n")
        f.write(f"filtered={filtered}\n")
        f.write(f"correct={correct}\n")
        f.write(f"accuracy={accuracy:.6f}\n")
        f.write(f"weighted_accuracy={weighted_acc:.6f}\n\n")

        avg_out = (
            output_tokens_sum / output_tokens_count if output_tokens_count else 0.0
        )
        avg_in = input_tokens_sum / input_tokens_count if input_tokens_count else 0.0
        avg_total = total_tokens_sum / total_tokens_count if total_tokens_count else 0.0
        f.write(f"total_output_tokens={output_tokens_sum:.0f}\n")
        f.write(f"total_input_tokens={input_tokens_sum:.0f}\n")
        f.write(f"total_tokens={total_tokens_sum:.0f}\n")
        f.write(f"avg_output_tokens={avg_out:.2f}\n")
        f.write(f"avg_input_tokens={avg_in:.2f}\n")
        f.write(f"avg_total_tokens={avg_total:.2f}\n")

        f.write("\n--- Per-type breakdown ---\n")
        for type_name in sorted(type_stats):
            ts = type_stats[type_name]
            t, c = ts["total"], ts["correct"]
            ta = c / t if t > 0 else 0.0
            twa = ts["w_correct"] / ts["w_total"] if ts["w_total"] > 0 else 0.0
            f.write(f"  {type_name}: {c}/{t} ({ta:.1%}) weighted_acc={twa:.4f}\n")

        f.write("\n--- Per-difficulty breakdown ---\n")
        for diff_name in sorted(difficulty_stats):
            ds = difficulty_stats[diff_name]
            dt, dc = ds["total"], ds["correct"]
            da = dc / dt if dt > 0 else 0.0
            f.write(f"  {diff_name}: {dc}/{dt} ({da:.1%})\n")

        f.write("\n--- Per-city breakdown ---\n")
        for city_name in sorted(city_stats):
            cs = city_stats[city_name]
            ct, cc = cs["total"], cs["correct"]
            ca = cc / ct if ct > 0 else 0.0
            cwa = cs["w_correct"] / cs["w_total"] if cs["w_total"] > 0 else 0.0
            f.write(f"  {city_name}: {cc}/{ct} ({ca:.1%}) weighted_acc={cwa:.4f}\n")

        f.write(tool_stats_text)

    print("ReasonMap Plus Evaluation Results")
    print("=" * 60)
    print(f"Evaluated: {total}")
    print(f"Filtered:  {filtered}")
    print(f"Correct:   {correct}/{total} ({accuracy * 100:.1f}%)")
    print(f"Weighted Accuracy: {weighted_acc:.4f}")
    print()
    for type_name in sorted(type_stats):
        ts = type_stats[type_name]
        t, c = ts["total"], ts["correct"]
        ta = c / t if t > 0 else 0.0
        print(f"  {type_name}: {c}/{t} ({ta:.1%})")
    print(f"\nResults saved to: {args.output_path}")


if __name__ == "__main__":
    main()
