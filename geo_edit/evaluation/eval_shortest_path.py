from __future__ import annotations

import argparse
import json
import os
from typing import Optional

from geo_edit.evaluation.utils import (
    compute_tool_combination_statistics,
    get_final_prediction,
    get_input_tokens_total,
    get_output_tokens_total,
    get_total_tokens,
    iter_meta_info_files,
    load_records,
)
from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)


def parse_path(text: str) -> Optional[str]:
    if not text or "," not in text:
        return None
    parts = [p.strip().upper() for p in text.strip().split(",") if p.strip()]
    if len(parts) < 2:
        return None
    return ",".join(parts)


def gt_to_string(gt_value) -> str:
    if isinstance(gt_value, list):
        return ",".join(str(x).strip().upper() for x in gt_value if str(x).strip())
    return str(gt_value).replace(" ", "").upper()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate shortest-path results.")
    parser.add_argument("--result_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--extract_answer_tags",
        type=str,
        default="none",
        choices=["split", "strict", "none"],
    )
    args = parser.parse_args()

    extract_mode = None if args.extract_answer_tags == "none" else args.extract_answer_tags
    os.makedirs(args.output_path, exist_ok=True)
    eval_output_path = os.path.join(args.output_path, "eval_result.jsonl")
    summary_path = os.path.join(args.output_path, "summary.txt")

    total = correct = filtered = 0
    eval_results = []
    output_tokens_sum = input_tokens_sum = total_tokens_sum = 0.0
    output_tokens_count = input_tokens_count = total_tokens_count = 0

    with open(eval_output_path, "w", encoding="utf-8") as out_f:
        for meta_path in iter_meta_info_files(args.result_path):
            record_id = os.path.basename(os.path.dirname(meta_path))
            for record in load_records(meta_path):
                output_text = record.get("output_text", "")
                predict_str_list = [str(x) for x in output_text] if isinstance(output_text, list) else [str(output_text)]
                raw_pred = get_final_prediction(predict_str_list, extract_mode)
                parsed_pred = parse_path(raw_pred)

                gt_value = gt_to_string(record.get("answer"))
                if parsed_pred is None:
                    filtered += 1
                    logger.warning("No comma-separated path for id=%s: %s", record_id, raw_pred)
                    result = {"is_filter": True, "info": "no_comma_path", "raw_pred": raw_pred}
                else:
                    total += 1
                    result = 1.0 if parsed_pred == gt_value else 0.0
                    if result == 1.0:
                        correct += 1

                eval_item = {
                    "id": record_id,
                    "question": record.get("question"),
                    "image_path": record.get("image_path"),
                    "total_steps": record.get("total_steps"),
                    "function_call_each_count": record.get("function_call_each_count"),
                    "function_call_total_count": record.get("function_call_total_count"),
                    "function_call_per_step": record.get("function_call_per_step"),
                    "tokens_used_total": record.get("tokens_used_total"),
                    "tokens_used_per_step": record.get("tokens_used_per_step"),
                    "tokens_output_total": record.get("tokens_output_total"),
                    "tokens_input_total": record.get("tokens_input_total"),
                    "tokens_input_per_step": record.get("tokens_input_per_step"),
                    "tokens_total_per_step": record.get("tokens_total_per_step"),
                    "output_text": output_text,
                    "ground_truth": gt_value,
                    "prediction": parsed_pred,
                    "result": result,
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
                total_total = get_total_tokens(eval_item)
                if total_total is not None:
                    total_tokens_sum += float(total_total)
                    total_tokens_count += 1

    tool_stats_text = compute_tool_combination_statistics(eval_results)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"evaluated={total}\n")
        f.write(f"correct={correct}\n")
        f.write(f"filtered={filtered}\n")
        f.write(f"accuracy={(correct / total) if total else 0.0:.6f}\n")
        avg_output = output_tokens_sum / output_tokens_count if output_tokens_count else 0.0
        avg_input = input_tokens_sum / input_tokens_count if input_tokens_count else 0.0
        avg_total = total_tokens_sum / total_tokens_count if total_tokens_count else 0.0
        f.write(f"total_output_tokens={output_tokens_sum:.0f}\n")
        f.write(f"total_input_tokens={input_tokens_sum:.0f}\n")
        f.write(f"total_tokens={total_tokens_sum:.0f}\n")
        f.write(f"avg_output_tokens={avg_output:.2f}\n")
        f.write(f"avg_input_tokens={avg_input:.2f}\n")
        f.write(f"avg_total_tokens={avg_total:.2f}\n")
        f.write(tool_stats_text)


if __name__ == "__main__":
    main()
