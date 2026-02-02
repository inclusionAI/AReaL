import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from geo_edit.constants import MAX_TOOL_CALLS


def _find_int(pattern: str, text: str) -> Optional[int]:
    match = re.search(pattern, text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _parse_tokens_from_response(
    text: str,
) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    if not text:
        return None, None, None, None

    input_tokens = _find_int(r"prompt_token_count=(\d+)", text)
    if input_tokens is None:
        input_tokens = _find_int(r"prompt_tokens=(\d+)", text)
    if input_tokens is None:
        input_tokens = _find_int(r"input_tokens=(\d+)", text)

    output_tokens = _find_int(r"candidates_token_count=(\d+)", text)
    if output_tokens is None:
        output_tokens = _find_int(r"completion_tokens=(\d+)", text)
    if output_tokens is None:
        output_tokens = _find_int(r"output_tokens=(\d+)", text)

    total_tokens = _find_int(r"total_token_count=(\d+)", text)
    if total_tokens is None:
        total_tokens = _find_int(r"total_tokens=(\d+)", text)

    thoughts_tokens = _find_int(r"thoughts_token_count=(\d+)", text)
    if thoughts_tokens is None:
        thoughts_tokens = _find_int(r"reasoning_tokens=(\d+)", text)

    return input_tokens, output_tokens, total_tokens, thoughts_tokens


def _extract_tokens(
    extra_info: Dict[str, Any],
) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    tokens_input = extra_info.get("tokens_input")
    tokens_output = extra_info.get("tokens_output")
    tokens_total = extra_info.get("tokens_used")
    tokens_thoughts = extra_info.get("tokens_thoughts")

    if (
        tokens_input is None
        or tokens_output is None
        or tokens_total is None
        or tokens_thoughts is None
    ):
        parsed_input, parsed_output, parsed_total, parsed_thoughts = _parse_tokens_from_response(
            str(extra_info.get("original_response") or "")
        )
        if tokens_input is None:
            tokens_input = parsed_input
        if tokens_output is None:
            tokens_output = parsed_output
        if tokens_total is None:
            tokens_total = parsed_total
        if tokens_thoughts is None:
            tokens_thoughts = parsed_thoughts

    if tokens_total is None and tokens_input is not None and tokens_output is not None:
        tokens_total = tokens_input + tokens_output

    if tokens_output is None and tokens_total is not None and tokens_input is not None:
        tokens_output = tokens_total - tokens_input
        if tokens_output < 0:
            tokens_output = None

    if tokens_input is None and tokens_total is not None and tokens_output is not None:
        tokens_input = tokens_total - tokens_output
        if tokens_input < 0:
            tokens_input = None

    return tokens_input, tokens_output, tokens_total, tokens_thoughts


def _last_numeric(values: List[Any]) -> Optional[float]:
    for value in reversed(values):
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _load_first_json_line(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    return {}


def _load_extra_info(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _recompute_tokens(meta_info: Dict[str, Any], extra_info_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_steps = meta_info.get("total_steps")
    if not isinstance(total_steps, int) or total_steps <= 0:
        total_steps = max((r.get("step", 0) for r in extra_info_records), default=0)

    tokens_total_per_step: List[Optional[int]] = [None] * total_steps
    tokens_input_per_step: List[Optional[int]] = [None] * total_steps
    tokens_output_per_step: List[Optional[int]] = [None] * total_steps

    for record in extra_info_records:
        step = record.get("step")
        if not isinstance(step, int):
            continue
        idx = step - 1
        if idx < 0:
            continue
        if idx >= len(tokens_total_per_step):
            pad = idx + 1 - len(tokens_total_per_step)
            tokens_total_per_step.extend([None] * pad)
            tokens_input_per_step.extend([None] * pad)
            tokens_output_per_step.extend([None] * pad)

        extra_info = record.get("extra_info") or {}
        tokens_input, tokens_output, tokens_total, tokens_thoughts = _extract_tokens(extra_info)
        tokens_total_per_step[idx] = tokens_total if isinstance(tokens_total, (int, float)) else None
        tokens_input_per_step[idx] = tokens_input if isinstance(tokens_input, (int, float)) else None

        output_no_thoughts = None
        output_with_thoughts = None
        if isinstance(tokens_output, (int, float)) and isinstance(tokens_thoughts, (int, float)):
            output_no_thoughts = tokens_output
            output_with_thoughts = tokens_output + tokens_thoughts
        elif isinstance(tokens_output, (int, float)):
            output_no_thoughts = tokens_output
            output_with_thoughts = tokens_output
        elif (
            isinstance(tokens_total, (int, float))
            and isinstance(tokens_input, (int, float))
            and isinstance(tokens_thoughts, (int, float))
        ):
            output_no_thoughts = tokens_total - tokens_input - tokens_thoughts
            output_with_thoughts = tokens_total - tokens_input
        elif isinstance(tokens_total, (int, float)) and isinstance(tokens_input, (int, float)):
            output_with_thoughts = tokens_total - tokens_input

        if isinstance(output_no_thoughts, (int, float)) and output_no_thoughts < 0:
            output_no_thoughts = None
        if isinstance(output_with_thoughts, (int, float)) and output_with_thoughts < 0:
            output_with_thoughts = None

        tokens_output_per_step[idx] = output_with_thoughts if isinstance(output_with_thoughts, (int, float)) else None

    tokens_output_total = sum(
        float(v) for v in tokens_output_per_step if isinstance(v, (int, float))
    )

    last_total = _last_numeric(tokens_total_per_step)
    if last_total is not None:
        tokens_used_total = last_total
    else:
        tokens_used_total = tokens_output_total

    tokens_input_total = None
    if isinstance(tokens_used_total, (int, float)) and isinstance(tokens_output_total, (int, float)):
        tokens_input_total = float(tokens_used_total) - float(tokens_output_total)
        if tokens_input_total < 0:
            tokens_input_total = 0.0

    meta_info["tokens_used_total"] = tokens_used_total
    meta_info["tokens_used_per_step"] = tokens_output_per_step
    meta_info["tokens_output_total"] = tokens_output_total
    meta_info["tokens_input_total"] = tokens_input_total
    meta_info["tokens_input_per_step"] = tokens_input_per_step
    meta_info["tokens_total_per_step"] = tokens_total_per_step
    meta_info["total_steps"] = len(tokens_total_per_step)

    meta_info.pop("tokens_output_total_no_thoughts", None)
    meta_info.pop("tokens_output_per_step_no_thoughts", None)
    meta_info.pop("tokens_thoughts_per_step", None)
    meta_info.pop("tokens_billing_total", None)

    return meta_info


def _aggregate_meta_info(meta_info_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_tool_calls = 0
    total_tokens = 0.0
    total_output_tokens = 0.0
    total_input_tokens = 0.0
    tool_usage_counts: Dict[str, int] = {}
    reach_max_tool_call_count = 0
    direct_answer_count = 0
    unique_tasks: set = set()

    for info in meta_info_list:
        total_tool_calls += info.get("function_call_total_count", 0)
        tokens_used_total = info.get("tokens_used_total")
        if isinstance(tokens_used_total, (int, float)):
            total_tokens += float(tokens_used_total)
        if info.get("total_steps", 0) >= MAX_TOOL_CALLS:
            reach_max_tool_call_count += 1
        if info.get("function_call_total_count", 0) == 0:
            direct_answer_count += 1
        for tool_name, count in info.get("function_call_each_count", {}).items():
            tool_usage_counts[tool_name] = tool_usage_counts.get(tool_name, 0) + count

        unique_tasks.add(info.get("id"))
        output_total = info.get("tokens_output_total")
        if isinstance(output_total, (int, float)):
            total_output_tokens += float(output_total)
        input_total = info.get("tokens_input_total")
        if isinstance(input_total, (int, float)):
            total_input_tokens += float(input_total)
        elif isinstance(tokens_used_total, (int, float)) and isinstance(output_total, (int, float)):
            total_input_tokens += float(tokens_used_total) - float(output_total)

    return {
        "total_tasks": len(unique_tasks),
        "total_trajectories": len(meta_info_list),
        "total_tool_calls": total_tool_calls,
        "total_tokens": total_tokens,
        "total_output_tokens": total_output_tokens,
        "total_input_tokens": total_input_tokens,
        "tool_usage_counts": tool_usage_counts,
        "reach_max_tool_call_count": reach_max_tool_call_count,
        "direct_answer_count": direct_answer_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fix meta_info token statistics from extra_info.jsonl for a parent directory."
    )
    parser.add_argument("parent_dir", type=str, help="Parent directory containing task subfolders.")
    parser.add_argument("--dry_run", action="store_true", help="Scan and report without writing files.")
    args = parser.parse_args()

    parent_dir = args.parent_dir
    if not os.path.isdir(parent_dir):
        raise FileNotFoundError(f"Parent dir not found: {parent_dir}")

    meta_info_paths = []
    for root, _, files in os.walk(parent_dir):
        if "meta_info.jsonl" in files and "extra_info.jsonl" in files:
            meta_info_paths.append(os.path.join(root, "meta_info.jsonl"))

    updated_meta_infos: List[Dict[str, Any]] = []
    updated_count = 0
    skipped_count = 0

    for meta_path in meta_info_paths:
        extra_path = os.path.join(os.path.dirname(meta_path), "extra_info.jsonl")
        meta_info = _load_first_json_line(meta_path)
        extra_records = _load_extra_info(extra_path)
        if not meta_info or not extra_records:
            skipped_count += 1
            continue

        new_meta = _recompute_tokens(meta_info, extra_records)
        updated_meta_infos.append(new_meta)
        if not args.dry_run:
            with open(meta_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(new_meta) + "\n")
        updated_count += 1

    global_paths = []
    global_meta_path = os.path.join(parent_dir, "global_meta_info.jsonl")
    if os.path.exists(global_meta_path):
        global_paths.append(global_meta_path)
    global_info_path = os.path.join(parent_dir, "global_info.jsonl")
    if os.path.exists(global_info_path):
        global_paths.append(global_info_path)

    if updated_meta_infos:
        global_info = _aggregate_meta_info(updated_meta_infos)
        if not global_paths:
            global_paths.append(global_meta_path)
        if not args.dry_run:
            for path in global_paths:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(json.dumps(global_info) + "\n")

    print(f"Updated meta_info: {updated_count}, skipped: {skipped_count}")
    if updated_meta_infos:
        print("Updated global info: " + ", ".join(global_paths))


if __name__ == "__main__":
    main()
