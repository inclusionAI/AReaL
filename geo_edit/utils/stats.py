from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional

from geo_edit.constants import MAX_TOOL_CALLS


def get_output_tokens_total(item: Dict) -> Optional[float]:
    tokens_output_total = item.get("tokens_output_total")
    if isinstance(tokens_output_total, (int, float)):
        return float(tokens_output_total)
    per_step = item.get("tokens_used_per_step")
    if isinstance(per_step, list):
        values = [v for v in per_step if isinstance(v, (int, float))]
        if values:
            return float(sum(values))
    return None


def get_input_tokens_total(item: Dict) -> Optional[float]:
    input_total = item.get("tokens_input_total")
    if isinstance(input_total, (int, float)):
        return float(input_total)
    tokens_used_total = item.get("tokens_used_total")
    output_total = get_output_tokens_total(item)
    if isinstance(tokens_used_total, (int, float)) and isinstance(
        output_total, (int, float)
    ):
        value = float(tokens_used_total) - float(output_total)
        return value if value > 0 else 0.0
    per_step = item.get("tokens_input_per_step")
    if isinstance(per_step, list):
        last_idx, last_input = None, None
        for idx in range(len(per_step) - 1, -1, -1):
            value = per_step[idx]
            if isinstance(value, (int, float)):
                last_idx, last_input = idx, float(value)
                break
        if last_input is None:
            return None
        outputs = item.get("tokens_used_per_step")
        if isinstance(outputs, list) and last_idx is not None:
            output_before = sum(
                v for v in outputs[:last_idx] if isinstance(v, (int, float))
            )
            input_total = last_input - float(output_before)
            return input_total if input_total > 0 else 0.0
        return last_input
    return None


def get_total_tokens(item: Dict) -> Optional[float]:
    tokens_used_total = item.get("tokens_used_total")
    if isinstance(tokens_used_total, (int, float)):
        return float(tokens_used_total)
    per_step = item.get("tokens_total_per_step")
    if isinstance(per_step, list):
        values = [v for v in per_step if isinstance(v, (int, float))]
        if values:
            return float(sum(values))
    output_total = get_output_tokens_total(item)
    input_total = get_input_tokens_total(item)
    return (
        float(output_total + input_total)
        if output_total is not None and input_total is not None
        else None
    )


def compute_tool_combination_statistics(eval_results: List[Dict]) -> str:
    stats: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {
            "total": 0,
            "correct": 0,
            "token_total_sum": 0.0,
            "token_total_count": 0,
            "token_correct_sum": 0.0,
            "token_correct_count": 0,
        }
    )
    for item in eval_results:
        result = item.get("result")
        if isinstance(result, dict) and result.get("is_filter"):
            continue
        func_counts = item.get("function_call_each_count") or {}
        used = sorted([t for t, c in func_counts.items() if c > 0])
        category = "+".join(used) if used else "no_tool"
        s = stats[category]
        s["total"] += 1
        output_total = get_output_tokens_total(item)
        if isinstance(output_total, (int, float)):
            s["token_total_sum"] += float(output_total)
            s["token_total_count"] += 1
        if result == 1.0:
            s["correct"] += 1
            if isinstance(output_total, (int, float)):
                s["token_correct_sum"] += float(output_total)
                s["token_correct_count"] += 1
    lines = ["\n" + "=" * 60, "Tool Combination Statistics", "=" * 60]
    for cat in sorted(stats.keys(), key=lambda x: (x != "no_tool", x)):
        s = stats[cat]
        acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
        avg_tokens = (
            s["token_total_sum"] / s["token_total_count"]
            if s["token_total_count"] > 0
            else 0.0
        )
        avg_tokens_correct = (
            s["token_correct_sum"] / s["token_correct_count"]
            if s["token_correct_count"] > 0
            else 0.0
        )
        lines.append(
            "  "
            + f"{cat}: total={s['total']}, correct={s['correct']}, accuracy={acc:.4f}, "
            + f"avg_tokens={avg_tokens:.2f}, avg_tokens_correct={avg_tokens_correct:.2f}"
        )
    lines.append("=" * 60)
    return "\n".join(lines)


def aggregate_meta_info(
    meta_info_list: List[Dict[str, Any]], max_tool_calls: int = MAX_TOOL_CALLS
) -> Dict[str, Any]:
    total_tool_calls = 0
    total_tokens = 0.0
    total_output_tokens = 0.0
    total_input_tokens = 0.0
    has_io_totals = False
    tool_usage_counts: Dict[str, int] = {}
    reach_max_tool_call_count = 0
    direct_answer_count = 0
    unique_tasks: set = set()
    for info in meta_info_list:
        total_tool_calls += int(info.get("function_call_total_count", 0) or 0)
        tokens_used_total = info.get("tokens_used_total")
        if isinstance(tokens_used_total, (int, float)):
            total_tokens += float(tokens_used_total)
        if info.get("total_steps", 0) >= max_tool_calls:
            reach_max_tool_call_count += 1
        if info.get("function_call_total_count", 0) == 0:
            direct_answer_count += 1
        for tool_name, count in info.get("function_call_each_count", {}).items():
            tool_usage_counts[tool_name] = tool_usage_counts.get(tool_name, 0) + int(
                count
            )
        if "id" in info:
            unique_tasks.add(info["id"])
        output_total = info.get("tokens_output_total")
        if isinstance(output_total, (int, float)):
            total_output_tokens += float(output_total)
        input_total = info.get("tokens_input_total")
        if isinstance(input_total, (int, float)):
            total_input_tokens += float(input_total)
            has_io_totals = True
        elif isinstance(tokens_used_total, (int, float)) and isinstance(
            output_total, (int, float)
        ):
            value = float(tokens_used_total) - float(output_total)
            total_input_tokens += value if value > 0 else 0.0
            has_io_totals = True
    if has_io_totals:
        total_tokens = total_output_tokens + total_input_tokens
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


def save_global_meta_info(
    output_path: str,
    meta_info_list: List[Dict[str, Any]],
    max_tool_calls: int = MAX_TOOL_CALLS,
) -> None:
    global_meta_info = aggregate_meta_info(
        meta_info_list, max_tool_calls=max_tool_calls
    )
    global_meta_info_jsonl_path = os.path.join(output_path, "global_meta_info.jsonl")
    with open(global_meta_info_jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(global_meta_info, ensure_ascii=False) + "\n")
