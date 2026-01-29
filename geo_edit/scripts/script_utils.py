"""Common utilities for generation scripts."""

import json
import os
from typing import Any, Dict, List

from geo_edit.constants import MAX_TOOL_CALLS


def aggregate_meta_info(meta_info_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate statistics from a list of meta_info dictionaries.

    Args:
        meta_info_list: List of meta_info dictionaries from saved trajectories.

    Returns:
        Dictionary containing aggregated statistics.
    """
    total_tool_calls = 0
    total_tokens = 0
    tool_usage_counts: Dict[str, int] = {}
    reach_max_tool_call_count = 0
    direct_answer_count = 0

    for info in meta_info_list:
        total_tool_calls += info["function_call_total_count"]
        total_tokens += info["tokens_used_total"]
        if info["total_steps"] >= MAX_TOOL_CALLS:
            reach_max_tool_call_count += 1
        if info["function_call_total_count"] == 0:
            direct_answer_count += 1
        for tool_name, count in info["function_call_each_count"].items():
            tool_usage_counts[tool_name] = tool_usage_counts.get(tool_name, 0) + count

    return {
        "total_examples": len(meta_info_list),
        "total_tool_calls": total_tool_calls,
        "total_tokens": total_tokens,
        "tool_usage_counts": tool_usage_counts,
        "reach_max_tool_call_count": reach_max_tool_call_count,
        "direct_answer_count": direct_answer_count,
    }


def save_global_meta_info(output_path: str, meta_info_list: List[Dict[str, Any]]) -> None:
    """Save aggregated global meta info to a JSONL file.

    Args:
        output_path: Directory to save the global_meta_info.jsonl file.
        meta_info_list: List of meta_info dictionaries from saved trajectories.
    """
    global_meta_info = aggregate_meta_info(meta_info_list)
    global_meta_info_jsonl_path = os.path.join(output_path, "global_meta_info.jsonl")
    with open(global_meta_info_jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(global_meta_info) + "\n")


def load_existing_meta_info(task_save_dir: str) -> Dict[str, Any] | None:
    """Load existing meta_info from a task directory if it exists.

    Args:
        task_save_dir: Directory containing the task's saved data.

    Returns:
        The meta_info dictionary if it exists, None otherwise.
    """
    meta_path = os.path.join(task_save_dir, "meta_info.jsonl")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.loads(f.readline().strip())
    return None
