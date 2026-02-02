import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _load_first_json_line(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    return {}


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _write_jsonl(path: str, items: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _last_numeric(values: List[Any]) -> Optional[float]:
    for value in reversed(values):
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _read_max_tool_calls() -> Optional[int]:
    constants_path = os.path.join(os.path.dirname(__file__), "..", "constants.py")
    if not os.path.exists(constants_path):
        return None
    try:
        with open(constants_path, "r", encoding="utf-8") as f:
            for line in f:
                if "MAX_TOOL_CALLS" in line:
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        value = parts[1].strip()
                        if value.isdigit():
                            return int(value)
    except OSError:
        return None
    return None


def _find_meta_info(
    fix_dir: str, case_id: str, traj_id: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    base_dir = os.path.join(fix_dir, str(case_id))
    direct_meta = os.path.join(base_dir, "meta_info.jsonl")
    if os.path.exists(direct_meta):
        return _load_first_json_line(direct_meta)

    if not os.path.isdir(base_dir):
        return None

    traj_dirs = []
    for name in os.listdir(base_dir):
        if name.startswith("traj_"):
            traj_dirs.append(os.path.join(base_dir, name))
    traj_dirs.sort()
    if not traj_dirs:
        return None

    if traj_id is not None:
        candidate = os.path.join(base_dir, f"traj_{traj_id}", "meta_info.jsonl")
        if os.path.exists(candidate):
            return _load_first_json_line(candidate)

    for tdir in traj_dirs:
        candidate = os.path.join(tdir, "meta_info.jsonl")
        if os.path.exists(candidate):
            return _load_first_json_line(candidate)
    return None


def _merge_eval_with_meta(eval_item: Dict[str, Any], meta: Dict[str, Any]) -> None:
    fields = [
        "image_path",
        "total_steps",
        "function_call_each_count",
        "function_call_total_count",
        "function_call_per_step",
        "tokens_used_total",
        "tokens_used_per_step",
        "tokens_output_total",
        "tokens_input_total",
        "tokens_input_per_step",
        "tokens_total_per_step",
    ]
    for key in fields:
        if key in meta:
            eval_item[key] = meta[key]


def _get_output_tokens_total(item: Dict[str, Any]) -> Optional[float]:
    tokens_output_total = item.get("tokens_output_total")
    if isinstance(tokens_output_total, (int, float)):
        return float(tokens_output_total)
    per_step = item.get("tokens_used_per_step")
    if isinstance(per_step, list):
        values = [v for v in per_step if isinstance(v, (int, float))]
        if values:
            return float(sum(values))
    return None


def _get_input_tokens_total(item: Dict[str, Any]) -> Optional[float]:
    input_total = item.get("tokens_input_total")
    if isinstance(input_total, (int, float)):
        return float(input_total)
    tokens_used_total = item.get("tokens_used_total")
    output_total = _get_output_tokens_total(item)
    if isinstance(tokens_used_total, (int, float)) and isinstance(output_total, (int, float)):
        value = float(tokens_used_total) - float(output_total)
        if value < 0:
            value = 0.0
        return value
    per_step = item.get("tokens_input_per_step")
    if isinstance(per_step, list):
        last_idx = None
        last_input = None
        for idx in range(len(per_step) - 1, -1, -1):
            value = per_step[idx]
            if isinstance(value, (int, float)):
                last_idx = idx
                last_input = float(value)
                break
        if last_input is None:
            return None
        outputs = item.get("tokens_used_per_step")
        if isinstance(outputs, list) and last_idx is not None:
            output_before = sum(
                v for v in outputs[:last_idx] if isinstance(v, (int, float))
            )
            input_total = last_input - float(output_before)
            if input_total < 0:
                input_total = 0.0
            return float(input_total)
        return float(last_input)
    return None


def _get_total_tokens(item: Dict[str, Any]) -> Optional[float]:
    tokens_used_total = item.get("tokens_used_total")
    if isinstance(tokens_used_total, (int, float)):
        return float(tokens_used_total)
    per_step = item.get("tokens_total_per_step")
    if isinstance(per_step, list):
        values = [v for v in per_step if isinstance(v, (int, float))]
        if values:
            return float(sum(values))
    output_total = _get_output_tokens_total(item)
    input_total = _get_input_tokens_total(item)
    if output_total is not None and input_total is not None:
        return float(output_total + input_total)
    return None


def _compute_tool_combination_statistics(eval_results: List[Dict[str, Any]]) -> str:
    stats: Dict[str, Dict[str, float]] = {}

    for item in eval_results:
        result = item.get("result")
        if isinstance(result, dict) and result.get("is_filter"):
            continue
        func_counts = item.get("function_call_each_count", {}) or {}
        used = sorted([t for t, c in func_counts.items() if c > 0])
        category = "+".join(used) if used else "no_tool"
        if category not in stats:
            stats[category] = {
                "total": 0,
                "correct": 0,
                "token_total_sum": 0.0,
                "token_total_count": 0,
                "token_correct_sum": 0.0,
                "token_correct_count": 0,
            }
        s = stats[category]
        s["total"] += 1
        output_total = _get_output_tokens_total(item)
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
            s["token_total_sum"] / s["token_total_count"] if s["token_total_count"] > 0 else 0.0
        )
        avg_tokens_correct = (
            s["token_correct_sum"] / s["token_correct_count"] if s["token_correct_count"] > 0 else 0.0
        )
        lines.append(
            "  "
            + f"{cat}: total={s['total']}, correct={s['correct']}, accuracy={acc:.4f}, "
            + f"avg_tokens={avg_tokens:.2f}, avg_tokens_correct={avg_tokens_correct:.2f}"
        )
    lines.append("=" * 60)
    return "\n".join(lines)


def _update_summary(summary_path: str, eval_results: List[Dict[str, Any]]) -> None:
    total = 0
    correct = 0
    filtered = 0
    output_tokens_sum = 0.0
    input_tokens_sum = 0.0
    total_tokens_sum = 0.0
    output_tokens_count = 0
    input_tokens_count = 0
    total_tokens_count = 0

    for item in eval_results:
        result = item.get("result")
        is_filter = isinstance(result, dict) and result.get("is_filter")
        if is_filter:
            filtered += 1
        else:
            total += 1
            if result == 1.0:
                correct += 1

        output_total = _get_output_tokens_total(item)
        if isinstance(output_total, (int, float)):
            output_tokens_sum += float(output_total)
            output_tokens_count += 1
        input_total = _get_input_tokens_total(item)
        if isinstance(input_total, (int, float)):
            input_tokens_sum += float(input_total)
            input_tokens_count += 1
        total_total = _get_total_tokens(item)
        if isinstance(total_total, (int, float)):
            total_tokens_sum += float(total_total)
            total_tokens_count += 1

    accuracy = (correct / total) if total else 0.0
    tool_stats_text = _compute_tool_combination_statistics(eval_results)
    avg_output = output_tokens_sum / output_tokens_count if output_tokens_count else 0.0
    avg_input = input_tokens_sum / input_tokens_count if input_tokens_count else 0.0
    avg_total = total_tokens_sum / total_tokens_count if total_tokens_count else 0.0

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"evaluated={total}\n")
        f.write(f"correct={correct}\n")
        f.write(f"filtered={filtered}\n")
        f.write(f"accuracy={accuracy:.6f}\n")
        f.write(f"total_output_tokens={output_tokens_sum:.0f}\n")
        f.write(f"total_input_tokens={input_tokens_sum:.0f}\n")
        f.write(f"total_tokens={total_tokens_sum:.0f}\n")
        f.write(f"avg_output_tokens={avg_output:.2f}\n")
        f.write(f"avg_input_tokens={avg_input:.2f}\n")
        f.write(f"avg_total_tokens={avg_total:.2f}\n")
        f.write(tool_stats_text)


def _aggregate_fix_meta(meta_infos: List[Dict[str, Any]]) -> Dict[str, Any]:
    max_tool_calls = _read_max_tool_calls()
    total_tool_calls = 0
    total_tokens = 0.0
    tool_usage_counts: Dict[str, int] = {}
    reach_max_tool_call_count = 0
    direct_answer_count = 0
    unique_tasks: set = set()
    total_output_tokens = 0.0
    total_input_tokens = 0.0

    for info in meta_infos:
        total_tool_calls += int(info.get("function_call_total_count", 0) or 0)
        tokens_used_total = info.get("tokens_used_total")
        if isinstance(tokens_used_total, (int, float)):
            total_tokens += float(tokens_used_total)
        if max_tool_calls is not None and info.get("total_steps", 0) >= max_tool_calls:
            reach_max_tool_call_count += 1
        if info.get("function_call_total_count", 0) == 0:
            direct_answer_count += 1
        for tool_name, count in (info.get("function_call_each_count") or {}).items():
            tool_usage_counts[tool_name] = tool_usage_counts.get(tool_name, 0) + int(count)

        unique_tasks.add(info.get("id"))

        output_total = info.get("tokens_output_total")
        if isinstance(output_total, (int, float)):
            total_output_tokens += float(output_total)

        input_total = info.get("tokens_input_total")
        if isinstance(input_total, (int, float)):
            total_input_tokens += float(input_total)
        elif isinstance(tokens_used_total, (int, float)) and isinstance(output_total, (int, float)):
            value = float(tokens_used_total) - float(output_total)
            if value < 0:
                value = 0.0
            total_input_tokens += value

    return {
        "total_tasks": len(unique_tasks),
        "total_trajectories": len(meta_infos),
        "total_tool_calls": total_tool_calls,
        "total_tokens": total_tokens,
        "tool_usage_counts": tool_usage_counts,
        "reach_max_tool_call_count": reach_max_tool_call_count,
        "direct_answer_count": direct_answer_count,
        "total_output_tokens": total_output_tokens,
        "total_input_tokens": total_input_tokens,
    }


def _update_fix_global_meta(fix_dir: str) -> None:
    global_meta_path = os.path.join(fix_dir, "global_meta_info.jsonl")
    meta_infos: List[Dict[str, Any]] = []
    for root, _, files in os.walk(fix_dir):
        if "meta_info.jsonl" not in files:
            continue
        meta_path = os.path.join(root, "meta_info.jsonl")
        meta_infos.append(_load_first_json_line(meta_path))

    if not meta_infos:
        return

    global_info = _aggregate_fix_meta(meta_infos)
    with open(global_meta_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(global_info) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update eval results and summary using *_fix meta_info, and refresh *_fix global meta."
    )
    parser.add_argument(
        "--root", type=str, default="AReaL", help="Root directory containing eval_*_result and data folders."
    )
    args = parser.parse_args()

    root = args.root
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Root dir not found: {root}")

    eval_dirs = [
        name
        for name in os.listdir(root)
        if name.startswith("eval_") and name.endswith("_result") and os.path.isdir(os.path.join(root, name))
    ]

    for eval_dir_name in sorted(eval_dirs):
        eval_dir = os.path.join(root, eval_dir_name)
        dataset_name = eval_dir_name[len("eval_") : -len("_result")]
        fix_dir = os.path.join(root, f"{dataset_name}_fix")
        eval_path = os.path.join(eval_dir, "eval_result.jsonl")
        summary_path = os.path.join(eval_dir, "summary.txt")
        if not os.path.exists(eval_path):
            continue
        if not os.path.isdir(fix_dir):
            continue

        eval_items = _load_jsonl(eval_path)
        updated = 0
        for item in eval_items:
            case_id = str(item.get("id"))
            traj_id = item.get("traj_id")
            meta = _find_meta_info(fix_dir, case_id, traj_id if isinstance(traj_id, int) else None)
            if meta is None:
                continue
            _merge_eval_with_meta(item, meta)
            updated += 1

        _write_jsonl(eval_path, eval_items)
        _update_summary(summary_path, eval_items)

        _update_fix_global_meta(fix_dir)

        print(f"{eval_dir_name}: updated {updated}/{len(eval_items)}")


if __name__ == "__main__":
    main()
