import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Optional

from geo_edit.utils.stats import aggregate_meta_info, compute_tool_combination_statistics, get_input_tokens_total, get_output_tokens_total, get_total_tokens


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
            if line:
                items.append(json.loads(line))
    return items


def _write_jsonl(path: str, items: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _find_meta_info(fix_dir: str, case_id: str, traj_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
    base_dir = os.path.join(fix_dir, case_id)
    direct_meta = os.path.join(base_dir, "meta_info.jsonl")
    if os.path.exists(direct_meta):
        return _load_first_json_line(direct_meta)
    if not os.path.isdir(base_dir):
        return None
    if traj_id is not None:
        candidate = os.path.join(base_dir, f"traj_{traj_id}", "meta_info.jsonl")
        if os.path.exists(candidate):
            return _load_first_json_line(candidate)
    traj_dirs = sorted(os.path.join(base_dir, name) for name in os.listdir(base_dir) if name.startswith("traj_"))
    for tdir in traj_dirs:
        candidate = os.path.join(tdir, "meta_info.jsonl")
        if os.path.exists(candidate):
            return _load_first_json_line(candidate)
    return None


def _merge_eval_with_meta(eval_item: Dict[str, Any], meta: Dict[str, Any]) -> None:
    fields = ["image_path", "total_steps", "function_call_each_count", "function_call_total_count", "function_call_per_step", "tokens_used_total", "tokens_used_per_step", "tokens_output_total", "tokens_input_total", "tokens_input_per_step", "tokens_total_per_step"]
    for key in fields:
        if key in meta:
            eval_item[key] = meta[key]


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
        if isinstance(result, dict) and result.get("is_filter"):
            filtered += 1
        else:
            total += 1
            if result == 1.0:
                correct += 1
        output_total = get_output_tokens_total(item)
        if isinstance(output_total, (int, float)):
            output_tokens_sum += output_total
            output_tokens_count += 1
        input_total = get_input_tokens_total(item)
        if isinstance(input_total, (int, float)):
            input_tokens_sum += input_total
            input_tokens_count += 1
        total_total = get_total_tokens(item)
        if isinstance(total_total, (int, float)):
            total_tokens_sum += total_total
            total_tokens_count += 1

    accuracy = correct / total if total else 0.0
    tool_stats_text = compute_tool_combination_statistics(eval_results)
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


def _update_fix_global_meta(fix_dir: str) -> None:
    global_meta_path = os.path.join(fix_dir, "global_meta_info.jsonl")
    meta_infos: List[Dict[str, Any]] = []
    for root, _, files in os.walk(fix_dir):
        if "meta_info.jsonl" in files:
            meta_infos.append(_load_first_json_line(os.path.join(root, "meta_info.jsonl")))
    if not meta_infos:
        return
    with open(global_meta_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(aggregate_meta_info(meta_infos)) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Update eval results and summary using *_fix meta_info, and refresh *_fix global meta.")
    parser.add_argument("--root", type=str, default="AReaL", help="Root directory containing eval_*_result and data folders.")
    args = parser.parse_args()
    root = args.root
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Root dir not found: {root}")

    eval_dirs = [name for name in os.listdir(root) if name.startswith("eval_") and name.endswith("_result") and os.path.isdir(os.path.join(root, name))]
    for eval_dir_name in sorted(eval_dirs):
        eval_dir = os.path.join(root, eval_dir_name)
        dataset_name = eval_dir_name[len("eval_") : -len("_result")]
        fix_dir = os.path.join(root, f"{dataset_name}_fix")
        eval_path = os.path.join(eval_dir, "eval_result.jsonl")
        summary_path = os.path.join(eval_dir, "summary.txt")
        if not os.path.exists(eval_path) or not os.path.isdir(fix_dir):
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
