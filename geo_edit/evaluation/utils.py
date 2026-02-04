from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

ANSWER_TEMPLATE = "<answer>{}</answer>"


def parse_score(text: str) -> str:
    match = re.search(r"\bscore\s*:\s*([01])\b", text, re.IGNORECASE)
    return match.group(1) if match else ""


def extract_answer(text: str, mode: str) -> Optional[str]:
    parts = ANSWER_TEMPLATE.split("{}")
    if mode == "split":
        if parts[0] not in text or parts[1] not in text:
            return None
        return text.split(parts[0])[-1].split(parts[1])[0].strip()
    if mode == "strict":
        start = text.find(parts[0])
        if start == -1:
            return None
        start += len(parts[0])
        if parts[1]:
            end = text.find(parts[1], start)
            if end == -1:
                return None
            return text[start:end].strip()
        return text[start:].strip()
    raise ValueError(f"Unknown extract mode: {mode}")


def get_final_prediction(predict_str_list: List[str], extract_mode: Optional[str]) -> str:
    if not predict_str_list:
        return ""
    last = predict_str_list[-1].strip()
    if not extract_mode:
        return last
    extracted = extract_answer(last, extract_mode)
    return extracted if extracted is not None else last


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
    if output_total is not None and input_total is not None:
        return float(output_total + input_total)
    return None


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
        func_counts = item.get("function_call_each_count", {})
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


def iter_meta_info_files(result_path: str) -> Iterable[str]:
    for name in os.listdir(result_path):
        subdir = os.path.join(result_path, name)
        if not os.path.isdir(subdir):
            continue
        meta_path = os.path.join(subdir, "meta_info.jsonl")
        if os.path.isfile(meta_path):
            yield meta_path


def load_records(meta_path: str) -> Iterable[dict]:
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON line in {meta_path}: {line}")
