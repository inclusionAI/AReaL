from __future__ import annotations

import argparse
import json
import os
from typing import Dict


def convert(input_jsonl: str, output_dir: str) -> int:
    os.makedirs(output_dir, exist_ok=True)
    seen: Dict[str, int] = {}
    count = 0
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            task_id = str(rec["id"])
            seen[task_id] = seen.get(task_id, 0) + 1
            if seen[task_id] > 1:
                continue

            out = {
                "id": task_id,
                "question": rec.get("question", ""),
                "answer": rec.get("ground_truth", rec.get("answer", "")),
                "output_text": rec.get("model_response") or rec.get("predicted", ""),
                "image_path": rec.get("image_path"),
                "total_steps": rec.get("num_turns") or rec.get("total_steps", 1),
                "function_call_total_count": rec.get("num_crops") or (1 if rec.get("tool_used") else 0),
                "function_call_each_count": {},
                "function_call_per_step": [],
                "tokens_input_total": 0,
                "tokens_output_total": 0,
                "tokens_used_total": 0,
                "tokens_input_per_step": [],
                "tokens_output_per_step": [],
                "tokens_used_per_step": [],
                "tokens_total_per_step": [],
            }

            subdir = os.path.join(output_dir, task_id)
            os.makedirs(subdir, exist_ok=True)
            with open(os.path.join(subdir, "meta_info.jsonl"), "w", encoding="utf-8") as g:
                g.write(json.dumps(out, ensure_ascii=False) + "\n")
            count += 1
    return count


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="baseline inference output jsonl")
    p.add_argument("--output_dir", required=True, help="target dir matching eval_unified layout")
    args = p.parse_args()
    n = convert(args.input, args.output_dir)
    print(f"Converted {n} records -> {args.output_dir}")


if __name__ == "__main__":
    main()
