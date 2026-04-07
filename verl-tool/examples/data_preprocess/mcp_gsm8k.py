#!/usr/bin/env python3
"""
Convert the openai/gsm8k dataset into an MCP calculator training corpus.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, Any, List, Optional

from datasets import Dataset, load_dataset


SYSTEM_PROMPT = """You are a ReAct (Reasoning and Acting) agent connected to an MCP calculator server.
When arithmetic is necessary, think within <think>...</think>, then invoke the tool via <tool_call> tags.

Tool descriptions:
<tools>
{"name": "calculate", "description": "Evaluate an arithmetic expression and return the string result.", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "Expression to evaluate, e.g. (12 + 7) * 3"}}, "required": ["expression"]}}
</tools>

Response rules:
1. Use <think>...</think> to reason before every tool call or final answer.
2. Tool calls must look like:
<think>Explain why calculator is needed</think><tool_call>{"name":"calculate","arguments":{"expression":"(8 + 4) * 3"}}</tool_call>
3. Final answers must be wrapped in:
<think>Explain the final reasoning</think><answer>FINAL_NUMBER_OR_UNIT</answer>
4. Output must be valid XML; never omit the <answer> block.
5. Do NOT return JSON, words, or sentences in <answer>; only the final number (or unit if required).
"""

ANSWER_PATTERN = re.compile(r"####\s*(.+)", re.DOTALL)


def extract_final_answer(answer: str) -> str:
    """Grab the text that follows the '####' marker and trim whitespace."""
    match = ANSWER_PATTERN.search(answer.strip())
    if not match:
        raise ValueError("Unable to locate the '####' section in the answer text.")
    tail = match.group(1).strip()
    final_line = tail.splitlines()[0].strip()
    return final_line


def build_row(question: str, ground_truth: str, split: str, idx: int, full_answer: str) -> Dict[str, Any]:
    return {
        "data_source": "mcp_gsm8k",
        "ability": "mcp",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question.strip()},
        ],
        "reward_model": {
            "style": "rule",
            "ground_truth": ground_truth,
        },
        "extra_info": {
            "hf_split": split,
            "hf_index": idx,
            "question": question.strip(),
            "reference_answer": full_answer.strip(),
            "target_answer": ground_truth,
            "use_specified_server": True,
            "mcp_servers": [
                {"name": "calculator"}
            ],
        },
    }


def convert_split(split: str, out_dir: str, max_samples: Optional[int] = None, seed: int = 42) -> str:
    ds = load_dataset("openai/gsm8k", "main", split=split)
    if max_samples is not None and max_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(max_samples))

    rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(ds):
        question = item.get("question", "").strip()
        answer = (item.get("answer") or "").strip()
        if not question or not answer:
            continue
        try:
            final_answer = extract_final_answer(answer)
        except ValueError as e:
            raise ValueError(f"Split {split} index {idx}: {e}\nRaw answer:\n{answer}") from e
        rows.append(build_row(question, final_answer, split, idx, answer))

    if not rows:
        raise RuntimeError(f"Split {split} has no valid rows to persist.")

    os.makedirs(out_dir, exist_ok=True)
    file_name = "train.parquet" if split == "train" else "test.parquet"
    out_path = os.path.join(out_dir, file_name)
    Dataset.from_list(rows).to_parquet(out_path)
    print(f"[{split}] wrote {len(rows)} rows -> {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Convert openai/gsm8k to MCP calculator dataset.")
    parser.add_argument("--out_dir", type=str, default="data/mcp_gsm8k", help="Directory to save parquet files.")
    parser.add_argument("--max_train", type=int, default=None, help="Optional cap on the number of training samples.")
    parser.add_argument("--max_test", type=int, default=None, help="Optional cap on the number of test samples.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed used when sub-sampling.")
    args = parser.parse_args()

    train_path = convert_split("train", args.out_dir, args.max_train, args.seed)
    test_path = convert_split("test", args.out_dir, args.max_test, args.seed + 1)

    sample_preview = {
        "train_path": train_path,
        "test_path": test_path,
    }
    print("Conversion completed:")
    print(json.dumps(sample_preview, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
