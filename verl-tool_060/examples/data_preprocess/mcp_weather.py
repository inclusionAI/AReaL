#!/usr/bin/env python3
"""
Generate a dynamic MCP weather dataset.
Each sample carries validation_calls that will be executed at eval time
to grade the model output against live MCP responses.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

from datasets import Dataset


SYSTEM_PROMPT = """You are a ReAct agent connected to an MCP weather server.
Always use the tool to retrieve data, then produce the final answer in <answer> tags.

Tool:
<tools>
{"name": "get_alerts", "description": "Get weather alerts for a US state (two-letter code, e.g., CA).", "parameters": {"type": "object", "properties": {"state": {"type": "string"}}, "required": ["state"]}}
</tools>

Response rules:
1) Think before tools and answers: <think>...</think>
2) Tool calls: <think>why</think><tool_call>{"server":"weather","name":"get_alerts","arguments":{"state":"CA"}}</tool_call>
3) Final answer: <think>final reasoning</think><answer>Final concise alert summary</answer>
4) Do not fabricate alerts; always rely on tool output.
"""


STATE_PROMPTS = [
    ("CA", "Use the weather tool to report any active alerts for California. If none, say there are no active alerts."),
    ("TX", "Fetch active weather alerts for Texas using the weather MCP tool and summarize."),
    ("NY", "Using the weather tool, list active alerts for New York state; if none, state clearly."),
    ("FL", "Check current alerts for Florida via the weather MCP and return a concise summary."),
    ("WA", "Query active alerts for Washington state with the weather tool and report the result."),
    ("OR", "Call the weather MCP to get current alerts for Oregon and summarize."),
    ("CO", "Retrieve active alerts for Colorado using the weather tool; if none, say so."),
    ("NV", "Using the weather MCP, report any alerts for Nevada; mention if no alerts."),
    ("AZ", "Query the weather tool for active alerts in Arizona and summarize the output."),
    ("IL", "Call the weather MCP to check active alerts for Illinois and give a brief answer."),
]


def build_row(state: str, question: str, split: str, idx: int) -> Dict[str, Any]:
    validation_calls = [
        {
            "server": "weather",
            "name": "get_alerts",
            "arguments": {"state": state},
        }
    ]
    return {
        "data_source": "mcp_weather",
        "ability": "mcp",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question.strip()},
        ],
        "reward_model": {
            "style": "dynamic",
            "ground_truth": "",  # evaluated dynamically via validation_calls
        },
        "extra_info": {
            "hf_split": split,
            "hf_index": idx,
            "state": state,
            "validation_calls": validation_calls,
            "use_specified_server": True,
            "mcp_servers": [{"name": "weather"}],
        },
    }


def convert_split(states: List[tuple[str, str]], out_dir: str, split: str, max_samples: Optional[int]) -> str:
    rows: List[Dict[str, Any]] = []
    for idx, (state, question) in enumerate(states):
        if max_samples is not None and idx >= max_samples:
            break
        rows.append(build_row(state, question, split, idx))

    if not rows:
        raise RuntimeError(f"Split {split} has no rows.")

    os.makedirs(out_dir, exist_ok=True)
    file_name = "train.parquet" if split == "train" else "test.parquet"
    out_path = os.path.join(out_dir, file_name)
    Dataset.from_list(rows).to_parquet(out_path)
    print(f"[{split}] wrote {len(rows)} rows -> {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate MCP weather dataset with dynamic validation.")
    parser.add_argument("--out_dir", type=str, default="data/mcp_weather", help="Directory to save parquet files.")
    parser.add_argument("--max_train", type=int, default=None, help="Optional cap on training samples.")
    parser.add_argument("--max_test", type=int, default=None, help="Optional cap on test samples.")
    args = parser.parse_args()

    train_path = convert_split(STATE_PROMPTS, args.out_dir, "train", args.max_train)
    test_path = convert_split(list(reversed(STATE_PROMPTS)), args.out_dir, "test", args.max_test)

    preview = {"train_path": train_path, "test_path": test_path}
    print("Conversion completed:")
    print(json.dumps(preview, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
