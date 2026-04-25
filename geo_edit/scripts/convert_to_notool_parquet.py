"""Convert tool-calling RL parquet datasets to notool versions.

Replaces:
  - system prompt: TOOL_CALL_SYSTEM_PROMPT + tool defs → VLLM_NO_TOOL_SYSTEM_PROMPT
  - user prompt: removes tool-execution references from instructions

Usage:
    python -m geo_edit.scripts.convert_to_notool_parquet
"""

import re
import pandas as pd

NOTOOL_SYSTEM_PROMPT = (
    "You are an advanced AI assistant capable of complex reasoning.\n"
    "You must strictly adhere to the following protocol:\n\n"
    "1. Reasoning Process: Before providing your answer, analyze the\n"
    "problem step by step. Output your reasoning inside <think> and </think> tags.\n\n"
    "2. Final Output: When you have formulated your conclusion,\n"
    "wrap your final answer in <answer> and </answer> tags."
)

TOOL_USER_INSTRUCTIONS = (
    "Please provide a complete step-by-step solution to\n"
    "this problem. Your reasoning should:\n"
    "1. Analyze the problem systematically\n"
    "2. Check if the tool execution and answer are correct\n"
    "3. If there are errors, explain what went wrong and\n"
    "provide the correct reasoning\n"
    "4. Provide the final answer\n"
    "Use natural expressions like 'let me think' or 'hmm'\n"
    "when helpful, but keep it concise. It's encouraged\n"
    "to use self-reflection or verification especially in the\n"
    "verifying tool output in the reasoning process.\n"
    "Provide your detailed reasoning between <think>\n"
    "and </think> tags, then give your final answer\n"
    "between <answer> and </answer> tags."
)

NOTOOL_USER_INSTRUCTIONS = (
    "Please provide a complete step-by-step solution to\n"
    "this problem. Your reasoning should:\n"
    "1. Analyze the problem systematically\n"
    "2. Show clear logical steps\n"
    "3. Provide the final answer\n"
    "Use natural expressions like 'let me think' or 'hmm'\n"
    "when helpful, but keep it concise.\n"
    "Provide your detailed reasoning between <think>\n"
    "and </think> tags, then give your final answer\n"
    "between <answer> and </answer> tags."
)


def convert_prompt(msgs):
    """Convert a single prompt (list of message dicts) to notool version."""
    out = []
    for m in msgs:
        m = dict(m)
        if m["role"] == "system":
            m["content"] = NOTOOL_SYSTEM_PROMPT
        elif m["role"] == "user":
            text = m["content"]
            text = text.replace(TOOL_USER_INSTRUCTIONS, NOTOOL_USER_INSTRUCTIONS)
            # fallback: regex strip any remaining tool-execution references
            text = re.sub(
                r"2\.\s*Check if the tool execution and answer are correct\n"
                r"3\.\s*If there are errors, explain what went wrong and\n"
                r"provide the correct reasoning\n",
                "2. Show clear logical steps\n",
                text,
            )
            text = re.sub(
                r"to use self-reflection or verification especially in the\n"
                r"verifying tool output in the reasoning process\.",
                "to use self-reflection or verification in the reasoning process.",
                text,
            )
            m["content"] = text
        out.append(m)
    return out


def convert_file(src: str, dst: str):
    df = pd.read_parquet(src)
    df["prompt"] = df["prompt"].apply(lambda msgs: convert_prompt(list(msgs)))
    df.to_parquet(dst, index=False)
    print(f"{src} -> {dst}  ({len(df)} rows)")


FILES = [
    (
        "/storage/openpsi/data/reasonmap_rl/combined_train_rl_only.parquet",
        "/storage/openpsi/data/reasonmap_rl/combined_train_rl_only_notool.parquet",
    ),
    (
        "/storage/openpsi/data/lcy_image_edit/mixed_rl/new_train.parquet",
        "/storage/openpsi/data/lcy_image_edit/mixed_rl/new_train_notool.parquet",
    ),
    (
        "/storage/openpsi/data/reasonmap_rl/combined_test_10pct.parquet",
        "/storage/openpsi/data/reasonmap_rl/combined_test_10pct_notool.parquet",
    ),
    (
        "/storage/openpsi/data/lcy_image_edit/mixed_rl/new_val.parquet",
        "/storage/openpsi/data/lcy_image_edit/mixed_rl/new_val_notool.parquet",
    ),
    (
        "/storage/openpsi/data/lcy_image_edit/mixed_rl/mapqa_val_200.parquet",
        "/storage/openpsi/data/lcy_image_edit/mixed_rl/mapqa_val_200_notool.parquet",
    ),
]

if __name__ == "__main__":
    for src, dst in FILES:
        convert_file(src, dst)
