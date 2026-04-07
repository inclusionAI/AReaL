#!/usr/bin/env python3
"""
Judge test harness for MCP‑Universe evaluators using prepared <answer> outputs.

Usage examples:
  python run.py \
    --task test/browser_automation/playwright_paper_task_0001.json \
    --answer_file answers/browser_automation_playwright_paper_task_0001.json

  python run.py \
    --task test/web_search/info_search_task_0011.json \
    --answer_text "Cigarette"

Notes:
- If you pass JSON via --answer_file, it will be put inside <answer>...</answer>.
- The harness extracts <answer> content before calling MCP‑Universe evaluators (same as our RM).
- For web_search, set OPENAI_API_KEY in environment.
"""
from __future__ import annotations

import os
import re
import json
import argparse
from typing import Any, List

from verl_tool.workers.reward_manager.mcp_universe_eval import eval_answer_for_task


DEFAULT_CONFIGS_ROOT = "benchmarks/MCP-Universe/mcpuniverse/benchmark/configs"


def _load_answer_text(args: argparse.Namespace) -> str:
    if args.answer_text is not None:
        return str(args.answer_text)
    if args.answer_file:
        with open(args.answer_file, "r", encoding="utf-8") as f:
            txt = f.read().strip()
        # If file appears to be JSON, keep as-is; otherwise raw text
        try:
            _ = json.loads(txt)
            return txt
        except Exception:
            return txt
    raise ValueError("Provide --answer_text or --answer_file")


def _wrap_in_answer_block(s: str) -> str:
    return f"<answer>{s}</answer>"


def _extract_answer_block(s: str) -> str:
    m = list(re.finditer(r"<answer>(.*?)</answer>", s, re.DOTALL))
    if not m:
        return s
    return m[-1].group(1).strip()


def _resolve_task_path(task: str, configs_root: str) -> str:
    if os.path.isabs(task):
        return task
    # Allow both relative to configs root or already rooted under it
    p1 = os.path.join(configs_root, task)
    return p1 if os.path.exists(p1) else task


# No local scoring; reuse RM helper's summary


def main():
    ap = argparse.ArgumentParser(description="Judge <answer> outputs against MCP‑Universe evaluators")
    ap.add_argument("--task", required=True, help="Task JSON path (relative to configs_root or absolute)")
    ap.add_argument("--configs_root", default=DEFAULT_CONFIGS_ROOT, help="Configs root folder")
    ap.add_argument("--answer_text", default=None, help="Inline answer text (inside <answer>)")
    ap.add_argument("--answer_file", default=None, help="Path to a file containing answer JSON or text")
    args = ap.parse_args()

    task_path = _resolve_task_path(args.task, args.configs_root)
    if not os.path.exists(task_path):
        raise FileNotFoundError(f"Task file not found: {task_path}")

    # Build final response and extract payload like the RM
    answer_raw = _load_answer_text(args)
    response = _wrap_in_answer_block(answer_raw)
    payload = _extract_answer_block(response)

    # Evaluate via reward manager helper
    summary = eval_answer_for_task(response, args.task, configs_root=args.configs_root)
    s = float(summary["score"])
    details = summary["details"]
    passed = int(summary["passed"])
    total = int(summary["total"])

    # Pretty print
    print("Task:", args.task)
    print("Payload (first 200 chars):", payload[:200] + ("..." if len(payload) > 200 else ""))
    print("---- Evaluators ----")
    for idx, d in enumerate(details):
        print(f"[{idx}] func={d.get('func','')} op={d.get('op','')} passed={d.get('passed')} reason={d.get('reason','')}")
    print("---------------------")
    print(f"Score: {s:.4f} (passed {passed}/{total})")


if __name__ == "__main__":
    main()
