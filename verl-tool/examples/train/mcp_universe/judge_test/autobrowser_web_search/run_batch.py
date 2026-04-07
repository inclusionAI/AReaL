#!/usr/bin/env python3
"""
Batch-run judge on auto-built answers (see build_answers.py).

It reads answers_auto/index.json, evaluates each task, and reports pass rate.
"""
from __future__ import annotations

import os
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List

from verl_tool.workers.reward_manager.mcp_universe_eval import eval_answer_for_task


DEFAULT_CONFIGS_ROOT = "benchmarks/MCP-Universe/mcpuniverse/benchmark/configs"
DEFAULT_INDEX = "examples/train/mcp_universe/judge_test/answers_auto/index.json"


def _read(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read().strip()


def _eval_one(configs_root: Path, task_rel: str, answer_path: Path) -> float:
    # Build payload (<answer> content only)
    txt = _read(answer_path)
    response = txt
    summary = eval_answer_for_task(response, task_rel, configs_root=str(configs_root))
    return float(summary.get("score", 0.0))


def main():
    ap = argparse.ArgumentParser(description="Batch judge for auto-built answers")
    ap.add_argument("--configs_root", default=DEFAULT_CONFIGS_ROOT, help="Configs root folder")
    ap.add_argument("--index", default=DEFAULT_INDEX, help="answers_auto/index.json")
    args = ap.parse_args()

    idx_path = Path(args.index)
    if not idx_path.exists():
        raise FileNotFoundError(f"Index file not found: {idx_path}")

    items = json.loads(_read(idx_path))
    total, passed = 0, 0
    for it in items:
        task_rel = it["task"]
        raw_path = Path(it["answer_path"])  # may be repo-relative or index-relative
        if raw_path.is_absolute() or raw_path.exists():
            a_path = raw_path
        else:
            a_path = (idx_path.parent / raw_path)
        score = _eval_one(Path(args.configs_root), task_rel, a_path)
        ok = (abs(score - 1.0) < 1e-6)
        print(f"[{ 'OK' if ok else 'FAIL'}] {task_rel} -> {score:.4f}")
        total += 1
        passed += int(ok)

    print(f"Summary: {passed}/{total} tasks full-pass")


if __name__ == "__main__":
    main()
