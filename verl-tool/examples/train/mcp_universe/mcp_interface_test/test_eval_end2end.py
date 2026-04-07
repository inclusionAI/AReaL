#!/usr/bin/env python3
import os
import sys
import json
import argparse
from pathlib import Path


def ensure_project_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    s = str(repo_root)
    if s not in sys.path:
        sys.path.insert(0, s)


def main() -> None:
    ensure_project_on_path()

    ap = argparse.ArgumentParser(description="Minimal end-to-end eval using reward_manager.mcp_universe_eval (simulate agent <answer>...</answer> output)")
    ap.add_argument("task", help="Task path (absolute or relative to configs root)")
    ap.add_argument(
        "agent_response",
        help=r"Full agent response string, e.g. '<answer>{\"answer\":\"...\"}</answer>' or '<answer>...</answer>'",
    )
    ap.add_argument("--root", default="verl-tool/benchmarks/MCP-Universe/mcpuniverse/benchmark/configs", help="Configs root when task is relative")
    args = ap.parse_args()

    # Only use info from reward_manager/mcp_universe_eval.py
    from verl_tool.workers.reward_manager.mcp_universe_eval import eval_answer_for_task

    # Pass the full agent response; reward_manager will decode <answer> blocks itself
    result = eval_answer_for_task(args.agent_response, args.task, configs_root=args.root)

    out = {
        "task": args.task,
        "agent_response": args.agent_response,
        "score": result.get("score", 0.0),
        "passed": result.get("passed", 0),
        "total": result.get("total", 0),
        "details": result.get("details", []),
    }

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


