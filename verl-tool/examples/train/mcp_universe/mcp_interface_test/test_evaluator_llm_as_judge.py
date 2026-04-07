#!/usr/bin/env python3
import os
import sys
import argparse
import asyncio
from pathlib import Path


def ensure_project_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


async def _run_eval(task_path: str, payload: str):
    from mcpuniverse.benchmark.task import Task
    from mcpuniverse.common.context import Context
    task = Task(task_path, context=Context())
    return await task.evaluate(payload)


def main():
    ensure_project_on_path()

    ap = argparse.ArgumentParser(description="Minimal tester for raw/google_search.llm_as_a_judge evaluator")
    ap.add_argument("task", help="Path to task JSON (absolute or relative to MCP-Universe configs root)")
    ap.add_argument("answer", help="Answer text to evaluate (use the final title string)")
    ap.add_argument("--root", default="verl-tool/benchmarks/MCP-Universe/mcpuniverse/benchmark/configs", help="Configs root if task is relative")
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("[WARN] OPENAI_API_KEY is not set; evaluator may fail or return 0/0.")

    # Resolve task path
    task_path = args.task
    if not os.path.isabs(task_path):
        cand1 = os.path.join(args.root, task_path)
        cand2 = os.path.join(args.root, "test", task_path)
        task_path = cand1 if os.path.exists(cand1) else (cand2 if os.path.exists(cand2) else cand1)

    try:
        results = asyncio.run(_run_eval(task_path, args.answer))
    except Exception as e:
        print(f"[ERROR] evaluator raised: {e}")
        sys.exit(1)

    total = len(results)
    passed = sum(1 for r in results if getattr(r, "passed", False))
    print(f"passed/total: {passed}/{total}")
    for i, r in enumerate(results, 1):
        func = getattr(getattr(r, "config", None), "func", "")
        op = getattr(getattr(r, "config", None), "op", "")
        reason = getattr(r, "reason", "")
        print(f"[{i}] func={func} op={op} passed={getattr(r, 'passed', False)} reason={reason}")


if __name__ == "__main__":
    main()


