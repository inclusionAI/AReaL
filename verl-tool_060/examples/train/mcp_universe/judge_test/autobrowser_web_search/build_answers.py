#!/usr/bin/env python3
"""
Auto-build answer inputs for a subset of tasks by extracting ground-truth
directly from task evaluators.

Currently supports:
- browser_automation: playwright.is_dict_equal with a literal value
- web_search: google_search.llm_as_a_judge with op_args.correct_answer

Outputs answer files under answers_auto/<domain>/, and an index.json mapping
each answer file to its task path for batch testing.
"""
from __future__ import annotations

import os
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_CONFIGS_ROOT = "benchmarks/MCP-Universe/mcpuniverse/benchmark/configs/test"
DEFAULT_OUT_ROOT = "examples/train/mcp_universe/judge_test/answers_auto"

DOMAINS = [
    "browser_automation",
    "web_search",
    # "location_navigation",  # dynamic; needs API keys
    # "financial_analysis",    # dynamic; guarded by --include_finance
]


def _iter_tasks(configs_root: str, domain: str) -> List[Path]:
    folder = Path(configs_root) / domain
    if not folder.exists():
        return []
    return sorted(p for p in folder.glob("*.json") if p.is_file())


def _load_json(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _save_text(p: Path, s: str):
    with p.open("w", encoding="utf-8") as f:
        f.write(s)


def _save_json(p: Path, obj: Any):
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _build_browser_automation(task_obj: Dict[str, Any]) -> Dict[str, Any] | None:
    """Extract literal expected dict from playwright.is_dict_equal."""
    for ev in (task_obj.get("evaluators") or []):
        if ev.get("func") == "json" and ev.get("op") == "playwright.is_dict_equal":
            val = ev.get("value")
            if isinstance(val, (dict, list)):
                return val
    return None


def _build_web_search(task_obj: Dict[str, Any]) -> str | None:
    """Extract correct answer for LLM judge."""
    for ev in (task_obj.get("evaluators") or []):
        if ev.get("func") == "raw" and ev.get("op") == "google_search.llm_as_a_judge":
            op_args = ev.get("op_args") or {}
            val = op_args.get("correct_answer")
            if isinstance(val, str) and val.strip():
                return val.strip()
    return None


def _build_financial_analysis(task_obj: Dict[str, Any]) -> Dict[str, Any] | None:
    """Build answers for a subset of yfinance tasks that are computable locally.

    Currently supports evaluator op: yfinance.check_portfolio_task_output
    by calling yfinance evaluator helper yfinance__calculate_portfolio_return.
    """
    evs = task_obj.get("evaluators") or []
    target = None
    for ev in evs:
        if ev.get("func") == "json" and ev.get("op") == "yfinance.check_portfolio_task_output":
            target = ev
            break
    if target is None:
        return None
    try:
        from mcpuniverse.evaluator.yfinance.functions import (
            yfinance__calculate_portfolio_return,
        )
    except Exception:
        return None
    op_args = target.get("op_args") or {}
    tickers = op_args.get("tickers")
    start_date = op_args.get("start_date")
    end_date = op_args.get("end_date")
    initial_investment = op_args.get("initial_investment")
    split = op_args.get("split")
    if not (tickers and start_date and end_date and initial_investment is not None and split):
        return None
    try:
        final_value, percentage_return = yfinance__calculate_portfolio_return(
            tickers, start_date, end_date, float(initial_investment), split
        )
        return {
            "total value": float(final_value),
            "total percentage return": float(percentage_return),
        }
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="Auto-build demo answers for test judge harness")
    ap.add_argument("--configs_root", default=DEFAULT_CONFIGS_ROOT, help="Path to configs/test root")
    ap.add_argument("--out_root", default=DEFAULT_OUT_ROOT, help="Where to write answers")
    ap.add_argument("--include_finance", action="store_true", help="Also build financial_analysis answers (requires yfinance & network)")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    _ensure_dir(out_root)
    index: List[Dict[str, str]] = []

    domains = list(DOMAINS)
    if args.include_finance:
        domains.append("financial_analysis")

    for domain in domains:
        tasks = _iter_tasks(args.configs_root, domain)
        if not tasks:
            continue
        out_dir = out_root / domain
        _ensure_dir(out_dir)
        built = 0
        for tpath in tasks:
            try:
                obj = _load_json(tpath)
            except Exception:
                continue

            if domain == "browser_automation":
                ans_obj = _build_browser_automation(obj)
                if ans_obj is None:
                    continue
                out_file = out_dir / f"{tpath.stem}.json"
                _save_json(out_file, ans_obj)
                index.append({
                    "task": f"{domain}/{tpath.name}",
                    "answer_path": str(out_file),
                    "type": "json"
                })
                built += 1

            elif domain == "web_search":
                ans_txt = _build_web_search(obj)
                if ans_txt is None:
                    continue
                out_file = out_dir / f"{tpath.stem}.txt"
                _save_text(out_file, ans_txt)
                index.append({
                    "task": f"{domain}/{tpath.name}",
                    "answer_path": str(out_file),
                    "type": "text"
                })
                built += 1

            elif domain == "financial_analysis":
                ans_obj = _build_financial_analysis(obj)
                if ans_obj is None:
                    continue
                out_file = out_dir / f"{tpath.stem}.json"
                _save_json(out_file, ans_obj)
                index.append({
                    "task": f"{domain}/{tpath.name}",
                    "answer_path": str(out_file),
                    "type": "json"
                })
                built += 1

        print(f"Built {built} answers for domain: {domain}")

    # Write index
    with (out_root / "index.json").open("w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print(f"Wrote index with {len(index)} items: {out_root / 'index.json'}")


if __name__ == "__main__":
    main()
