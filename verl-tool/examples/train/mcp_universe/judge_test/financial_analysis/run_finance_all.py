#!/usr/bin/env python3
from __future__ import annotations

"""
Auto-build JSON answers for all financial_analysis tasks under configs/test
and immediately run judge for each. Saves JSON answers for inspection.

Usage (from repo root):
  python examples/train/mcp_universe/judge_test/run_finance_all.py \
    --configs_root benchmarks/MCP-Universe/mcpuniverse/benchmark/configs \
    --out_dir examples/train/mcp_universe/judge_test/answers_financial
"""

import os
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Set

from verl_tool.workers.reward_manager.mcp_universe_eval import eval_answer_for_task


DEFAULT_CONFIGS_ROOT = "benchmarks/MCP-Universe/mcpuniverse/benchmark/configs"
DEFAULT_OUT_DIR = "examples/train/mcp_universe/judge_test/financial_analysis/answers_financial"


def _iter_financial_tasks(configs_root: str) -> List[Path]:
    test_dir = Path(configs_root) / "test" / "financial_analysis"
    return sorted(p for p in test_dir.glob("*.json") if p.is_file())


def _load_task(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---- Builders for supported yfinance tasks ----

def _build_portfolio_answer(task_obj: Dict[str, Any]) -> Dict[str, Any] | None:
    try:
        from mcpuniverse.evaluator.yfinance.functions import (
            yfinance__calculate_portfolio_return,
        )
    except Exception:
        return None
    for ev in (task_obj.get("evaluators") or []):
        if ev.get("func") == "json" and ev.get("op") == "yfinance.check_portfolio_task_output":
            op_args = ev.get("op_args") or {}
            tickers = op_args.get("tickers")
            start_date = op_args.get("start_date")
            end_date = op_args.get("end_date")
            initial_investment = op_args.get("initial_investment")
            split = op_args.get("split")
            if not (tickers and start_date and end_date and initial_investment is not None and split):
                continue
            fv, pr = yfinance__calculate_portfolio_return(
                tickers, start_date, end_date, float(initial_investment), split
            )
            return {"total value": float(fv), "total percentage return": float(pr)}
    return None


def _build_gross_profit_answer(task_obj: Dict[str, Any]) -> Dict[str, Any] | None:
    try:
        from mcpuniverse.evaluator.yfinance.functions import (
            yfinance__get_lastest_year_raw_gross_profit_margin,
        )
    except Exception:
        return None
    evs = task_obj.get("evaluators") or []
    tickers: Set[str] = set()
    cmp_tickers: List[str] = []
    for ev in evs:
        if ev.get("func") == "json" and ev.get("op") == "yfinance.check_gross_profit_margin":
            t = (ev.get("op_args") or {}).get("ticker")
            if isinstance(t, str):
                tickers.add(t)
        if ev.get("func") == "json" and ev.get("op") == "yfinance.compare_companies_gross_profit_margin":
            ts = (ev.get("op_args") or {}).get("tickers") or []
            if isinstance(ts, list):
                cmp_tickers = [str(x) for x in ts]
                tickers.update(cmp_tickers)
    if not tickers:
        return None
    ans: Dict[str, Any] = {}
    for t in tickers:
        m = yfinance__get_lastest_year_raw_gross_profit_margin(t)
        if m is None:
            return None
        ans.setdefault(t, {})["gross profit margin"] = float(m * 100.0)
    if cmp_tickers:
        best_t, best_v = None, -1e9
        for t in cmp_tickers:
            v = ans.get(t, {}).get("gross profit margin")
            if isinstance(v, (int, float)) and v > best_v:
                best_v = float(v)
                best_t = t
        if best_t is not None:
            ans["company with higher gross profit margin"] = best_t
    return ans


def _build_net_profit_answer(task_obj: Dict[str, Any]) -> Dict[str, Any] | None:
    try:
        from mcpuniverse.evaluator.yfinance.functions import (
            yfinance__get_lastest_year_raw_net_profit_margin,
        )
    except Exception:
        return None
    evs = task_obj.get("evaluators") or []
    tickers: Set[str] = set()
    cmp_tickers: List[str] = []
    for ev in evs:
        if ev.get("func") == "json" and ev.get("op") == "yfinance.check_net_profit_margin":
            t = (ev.get("op_args") or {}).get("ticker")
            if isinstance(t, str):
                tickers.add(t)
        if ev.get("func") == "json" and ev.get("op") == "yfinance.compare_companies_net_profit_margin":
            ts = (ev.get("op_args") or {}).get("tickers") or []
            if isinstance(ts, list):
                cmp_tickers = [str(x) for x in ts]
                tickers.update(cmp_tickers)
    if not tickers:
        return None
    ans: Dict[str, Any] = {}
    for t in tickers:
        m = yfinance__get_lastest_year_raw_net_profit_margin(t)
        if m is None:
            return None
        ans.setdefault(t, {})["net profit margin"] = float(m * 100.0)
    if cmp_tickers:
        best_t, best_v = None, -1e9
        for t in cmp_tickers:
            v = ans.get(t, {}).get("net profit margin")
            if isinstance(v, (int, float)) and v > best_v:
                best_v = float(v)
                best_t = t
        if best_t is not None:
            ans["company with highest net profit margin"] = best_t
    return ans


def _build_income_and_rd_answers(task_obj: Dict[str, Any]) -> Dict[str, Any] | None:
    """Build answers for RD expense/percentage, total revenue, operating income,
    net income (periodic), and net income difference if requested by evaluators.
    """
    try:
        from mcpuniverse.evaluator.yfinance.functions import (
            yfinance__get_latest_financial_data,
            yfinance__get_lastest_year_raw_rd_expense,
        )
    except Exception:
        return None

    evs = task_obj.get("evaluators") or []
    ans: Dict[str, Any] = {}
    # Cache for financial_data per (ticker, period)
    cache_fd: Dict[tuple, Dict[str, Any]] = {}

    def get_fd(ticker: str, period: str = 'annually') -> Dict[str, Any] | None:
        key = (ticker, period)
        if key not in cache_fd:
            try:
                cache_fd[key] = yfinance__get_latest_financial_data(ticker_symbol=ticker, period=period)
            except Exception:
                cache_fd[key] = None
        return cache_fd[key]

    # First pass: fill per-evaluator
    for ev in evs:
        op = ev.get("op")
        if not isinstance(op, str) or not op.startswith("yfinance."):
            continue
        op_args = ev.get("op_args") or {}

        if op == "yfinance.check_rd_expense":
            t = op_args.get("ticker")
            if isinstance(t, str):
                fd = get_fd(t)
                if fd is not None:
                    ans.setdefault(t, {})["R&D expense"] = float(fd.get('rd expense', 0.0))

        elif op == "yfinance.check_rd_expense_percentage":
            t = op_args.get("ticker")
            if isinstance(t, str):
                try:
                    pct = yfinance__get_lastest_year_raw_rd_expense(t)
                    if pct is not None:
                        ans.setdefault(t, {})["R&D expense percentage"] = float(pct * 100.0)
                except Exception:
                    pass

        elif op == "yfinance.compare_companies_rd_expense_percentage":
            ts = op_args.get("tickers") or []
            vals: Dict[str, float] = {}
            for t in ts:
                try:
                    pct = yfinance__get_lastest_year_raw_rd_expense(str(t))
                    if pct is not None:
                        vals[str(t)] = float(pct * 100.0)
                        ans.setdefault(str(t), {})["R&D expense percentage"] = float(pct * 100.0)
                except Exception:
                    continue
            if vals:
                best_t = max(vals, key=lambda k: vals[k])
                ans["company with higher R&D expense percentage"] = best_t

        elif op == "yfinance.check_total_revenue":
            t = op_args.get("ticker")
            if isinstance(t, str):
                fd = get_fd(t)
                if fd is not None:
                    ans.setdefault(t, {})["total revenue"] = float(fd.get('total revenue', 0.0))

        elif op == "yfinance.check_operating_income":
            t = op_args.get("ticker")
            if isinstance(t, str):
                fd = get_fd(t)
                if fd is not None:
                    ans.setdefault(t, {})["operating income"] = float(fd.get('operating income', 0.0))

        elif op == "yfinance.check_net_income":
            t = op_args.get("ticker")
            period = op_args.get("period", 'annually')
            if isinstance(t, str) and period in ['annually', 'quarterly']:
                fd = get_fd(t, period)
                if fd is not None:
                    key = f"net income common stockholders {period}"
                    ans.setdefault(t, {})[key] = float(fd.get('net income', 0.0))

        elif op == "yfinance.check_net_income_difference":
            t = op_args.get("ticker")
            periods = op_args.get("periods") or []
            if isinstance(t, str) and isinstance(periods, list) and len(periods) == 2:
                vals = []
                for p in periods:
                    if p not in ['annually', 'quarterly']:
                        break
                    fd = get_fd(t, p)
                    if fd is None:
                        break
                    vals.append(float(fd.get('net income', 0.0)))
                if len(vals) == 2:
                    diff = abs(vals[0] - vals[1])
                    ans.setdefault(t, {})['difference'] = diff

    return ans or None


def _build_percentage_change_and_increase(task_obj: Dict[str, Any]) -> Dict[str, Any] | None:
    """Build answers for percentage change / largest positive increase tasks.
    These rely on institutional holders and BlackRock percent change helpers.
    """
    try:
        from mcpuniverse.evaluator.yfinance.functions import (
            yfinance__get_filtered_institutional_holders,
            yfinance__get_blackrock_pct_change,
        )
    except Exception:
        return None
    evs = task_obj.get("evaluators") or []
    ans: Dict[str, Any] = {}

    for ev in evs:
        op = ev.get("op")
        if not isinstance(op, str) or not op.startswith("yfinance."):
            continue
        op_args = ev.get("op_args") or {}

        if op == "yfinance.check_percentage_change":
            t = op_args.get('ticker')
            min_pct = op_args.get('minPctChange')
            try:
                data = yfinance__get_filtered_institutional_holders(t, min_pct)
                if isinstance(data, dict):
                    # evaluator expects top-level keys
                    ans['institutional holders'] = data.get('institutional holders', [])
                    ans['aggregate market value'] = data.get('aggregate market value', 0.0)
            except Exception:
                pass

        if op == "yfinance.check_largest_positive_increase":
            ts = op_args.get('tickers') or []
            vals: Dict[str, float] = {}
            for t in ts:
                try:
                    change = yfinance__get_blackrock_pct_change(str(t))
                    if change is not None:
                        vals[str(t)] = float(change)
                        ans.setdefault(str(t), {})['pctChange'] = float(change)
                except Exception:
                    continue
            if vals:
                best_t = max(vals, key=lambda k: vals[k])
                ans['company with largest positive increase'] = best_t

    return ans or None


def _build_answer(task_obj: Dict[str, Any]) -> Dict[str, Any] | None:
    # Merge outputs from multiple builders to satisfy multi-evaluator tasks
    merged: Dict[str, Any] = {}
    for builder in (
        _build_portfolio_answer,
        _build_gross_profit_answer,
        _build_net_profit_answer,
        _build_income_and_rd_answers,
        _build_percentage_change_and_increase,
    ):
        part = builder(task_obj)
        if part:
            # Deep merge dicts (shallow per nested ticker keys)
            for k, v in part.items():
                if isinstance(v, dict) and isinstance(merged.get(k), dict):
                    merged[k].update(v)
                else:
                    merged[k] = v
    return merged or None


def main():
    ap = argparse.ArgumentParser(description="Build & judge all financial_analysis tasks")
    ap.add_argument("--configs_root", default=DEFAULT_CONFIGS_ROOT, help="Configs root folder")
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="Where to save generated JSON answers")
    args = ap.parse_args()

    tasks = _iter_financial_tasks(args.configs_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    built = 0
    passed = 0
    for tpath in tasks:
        total += 1
        task_rel = f"financial_analysis/{tpath.name}"
        task_obj = _load_task(tpath)
        ans_obj = _build_answer(task_obj)
        if ans_obj is None:
            print(f"[SKIP] {task_rel} (unsupported evaluator)")
            continue

        built += 1
        out_path = out_dir / f"{tpath.stem}.json"
        out_path.write_text(json.dumps(ans_obj, ensure_ascii=False, indent=2), encoding="utf-8")

        response = f"<answer>```json\n{json.dumps(ans_obj, ensure_ascii=False, indent=2)}\n```</answer>"
        summary = eval_answer_for_task(response, task_rel, configs_root=args.configs_root)
        ok = abs(float(summary.get("score", 0.0)) - 1.0) < 1e-6
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {task_rel} -> {summary.get('score', 0.0):.4f}")
        if ok:
            passed += 1

    print(f"Summary: total={total}, built={built}, full_pass={passed}")
    print(f"Answers saved to: {out_dir}")


if __name__ == "__main__":
    main()
