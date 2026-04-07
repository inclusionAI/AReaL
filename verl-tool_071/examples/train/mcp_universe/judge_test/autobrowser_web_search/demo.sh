#!/usr/bin/env bash
set -euo pipefail

ROOT="examples/train/mcp_universe/judge_test"
CFG_ROOT="benchmarks/MCP-Universe/mcpuniverse/benchmark/configs"

echo "== Demo 1: browser_automation / paper -> arXiv IDs =="
python "$ROOT/run.py" \
  --task test/browser_automation/playwright_paper_task_0001.json \
  --configs_root "$CFG_ROOT" \
  --answer_file "$ROOT/answers/browser_automation_playwright_paper_task_0001.json"

echo
echo "== Demo 2: browser_automation / sports stat =="
python "$ROOT/run.py" \
  --task test/browser_automation/playwright_sports_task_0002.json \
  --configs_root "$CFG_ROOT" \
  --answer_file "$ROOT/answers/browser_automation_playwright_sports_task_0002.json"

echo
echo "== Demo 3: web_search / LLM judge =="
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "WARNING: OPENAI_API_KEY not set; web_search judge will fail." >&2
fi
python "$ROOT/run.py" \
  --task test/web_search/info_search_task_0011.json \
  --configs_root "$CFG_ROOT" \
  --answer_text "Cigarette"

echo
echo "== Demo 4: financial_analysis / yfinance portfolio =="
ANS_OUT="examples/train/mcp_universe/judge_test/answers/financial_analysis_yfinance_task_0003.json"
python examples/train/mcp_universe/judge_test/build_finance_answer.py \
  --task test/financial_analysis/yfinance_task_0003.json \
  --configs_root "$CFG_ROOT" \
  --out "$ANS_OUT"
python "$ROOT/run.py" \
  --task test/financial_analysis/yfinance_task_0003.json \
  --configs_root "$CFG_ROOT" \
  --answer_file "$ANS_OUT"
