## Judge Test for MCP-Universe Evaluators

This mini harness lets you feed prepared `<answer>` outputs into MCP-Universe evaluators and get a numeric score back — without running the full agent. It supports:

### What it does

* Wraps your answer in `<answer>…</answer>`
* Extracts the inner payload (if JSON, passed directly as string)
* Calls `MCP-Universe Task.evaluate(payload)`
* Aggregates `score = #passed / #total evaluators`
* Prints per-evaluator pass/fail and the final score

### Layout

* **run.py**: main script
* **answers/**: demo answers for a few tasks
* **build_answers.py**: auto-extract answers from tasks with literal ground-truth
* **answers_auto/**: auto-built answers (generated)
* **run_batch.py**: run judge on all auto-built answers and report pass rate

### Prerequisites

* Python deps: install MCP-Universe locally:

  ```bash
  uv pip install -e benchmarks/MCP-Universe
  ```
* For web_search: export `OPENAI_API_KEY`
* For tasks that depend on external services (e.g., Google Maps): provide the corresponding API key (e.g., `GOOGLE_MAPS_API_KEY`). Examples below prioritize tasks that require no external keys.

---

## Quick Start

1. **Browser automation — paper titles to arXiv IDs (pure JSON equality)**

```bash
python examples/train/mcp_universe/judge_test/run.py \
  --task test/browser_automation/playwright_paper_task_0001.json \
  --configs_root benchmarks/MCP-Universe/mcpuniverse/benchmark/configs \
  --answer_file examples/train/mcp_universe/judge_test/answers/browser_automation_playwright_paper_task_0001.json
```

Expected: All evaluators pass; score = 1.0

2. **Browser automation — sports stat (pure JSON equality)**

```bash
python examples/train/mcp_universe/judge_test/run.py \
  --task test/browser_automation/playwright_sports_task_0002.json \
  --configs_root benchmarks/MCP-Universe/mcpuniverse/benchmark/configs \
  --answer_file examples/train/mcp_universe/judge_test/answers/browser_automation_playwright_sports_task_0002.json
```

Expected: Pass; score = 1.0

3. **Web search — LLM judge (raw)**

```bash
export OPENAI_API_KEY=sk-...   # required
python examples/train/mcp_universe/judge_test/run.py \
  --task test/web_search/info_search_task_0011.json \
  --configs_root benchmarks/MCP-Universe/mcpuniverse/benchmark/configs \
  --answer_text "Cigarette"
```

Expected: Uses gpt-4.1 judge; should pass (score = 1.0)

4. **Auto-build answers and batch judge (recommended quick pipeline check)**

```bash
# Generate directly passable answers (currently supports browser_automation is_dict_equal tasks, web_search LLM judge tasks)
python examples/train/mcp_universe/judge_test/build_answers.py \
  --configs_root benchmarks/MCP-Universe/mcpuniverse/benchmark/configs/test \
  --out_root examples/train/mcp_universe/judge_test/answers_auto

# Batch evaluation and report pass rate
python examples/train/mcp_universe/judge_test/run_batch.py \
  --configs_root benchmarks/MCP-Universe/mcpuniverse/benchmark/configs \
  --index examples/train/mcp_universe/judge_test/answers_auto/index.json
```

Notes:

* Auto-build currently only extracts:

  * **browser_automation** tasks where evaluator = `playwright.is_dict_equal` with literal values (directly use value as JSON answer)
  * **web_search** tasks using `google_search.llm_as_a_judge` with a `correct_answer` string
* Other domains (location_navigation / financial_analysis) use dynamic checks (Google Maps / yfinance) and cannot auto-derive a single unique JSON answer, so are skipped.

5. **Financial analysis — auto-build portfolio return answer (example)**

```bash
# Generate ground-truth answer for yfinance_task_0003 (real-time data, requires network)
python examples/train/mcp_universe/judge_test/build_finance_answer.py \
  --task test/financial_analysis/yfinance_task_0003.json \
  --configs_root benchmarks/MCP-Universe/mcpuniverse/benchmark/configs \
  --out examples/train/mcp_universe/judge_test/answers/financial_analysis_yfinance_task_0003.json

# Evaluate
python examples/train/mcp_universe/judge_test/run.py \
  --task test/financial_analysis/yfinance_task_0003.json \
  --configs_root benchmarks/MCP-Universe/mcpuniverse/benchmark/configs \
  --answer_file examples/train/mcp_universe/judge_test/answers/financial_analysis_yfinance_task_0003.json
```

Explanation: This script reuses the evaluator’s own yfinance function to construct a JSON answer consistent with the required `output_format`; numbers are checked with tolerance against real-time data, so no manual calculation needed.

6. **Location navigation — caveats**

* This domain depends on Google Maps data and generally requires a `GOOGLE_MAPS_API_KEY`. The evaluator launches `@modelcontextprotocol/server-google-maps` via stdio; no gateway required.
* Since tasks often involve strict geographic/time validations (e.g., `compare_time_of_middle_point`), it’s not possible to auto-derive unique answers. Use `run.py` with your own JSON answer and a valid API key.

---

## Notes & Tips

* For JSON tasks: always output **a valid JSON object**; key names, case, and nesting must exactly match the task’s `output_format`. Value types must be parseable.
* **location_navigation** tasks usually include Google Maps validation (place id, rating, city match, etc.); require `GOOGLE_MAPS_API_KEY`, may start a gateway; not demoed here.
* **financial_analysis** relies on yfinance real-time data; values must match current data within tolerance. Best to compute yourself, then test with JSON.
* **web_search** judge model (default gpt-4.1) is set in evaluator; can be changed to read from env.

---

## Troubleshooting

* `ImportError: No module named jinja2`: run `uv pip install -e benchmarks/MCP-Universe`
* web_search 401 error: check `OPENAI_API_KEY`
* Playwright / Maps issues: require node/npx and corresponding API keys (e.g., `GOOGLE_MAPS_API_KEY`). Evaluators start MCP servers via stdio; gateway only needed for remote SSE connections. Demo examples prioritize JSON equality tasks that need no external keys.