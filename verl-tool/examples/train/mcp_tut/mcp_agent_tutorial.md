# MCP Agent Training & Evaluation Tutorial

This guide explains how to train/evaluation MCP agents end-to-end in this repo, including environment setup, local MCP gateway/tool server, reward managers, and two worked examples. It also covers how to bring up your own MCP server and extend datasets.

<div align="center">
  <img src="../../../assets/imgs/mcp_verltool.png" alt="MCP-VerlTool Overview" width="300">
</div>

## 1.How we use MCP in this repo
Model Context Protocol (MCP) standardizes tool invocation over a transport (stdio/SSE). In this repo:
- **Gateway**: [`verl_tool.servers.tools.utils.local_mcp.gateway`](../../../verl_tool/servers/tools/utils/local_mcp/gateway.py) loads servers from [`server_list.json`](../../../verl_tool/servers/tools/utils/local_mcp/configs/server_list.json) and exposes them via SSE. We include calculator, weather, google search, notion, etc.
- **Tool server**: [`verl_tool.servers.tool_server`](../../../verl_tool/servers/tool_server.py) with [`mcp_interface`](../../../verl_tool/servers/tools/mcp_interface.py) adapts LLM tool calls (`<tool_call>...</tool_call>`) to MCP gateway requests
- **Reward managers**:
  - [`mcp_static`](../../../verl_tool/workers/reward_manager/mcp_static.py): compares model `<answer>...</answer>` to static `ground_truth` with numeric and substring tolerance.
  - [`mcp_dynamic`](../../../verl_tool/workers/reward_manager/mcp_dynamic.py): executes `validation_calls` stored in the sample against MCP at eval time, then matches the model answer (with keyword fallback for summaries, e.g., "no active alerts" or alert names).
- **Datasets**:
  - [`mcp_gsm8k`](../../data_preprocess/mcp_gsm8k.py)
  - [`mcp_weather`](../../data_preprocess/mcp_weather.py)

## 2. Environment setup
```bash
# Install uv if missing
pip install uv

# Sync deps and create .venv
git submodule update --init --recursive
uv sync
source .venv/bin/activate

# Editable installs
uv pip install -e verl
uv pip install -e ".[vllm,acecoder,torl,search_tool]"

# FlashAttention (match torch version)
uv pip install "flash-attn==2.8.3" --no-build-isolation
# Or official wheel (torch 2.8.x/cu12.8 example):
# uv pip install "flash_attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
```
If your torch/OS combo differs, pick a compatible flash-attn/vLLM build, or set `ATTN_IMPL=sdpa` to avoid flash-attn dependency.

## 3. MCP setup and smoke test
Ensure [`server_list.json`](../../../verl_tool/servers/tools/utils/local_mcp/configs/server_list.json) includes the servers you need.

We initially provice following MCP servers:
- No extra keys: `calculator`, `date`, `echo`, `weather`, `wikipedia`, `yfinance`, `chrome`, `playwright`
- Require keys/config:
  - `google-search` → `SERP_API_KEY`
  - `google-maps` → `GOOGLE_MAPS_API_KEY`
  - `google-sheets` → `SERVICE_ACCOUNT_PATH`, `DRIVE_FOLDER_ID`
  - `filesystem` → `FILESYSTEM_DIRECTORY`
  - `postgres` → `POSTGRES_ADDRESS`
  - `notion` → `NOTION_API_KEY`

For configurate the API_KEYs, please import required API keys to `.env`.

Run the smoke script to start the gateway and list tools for target servers (no task calls):
```bash
source .venv/bin/activate
GATEWAY_PORT=18002 \
TARGET=calculator \
bash scripts/mcp_smoke_test.sh
```

## 4. MCP agent train/evaluation

### Example A: Calculator (static reward)

Data generation (static ground truth; extracts GSM8K final numbers after `####`):
```bash
python examples/data_preprocess/mcp_gsm8k.py --out_dir data/mcp_gsm8k --max_train 500 --max_test 100
```
Example data:
```json
{
  "prompt": [
    {"role": "system", "content": "...tool call rules..."},
    {"role": "user", "content": "Mimi picked up 2 dozen seashells... How many seashells did Leigh have?"}
  ],
  "reward_model": {"style": "rule", "ground_truth": "16"},
  "extra_info": {
    "mcp_servers": [{"name": "calculator"}],
    "target_answer": "16",
    "reference_answer": "...#### 16",
    "use_specified_server": true
  }
}
```
Scoring: [`mcp_static`](../../../verl_tool/workers/reward_manager/mcp_static.py) extracts `<answer>...</answer>`, normalizes text/number; reward 1 if it matches `ground_truth` (tolerance 1e-4 or substring), else 0.


Eval / train (scripts auto-start gateway + tool server):
```bash
source .venv/bin/activate
DATA_DIR=data/mcp_gsm8k \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
N_GPUS_PER_NODE=4 \
MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct \
bash examples/train/mcp_tut/mcp_gsm8k/eval.sh    # validation only

# Enable training (GRPO)
RUN_TRAINING=1 \
DATA_DIR=data/mcp_gsm8k \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
N_GPUS_PER_NODE=4 \
MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct \
bash examples/train/mcp_tut/mcp_gsm8k/train.sh
```

### Example B: Weather (dynamic reward)
Data generation (stores `validation_calls` so rewards are checked against live `weather/get_alerts`):
```bash
python examples/data_preprocess/mcp_weather.py --out_dir data/mcp_weather --max_train 50 --max_test 20
```
Example data:
```json
{
  "prompt": [
    {"role": "system", "content": "...tool call rules..."},
    {"role": "user", "content": "Use the weather tool to report any active alerts for California."}
  ],
  "reward_model": {"style": "dynamic", "ground_truth": ""},
  "extra_info": {
    "mcp_servers": [{"name": "weather"}],
    "state": "CA",
    "validation_calls": [{"server": "weather", "name": "get_alerts", "arguments": {"state": "CA"}}],
    "use_specified_server": true
  }
}
```
Scoring (`mcp_dynamic`): run `validation_calls` at eval time to get live tool text; extract model `<answer>`, normalize; reward 1 if it matches (numeric/substring) or hits keyword fallback (alert name / “no active alerts”), else 0.
Eval / train (scripts auto-start gateway + tool server; set RUN_TRAINING=1 to train):
```bash
source .venv/bin/activate
DATA_DIR=data/mcp_weather \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
N_GPUS_PER_NODE=4 \
MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct \
bash examples/train/mcp_tut/mcp_weather/eval.sh   # validation only

# Training
RUN_TRAINING=1 \
DATA_DIR=data/mcp_weather \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
N_GPUS_PER_NODE=4 \
MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct \
bash examples/train/mcp_tut/mcp_weather/train.sh
```

## 7. Custom MCP tasks
The examples above show minimal training/eval in verltool; real MCP workflows often involve richer environments (multiple MCPs, multi-step reasoning, tool chaining). Defining your own task can follow the pattern below:
1) Add your MCP server to [`server_list.json`](../../../verl_tool/servers/tools/utils/local_mcp/configs/server_list.json) (stdio/SSE command, env vars).  
2) In data generation, populate:
   - `extra_info.mcp_servers` to restrict server choices.
   - Static: `reward_model.ground_truth`; Dynamic: `validation_calls` (server, tool name, arguments).  
3) Pick reward manager: [`mcp_static`](../../../verl_tool/workers/reward_manager/mcp_static.py) for fixed answers, [`mcp_dynamic`](../../../verl_tool/workers/reward_manager/mcp_dynamic.py) for live tool answers (extend keyword rules if needed); for richer tasks, implement your own reward manager using these as references.  

## 8. Common Q&A
- *Ports are occupied — what to do?*  
  Scripts try to kill stale processes; if still blocked, set different `GATEWAY_PORT` / `TOOL_PORT` and rerun.
- *Smoke test shows tool errors — how to debug?*  
  Verify `MCP_GATEWAY_ADDRESS` / `TOOL_SERVER_URL`, ensure the server is defined in `server_list.json`, and rerun `TARGET=<server>` with `scripts/mcp_smoke_test.sh`.
- *Dynamic reward always 0 — why?*  
  Check that samples carry `validation_calls`, the MCP server is reachable, and the model outputs `<answer>` containing the key phrase (alert name or “no active alerts”).  
- *Model outputs long reasoning instead of final answer — does it affect score?*  
  Scoring extracts the last `<answer>...</answer>` and normalizes; ensure the final answer is inside `<answer>` tags and not buried elsewhere.

## 9. More resources
- We also provide the MCP-Universe benchmark within this framework for evaluation (`benchmarks/MCP-Universe`).
