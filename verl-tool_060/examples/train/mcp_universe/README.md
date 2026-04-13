MCP‑Universe Style Eval Scripts

This folder contains six eval-only scripts, one per task domain:

- eval_web_search.sh
- eval_location_navigation.sh
- eval_browser_automation.sh
- eval_financial_analysis.sh
- eval_repository_management.sh
- eval_design_3d.sh

Taks Web Search as example (end‑to‑end)

1) Install deps and export API keys

```bash
# Download node.js
export NVM="YOURPATH TO NVM/.nvm"
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
. "$NVM/nvm.sh"
nvm install 20
sudo apt update
sudo apt install -y docker.io
sudo systemctl enable --now docker
```

```bash
# Download playwright browswer
npm init -y               
npm install @playwright/test
npx playwright install chrome
```

```bash
# Recommended: install optional extras for MCP tooling
git submodule update --init --recursive
uv sync
source .venv/bin/activate
uv pip install -e verl
uv pip install -e ".[vllm,acecoder,torl,search_tool,mcp_tool]"
uv pip install -e benchmarks/MCP-Universe
uv pip install "flash-attn<2.8.0" --no-build-isolation
```
Fill out the API KEY setting in the env.example.sh

```bash
cp examples/train/mcp_universe/env.example.sh examples/train/mcp_universe/env.local.sh
set -a; source examples/train/mcp_universe/env.local.sh; set +a
```

2) Start MCP‑Universe gateway for web_search and export address

```bash
# Minimal set for web_search. If your dataset includes multi‑server tasks.
PYTHONPATH=benchmarks/MCP-Universe \
python -m mcpuniverse.mcp.gateway \
  --port 8001 \
  --config benchmarks/MCP-Universe/mcpuniverse/mcp/configs/server_list.json \
  --mode stdio \
  --servers "google-search,fetch,notion" &
```

3) Prepare the web_search dataset (Tools Description embedded)

```bash
export MCP_GATEWAY_ADDRESS="http://localhost:8001"
python examples/data_preprocess/mcp_universe.py \
  --configs_root benchmarks/MCP-Universe/mcpuniverse/benchmark/configs \
  --out_root data/mcp_universe \
  --domains web_search 

python examples/train/mcp_universe/parquet_to_readable.py \
  data/mcp_universe/web_search/test.parquet \
  --out data/mcp_universe/web_search/readable.md \
  --max_rows 1000 --max_chars 10000

# Output: data/mcp_universe/web_search/test.parquet
```

4) Stop the gateway

```bash
pkill -f mcpuniverse.mcp.gateway || true
```

Run one of the eval scripts. It will:
- infer required servers from the dataset (or use `SERVERS` if set),
- auto-launch the MCP‑Universe gateway and the tool server,
- run eval in val-only mode.

```bash
CUDA_VISIBLE_DEVICES=7 bash examples/train/mcp_universe/eval_web_search.sh
```

Optional environment overrides:
- `MODEL_NAME` (default `Qwen/Qwen2.5-3B-Instruct`)
- `SERVERS` (comma‑separated; otherwise inferred from dataset)
- `GW_PORT` (default 8010), `HOST`/`PORT` for the tool server
- `MAX_TURNS`, `TEMP`, `TOP_P`, `N_GPUS`, `N_NODES`

Notes:
- These scripts are eval-only (`trainer.val_only=True`) and use the `mcp_universe` reward manager.
- Each script auto‑launches the gateway. You can override the set with `SERVERS` (comma‑separated). Otherwise it infers from your dataset.
- Export the required API keys for each server (e.g., `SERP_API_KEY`, `GITHUB_PERSONAL_ACCESS_TOKEN`, `GOOGLE_MAPS_API_KEY`) before running.
- For a single remote MCP endpoint (no gateway), set `MCP_SERVER_URL` and leave `SERVERS` empty.
