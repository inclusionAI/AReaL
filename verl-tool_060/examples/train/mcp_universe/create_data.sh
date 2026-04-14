#!/usr/bin/env bash
set -euo pipefail

# Example gateway launches per domain to embed Tools Description while creating datasets.
# Run exactly one block at a time in a separate terminal, then set:
#   export MCP_GATEWAY_ADDRESS="http://localhost:<PORT>"
# and run the converter for that domain.

CFG="benchmarks/MCP-Universe/mcpuniverse/mcp/configs/server_list.json"
PYPATH="$(pwd)/benchmarks/MCP-Universe"
export NVM_DIR="$HOME/.nvm"

[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"
# 0) Web Search
# Requires: SERP_API_KEY (google-search); NOTION_API_KEY if 'notion' included
PYTHONPATH=$PYPATH \
python -m mcpuniverse.mcp.gateway \
   --port 8001 \
   --config "$CFG" \
   --mode stdio \
   --servers "google-search,fetch,notion" &

export MCP_GATEWAY_ADDRESS="http://localhost:8001"
python examples/data_preprocess/mcp_universe.py \
  --configs_root benchmarks/MCP-Universe/mcpuniverse/benchmark/configs \
  --out_root data/mcp_universe \
  --domains web_search 

python examples/train/mcp_universe/parquet_to_readable.py \
  data/mcp_universe/web_search/test.parquet \
  --out data/mcp_universe/web_search/readable.md \
  --max_rows 1000 --max_chars 2000000

# 1) Location Navigation
# Servers from tasks: google-maps; plus multi-server tasks use playwright / weather
# Requires: GOOGLE_MAPS_API_KEY; node/npm for playwright
nvm use 20
NODE_BIN="$(dirname "$(which node)")"
PATH="$NODE_BIN:$PATH"
PYTHONPATH=$PYPATH \
python -m mcpuniverse.mcp.gateway \
  --port 8002 \
  --config "$CFG" \
  --mode stdio \
  --servers "google-maps,playwright,weather" &

# After the gateway is up, convert dataset and dump human‑readable preview
export MCP_GATEWAY_ADDRESS="http://localhost:8002"
python examples/data_preprocess/mcp_universe.py \
    --configs_root benchmarks/MCP-Universe/mcpuniverse/benchmark/configs \
    --out_root data/mcp_universe \
    --domains location_navigation
python examples/train/mcp_universe/parquet_to_readable.py \
    data/mcp_universe/location_navigation/test.parquet \
    --out data/mcp_universe/location_navigation/readable.md \
    --max_rows 1000 --max_chars 2000000

# 2) Browser Automation
# Servers from tasks: playwright, date (no API keys; playwright needs node/npm)
nvm use 20
NODE_BIN="$(dirname "$(which node)")"
PATH="$NODE_BIN:$PATH"
PYTHONPATH=$PYPATH \
python -m mcpuniverse.mcp.gateway \
  --port 8003 \
  --config "$CFG" \
  --mode stdio \
  --servers "playwright,date,notion" &

# After the gateway is up, convert dataset and dump human‑readable preview
export MCP_GATEWAY_ADDRESS="http://localhost:8003"
python examples/data_preprocess/mcp_universe.py \
    --configs_root benchmarks/MCP-Universe/mcpuniverse/benchmark/configs \
    --out_root data/mcp_universe \
    --domains browser_automation
python examples/train/mcp_universe/parquet_to_readable.py \
    data/mcp_universe/browser_automation/test.parquet \
    --out data/mcp_universe/browser_automation/readable.md \
    --max_rows 1000 --max_chars 2000000

# 3) Financial Analysis
# Agent-level servers: yfinance, calculator (typically no API keys)
PYTHONPATH=$PYPATH \
python -m mcpuniverse.mcp.gateway \
  --port 8004 \
  --config "$CFG" \
  --mode stdio \
  --servers "yfinance,calculator" &

# After the gateway is up, convert dataset and dump human‑readable preview
export MCP_GATEWAY_ADDRESS="http://localhost:8004"
python examples/data_preprocess/mcp_universe.py \
    --configs_root benchmarks/MCP-Universe/mcpuniverse/benchmark/configs \
    --out_root data/mcp_universe \
    --domains financial_analysis
python examples/train/mcp_universe/parquet_to_readable.py \
    data/mcp_universe/financial_analysis/test.parquet \
   --out data/mcp_universe/financial_analysis/readable.md \
   --max_rows 1000 --max_chars 2000000

# 4) Repository Management
# Servers from tasks: github (requires GITHUB_PERSONAL_ACCESS_TOKEN)
nvm use 20
NODE_BIN="$(dirname "$(which node)")"
PATH="$NODE_BIN:$PATH"
PYTHONPATH=$PYPATH \
python -m mcpuniverse.mcp.gateway \
  --port 8005 \
  --config "$CFG" \
  --mode stdio \
  --servers "github,playwright" &

# After the gateway is up, convert dataset and dump human‑readable preview
export MCP_GATEWAY_ADDRESS="http://localhost:8005"
python examples/data_preprocess/mcp_universe.py \
    --configs_root benchmarks/MCP-Universe/mcpuniverse/benchmark/configs \
    --out_root data/mcp_universe \
    --domains repository_management
python examples/train/mcp_universe/parquet_to_readable.py \
    data/mcp_universe/repository_management/test.parquet \
    --out data/mcp_universe/repository_management/readable.md \
    --max_rows 1000 --max_chars 2000000

# 5) 3D Design
# Agent-level server: blender (requires local Blender environment; no API keys)
PYTHONPATH=$PYPATH \
python -m mcpuniverse.mcp.gateway \
  --port 8006 \
  --config "$CFG" \
  --mode stdio \
  --servers "blender" &

# After the gateway is up, convert dataset and dump human‑readable preview
export MCP_GATEWAY_ADDRESS="http://localhost:8006"
python examples/data_preprocess/mcp_universe.py \
    --configs_root benchmarks/MCP-Universe/mcpuniverse/benchmark/configs \
    --out_root data/mcp_universe \
    --domains 3d_design
python examples/train/mcp_universe/parquet_to_readable.py \
    data/mcp_universe/3d_design/test.parquet \
    --out data/mcp_universe/3d_design/readable.md \
   --max_rows 1000 --max_chars 2000000

echo "Launched example gateways. Pick the right port and set MCP_GATEWAY_ADDRESS before converting."
pkill -f mcpuniverse.mcp.gateway || true