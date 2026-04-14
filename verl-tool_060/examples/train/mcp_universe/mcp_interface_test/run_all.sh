#!/usr/bin/env bash
set -euo pipefail

# This script starts the MCP‑Universe gateway for a subset of servers,
# runs interface tests (list + simple calls), and shuts down the gateway.

GW_PORT=${GW_PORT:-8010}
SERVERS=${SERVERS:-"google-search,fetch,notion,playwright,weather,date,yfinance,calculator"}
CFG=${CFG:-"benchmarks/MCP-Universe/mcpuniverse/mcp/configs/server_list.json"}

echo "Starting gateway on port ${GW_PORT} for servers: ${SERVERS}"
PYTHONPATH=benchmarks/MCP-Universe \
python -m mcpuniverse.mcp.gateway \
  --port ${GW_PORT} \
  --config ${CFG} \
  --mode stdio \
  --servers "${SERVERS}" \
  >/dev/null 2>&1 &
GW_PID=$!
trap 'pkill -P $GW_PID >/dev/null 2>&1 || true; kill -9 $GW_PID >/dev/null 2>&1 || true' EXIT

export MCP_GATEWAY_ADDRESS="http://localhost:${GW_PORT}"

echo "Running interface tests via gateway at ${MCP_GATEWAY_ADDRESS}"
python examples/train/mcp_universe/mcp_interface_test/interface_runner.py \
  --root examples/train/mcp_universe/mcp_interface_test/cases

echo "Done."

