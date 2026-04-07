#!/usr/bin/env bash
set -euo pipefail

# Convenience script: start gateway, discover tools, generate cases, and run interface tests.

GW_PORT=${GW_PORT:-8012}
SERVERS=${SERVERS:-"google-search,fetch,notion,playwright,weather,date,yfinance,calculator"}
CFG=${CFG:-"benchmarks/MCP-Universe/mcpuniverse/mcp/configs/server_list.json"}
OUT_JSON=${OUT_JSON:-"examples/train/mcp_universe/mcp_interface_test/discovered_tools.json"}
OUT_CASES=${OUT_CASES:-"examples/train/mcp_universe/mcp_interface_test/cases_auto"}
STATEFUL_SERVERS=${STATEFUL_SERVERS:-""}

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
echo "Gateway at ${MCP_GATEWAY_ADDRESS}"

echo "Discovering tools..."
python examples/train/mcp_universe/mcp_interface_test/discover_tools.py \
  --servers "${SERVERS}" \
  --out "${OUT_JSON}"

echo "Generating cases from discovery..."
python examples/train/mcp_universe/mcp_interface_test/generate_cases_from_discovery.py \
  --infile "${OUT_JSON}" \
  --out_root "${OUT_CASES}"

echo "Running interface tests on generated cases..."
if [ -n "${STATEFUL_SERVERS}" ]; then
  python examples/train/mcp_universe/mcp_interface_test/interface_runner.py \
    --root "${OUT_CASES}" \
    --stateful_servers "${STATEFUL_SERVERS}"
else
  python examples/train/mcp_universe/mcp_interface_test/interface_runner.py \
    --root "${OUT_CASES}"
fi

echo "Done."
