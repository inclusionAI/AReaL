#!/usr/bin/env bash
set -euo pipefail

# Minimal MCP smoke test: start local MCP gateway, then list tools for requested
# servers to verify connectivity. No real task calls needed.
#
# Usage:
#   TARGET=calculator bash scripts/mcp_smoke_test.sh
#   TARGET=weather bash scripts/mcp_smoke_test.sh
#   TARGET="calculator,weather" bash scripts/mcp_smoke_test.sh
#   TARGET=all bash scripts/mcp_smoke_test.sh   # all servers defined in server_list.json

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GATEWAY_PORT="${GATEWAY_PORT:-18002}"
CONFIG_PATH_DEFAULT="$REPO_ROOT/verl_tool/servers/tools/utils/local_mcp/configs/server_list.json"
CONFIG_PATH="${CONFIG_PATH:-$CONFIG_PATH_DEFAULT}"
TARGET="${TARGET:-calculator}"   # comma-separated or "all"
export CONFIG_PATH

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/verl:${PYTHONPATH:-}"

cleanup() {
  if [[ -n "${GATEWAY_PID:-}" ]]; then kill "$GATEWAY_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT

# Resolve targets against server_list.json
TARGET_SERVERS=$(python - <<'PY'
import json, os, sys
from pathlib import Path
config_path = Path(os.environ["CONFIG_PATH"])
cfg = json.loads(config_path.read_text())
available = list(cfg.keys())
target_env = os.environ.get("TARGET", "calculator").strip()
if target_env.lower() == "all":
    targets = available
else:
    targets = [t.strip() for t in target_env.split(",") if t.strip()]
missing = [t for t in targets if t not in available]
if missing:
    raise SystemExit(f"Unknown TARGET not in server_list.json: {missing}")
print(",".join(targets))
PY
)
export TARGET_SERVERS
echo "[smoke] targets=${TARGET_SERVERS}"

echo "[mcp] starting gateway on :$GATEWAY_PORT"
python -m verl_tool.servers.tools.utils.local_mcp.gateway \
  --port "$GATEWAY_PORT" \
  --config "$CONFIG_PATH" \
  --mode stdio \
  --servers "$TARGET_SERVERS" >/tmp/mcp_gateway_smoke.log 2>&1 &
GATEWAY_PID=$!
export MCP_GATEWAY_ADDRESS="http://127.0.0.1:${GATEWAY_PORT}"
sleep 2

echo "[smoke] listing tools..."
python - <<'PY'
import os, sys, asyncio
from verl_tool.servers.tools.utils.mcp_client import mcp_sse_session

targets = [t.strip() for t in os.environ.get("TARGET_SERVERS","").split(",") if t.strip()]
if not targets:
    raise SystemExit("No targets resolved")

async def list_tools(server: str):
    async with mcp_sse_session(server) as session:
        tools = await session.list_tools()
        return [getattr(t, "name", None) or str(t) for t in tools]

async def main():
    ok = True
    for server in targets:
        try:
            tools = await list_tools(server)
            if tools:
                print(f"[smoke] {server} OK, tools={tools}")
            else:
                print(f"[smoke] {server} reachable but no tools listed")
            ok = ok and bool(tools)
        except Exception as e:
            print(f"[smoke] {server} FAILED: {e}")
            ok = False
    if not ok:
        sys.exit(1)

asyncio.run(main())
PY

echo "[smoke] done."
