#!/usr/bin/env bash
set -e

# VERL_ROOT = verl-tool repo root; AREAL_ROOT = AReaL repo root (for geo_edit module).
# code/verl-tool is NOT inside AReaL, so AREAL_ROOT must be set explicitly if using agent tools.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export VERL_ROOT="${VERL_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
export AREAL_ROOT="${AREAL_ROOT:-$(cd "$VERL_ROOT/.." && pwd)}"
export LOG_DIR="${LOG_DIR:-$VERL_ROOT/tool-server-logs}"
export PORT="${PORT:-30888}"

echo "VERL_ROOT=$VERL_ROOT"
echo "AREAL_ROOT=$AREAL_ROOT"

python3 "$SCRIPT_DIR/_start_tool_server.py"
