#!/usr/bin/env bash
set -e

# Only VERL_ROOT needed; everything else is derived.
# Override: export VERL_ROOT=/storage/openpsi/users/lichangye.lcy/antoinegg1/AReaL/verl-tool
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export VERL_ROOT="${VERL_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
export AREAL_ROOT="$(cd "$VERL_ROOT/.." && pwd)"
export LOG_DIR="${AREAL_ROOT}/tool-server-logs"
export PORT="${PORT:-30888}"

echo "VERL_ROOT=$VERL_ROOT"
echo "AREAL_ROOT=$AREAL_ROOT"

python3 "$SCRIPT_DIR/_start_tool_server.py"
