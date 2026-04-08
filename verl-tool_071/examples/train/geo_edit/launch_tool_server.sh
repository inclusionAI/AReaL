#!/usr/bin/env bash
# Launch geo_edit tool server on the worker node.
# Usage:
#   On worker node directly:  bash launch_tool_server.sh
#   With custom port:         PORT=30888 bash launch_tool_server.sh
#   With custom host:         HOST=0.0.0.0 bash launch_tool_server.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VERL_TOOL_ROOT="${VERL_TOOL_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30888}"

export PYTHONPATH=$VERL_TOOL_ROOT:${PYTHONPATH:-}
export GEOEDIT_ENABLE_TOOLS="${GEOEDIT_ENABLE_TOOLS:-general,chart}"

# Kill old tool server if any
echo "Cleaning up old processes on port $PORT..."
python3 -c "
import os, signal
for pid in os.listdir('/proc'):
    if not pid.isdigit(): continue
    try:
        cmd = open(f'/proc/{pid}/cmdline').read()
        if 'verl_tool.servers' in cmd and str(os.getpid()) != pid:
            print(f'  killing PID {pid}')
            os.kill(int(pid), signal.SIGKILL)
    except: pass
" 2>/dev/null || true
sleep 2

echo "Launching geo_edit tool server on $HOST:$PORT ..."
python3 -m verl_tool.servers.serve \
    --host "$HOST" \
    --port "$PORT" \
    --tool_type geo_edit_tool \
    --workers_per_tool 8 \
    --max_concurrent_requests 128 \
    --use_ray True &
SERVER_PID=$!
echo "Tool server PID=$SERVER_PID"

# Wait for healthy
echo "Waiting for server to be healthy..."
for i in $(seq 1 60); do
    if python3 -c "
import urllib.request
try:
    r = urllib.request.urlopen('http://127.0.0.1:$PORT/health', timeout=2)
    exit(0 if b'healthy' in r.read() else 1)
except: exit(1)
" 2>/dev/null; then
        echo "Tool server is healthy on port $PORT"
        echo ""
        echo "To stop:  kill $SERVER_PID"
        wait
        exit 0
    fi
    sleep 3
done

echo "ERROR: Tool server failed to start within 180s"
kill -9 $SERVER_PID 2>/dev/null
exit 1
