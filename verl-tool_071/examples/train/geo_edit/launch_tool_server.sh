#!/usr/bin/env bash
# Launch geo_edit tool server — each agent as a separate process.
#
# Usage:
#   bash launch_tool_server.sh                    # launch all agents
#   AGENTS="geo_edit_function,geo_chartr1" bash launch_tool_server.sh
#   PORT=30888 bash launch_tool_server.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VERL_TOOL_ROOT="${VERL_TOOL_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30888}"
LOG_DIR="${LOG_DIR:-$VERL_TOOL_ROOT/tool-server-logs}"

# For ChartQA: geo_edit_function + geo_paddleocr + geo_sam3 + geo_chartr1 + geo_grounding_dino
DEFAULT_AGENTS="geo_edit_function,geo_paddleocr,geo_sam3,geo_chartr1,geo_grounding_dino"
AGENTS="${AGENTS:-$DEFAULT_AGENTS}"

mkdir -p "$LOG_DIR"

# Kill old processes and free the port
echo "Cleaning up..."
pkill -9 -f "verl_tool.servers" 2>/dev/null || true
lsof -ti tcp:$PORT | xargs -r kill -9 2>/dev/null || true
sleep 2

echo "=========================================="
echo " Launching tool servers"
echo "  Router:  $HOST:$PORT"
echo "  Agents:  $AGENTS"
echo "=========================================="

IFS=',' read -ra AGENT_LIST <<< "$AGENTS"
BACKEND_URLS=""
BACKEND_PIDS=""

# ── Start each agent as a separate tool_server.py process ──
for agent in "${AGENT_LIST[@]}"; do
    agent=$(echo "$agent" | xargs)

    # Find a free port for this backend
    BACKEND_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1]); s.close()")
    BACKEND_LOG="$LOG_DIR/${agent}.log"

    echo ""
    echo "── Starting $agent on port $BACKEND_PORT ──"
    python3 -m verl_tool.servers.tool_server \
        --tool_type "$agent" \
        --host 127.0.0.1 \
        --port "$BACKEND_PORT" \
        --workers_per_tool 8 \
        --max_concurrent_requests 128 \
        --use_ray True \
        --log_level info \
        2>&1 | tee "$BACKEND_LOG" &
    PID=$!
    echo "  PID=$PID  log=$BACKEND_LOG"

    BACKEND_URLS="${BACKEND_URLS:+$BACKEND_URLS,}http://127.0.0.1:$BACKEND_PORT"
    BACKEND_PIDS="${BACKEND_PIDS:+$BACKEND_PIDS }$PID"

    # Wait for this backend to be healthy before starting the next one
    echo "  Waiting for $agent to be healthy..."
    HEALTHY=false
    for i in $(seq 1 60); do
        if python3 -c "
import urllib.request
try:
    r = urllib.request.urlopen('http://127.0.0.1:$BACKEND_PORT/health', timeout=2)
    exit(0 if b'healthy' in r.read() else 1)
except: exit(1)
" 2>/dev/null; then
            HEALTHY=true
            echo "  ✓ $agent is healthy"
            break
        fi
        sleep 3
    done

    if [ "$HEALTHY" != "true" ]; then
        echo "  ✗ $agent failed to start! Check $BACKEND_LOG"
        echo "  Last 20 lines:"
        tail -20 "$BACKEND_LOG"
        echo ""
        echo "Continuing with remaining agents..."
    fi
done

# ── Start the router on the main port ──
echo ""
echo "=========================================="
echo " Starting router on $HOST:$PORT"
echo "  Backends: $BACKEND_URLS"
echo "=========================================="

# Convert comma-separated URLs to JSON array for the router
BACKEND_URLS_JSON=$(python3 -c "print('[' + ','.join('\"' + u + '\"' for u in '$BACKEND_URLS'.split(',')) + ']')")
export VT_WORKER_BASE_URLS="$BACKEND_URLS_JSON"

python3 -c "
import uvicorn
from verl_tool.servers.serve import router_factory
app = router_factory()
uvicorn.run(app, host='$HOST', port=$PORT, log_level='info', access_log=False)
" 2>&1 | tee "$LOG_DIR/router.log" &
ROUTER_PID=$!

# Wait for router healthy
for i in $(seq 1 30); do
    if python3 -c "
import urllib.request
try:
    r = urllib.request.urlopen('http://127.0.0.1:$PORT/health', timeout=2)
    exit(0 if b'ok' in r.read() else 1)
except: exit(1)
" 2>/dev/null; then
        echo ""
        echo "=========================================="
        echo " All done!"
        echo "  URL:    http://$HOST:$PORT/get_observation"
        echo "  Router: PID=$ROUTER_PID"
        echo "  Stop:   kill $BACKEND_PIDS $ROUTER_PID"
        echo "=========================================="
        disown $ROUTER_PID
        for pid in $BACKEND_PIDS; do disown $pid 2>/dev/null; done
        exit 0
    fi
    sleep 2
done

echo "ERROR: Router failed to start"
exit 1
