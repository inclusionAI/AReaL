#!/usr/bin/env bash
# Launch geo_edit tool server with separate agent processes.
#
# Each agent (paddleocr, sam3, chartr1, etc.) runs as its own tool_server
# subprocess so you can see each one's logs independently.
#
# Usage:
#   bash launch_tool_server.sh                    # launch all agents
#   AGENTS="geo_edit_function,geo_chartr1" bash launch_tool_server.sh   # specific agents only
#   PORT=30888 bash launch_tool_server.sh         # custom port
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VERL_TOOL_ROOT="${VERL_TOOL_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30888}"
LOG_DIR="${LOG_DIR:-$VERL_TOOL_ROOT/tool-server-logs}"

# Which agents to load.
# geo_edit_function = CPU-only function tools (no GPU needed)
# geo_paddleocr, geo_sam3, geo_multimath, geo_chartr1, geo_grounding_dino, geo_gllava = GPU agents
#
# For ChartQA training: geo_edit_function + geo_paddleocr + geo_sam3 + geo_chartr1 + geo_grounding_dino
# To override: AGENTS="geo_edit_function,geo_chartr1" bash launch_tool_server.sh
DEFAULT_AGENTS="geo_edit_function,geo_paddleocr,geo_sam3,geo_multimath,geo_chartr1,geo_grounding_dino,geo_gllava"
AGENTS="${AGENTS:-$DEFAULT_AGENTS}"

mkdir -p "$LOG_DIR"

# Kill old tool server and free the port
echo "Cleaning up old processes..."
pkill -9 -f "verl_tool.servers" 2>/dev/null || true
lsof -ti tcp:$PORT | xargs -r kill -9 2>/dev/null || true
sleep 2

echo "=========================================="
echo " Launching geo_edit tool server"
echo "  Host:   $HOST:$PORT"
echo "  Agents: $AGENTS"
echo "  Logs:   $LOG_DIR"
echo "=========================================="

# Start server — logs stream to terminal via tee
SERVE_LOG="$LOG_DIR/serve_$(date +%Y%m%d_%H%M%S).log"
python3 -m verl_tool.servers.serve \
    --host "$HOST" \
    --port "$PORT" \
    --tool_type "$AGENTS" \
    --workers_per_tool 8 \
    --max_concurrent_requests 128 \
    --use_ray True \
    --log_level info \
    2>&1 | tee "$SERVE_LOG" &
SERVER_PID=$!

# Wait for router healthy (silent — server logs are already streaming above)
ROUTER_HEALTHY=false
for i in $(seq 1 90); do
    if python3 -c "
import urllib.request
try:
    r = urllib.request.urlopen('http://127.0.0.1:$PORT/health', timeout=2)
    exit(0 if b'ok' in r.read() else 1)
except: exit(1)
" 2>/dev/null; then
        ROUTER_HEALTHY=true
        break
    fi
    sleep 2
done

if [ "$ROUTER_HEALTHY" != "true" ]; then
    echo ""
    echo "ERROR: Router failed to start within 180s"
    kill -9 $SERVER_PID 2>/dev/null
    exit 1
fi

# Verify all expected agents loaded (check backend logs)
sleep 3
IFS=',' read -ra EXPECTED_AGENTS <<< "$AGENTS"
EXPECTED_COUNT=${#EXPECTED_AGENTS[@]}
ALL_OK=true
LOADED=0
FAILED_LIST=""

echo ""
echo "=========================================="
echo " Agent loading status"
echo "=========================================="
for agent in "${EXPECTED_AGENTS[@]}"; do
    agent=$(echo "$agent" | xargs)
    if grep -q "Initialized.*tool: $agent" "$LOG_DIR"/tool_server_backend_*.log 2>/dev/null || \
       grep -q "Initialized.*tool: $agent" "$SERVE_LOG" 2>/dev/null; then
        echo "  ✓ $agent"
        LOADED=$((LOADED + 1))
    elif grep -q "Failed to initialize.*tool $agent" "$LOG_DIR"/tool_server_backend_*.log 2>/dev/null || \
         grep -q "Failed to initialize.*tool $agent" "$SERVE_LOG" 2>/dev/null; then
        echo "  ✗ $agent  (FAILED)"
        ALL_OK=false
        FAILED_LIST="$FAILED_LIST $agent"
    else
        echo "  ? $agent  (no log entry)"
        ALL_OK=false
        FAILED_LIST="$FAILED_LIST $agent"
    fi
done

echo ""
echo " Result: $LOADED / $EXPECTED_COUNT agents loaded"
echo "=========================================="

if [ "$ALL_OK" = "true" ]; then
    echo "Server ready: http://$HOST:$PORT/get_observation  (PID=$SERVER_PID)"
    disown $SERVER_PID
    exit 0
else
    echo "WARNING: Failed agents:$FAILED_LIST"
    echo "Server still running (PID=$SERVER_PID) with loaded agents."
    disown $SERVER_PID
    exit 1
fi
