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

# Kill old tool server if any
echo "Cleaning up old processes..."
pkill -9 -f "verl_tool.servers" 2>/dev/null || true
sleep 2

echo "=========================================="
echo " Launching geo_edit tool server"
echo "  Host:   $HOST:$PORT"
echo "  Agents: $AGENTS"
echo "  Logs:   $LOG_DIR"
echo "=========================================="

# tool_server.py accepts comma-separated --tool_type.
# serve.py (router) spawns backend subprocesses, each loading all tool_types.
# We pass the agents as a single comma-separated tool_type string.
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
echo "Server PID=$SERVER_PID"

# ── Phase 1: Wait for router to be healthy ──
echo "Phase 1: Waiting for router to be healthy..."
ROUTER_HEALTHY=false
for i in $(seq 1 60); do
    if python3 -c "
import urllib.request
try:
    r = urllib.request.urlopen('http://127.0.0.1:$PORT/health', timeout=2)
    exit(0 if b'ok' in r.read() else 1)
except: exit(1)
" 2>/dev/null; then
        ROUTER_HEALTHY=true
        echo ""
        echo "Router is up on port $PORT"
        break
    fi
    echo -n "."
    sleep 3
done

if [ "$ROUTER_HEALTHY" != "true" ]; then
    echo ""
    echo "ERROR: Router failed to start within 180s"
    tail -50 "$SERVE_LOG" 2>/dev/null
    kill -9 $SERVER_PID 2>/dev/null
    exit 1
fi

# ── Phase 2: Verify all expected agents loaded successfully ──
echo "Phase 2: Verifying agent loading status..."
# Give backend logs a moment to flush
sleep 3

# Build expected agent list
IFS=',' read -ra EXPECTED_AGENTS <<< "$AGENTS"
EXPECTED_COUNT=${#EXPECTED_AGENTS[@]}

# Check backend logs for tool initialization results
# tool_server.py logs: "✓ Initialized tool: <tool_type>" or "✗ Failed to initialize tool <tool_type>: ..."
ALL_OK=true
LOADED=0
FAILED_LIST=""

for agent in "${EXPECTED_AGENTS[@]}"; do
    agent=$(echo "$agent" | xargs)  # trim whitespace
    # Check in both backend logs and serve log (tee captures everything)
    if grep -q "Initialized tool: $agent" "$LOG_DIR"/tool_server_backend_*.log 2>/dev/null || \
       grep -q "Initialized tool: $agent" "$SERVE_LOG" 2>/dev/null; then
        echo "  ✓ $agent"
        LOADED=$((LOADED + 1))
    elif grep -q "Failed to initialize tool $agent" "$LOG_DIR"/tool_server_backend_*.log 2>/dev/null || \
         grep -q "Failed to initialize tool $agent" "$SERVE_LOG" 2>/dev/null; then
        echo "  ✗ $agent  (FAILED — check logs)"
        ALL_OK=false
        FAILED_LIST="$FAILED_LIST $agent"
    else
        echo "  ? $agent  (no log entry found — may still be loading)"
        ALL_OK=false
        FAILED_LIST="$FAILED_LIST $agent"
    fi
done

echo ""
echo "=========================================="
echo " Result: $LOADED / $EXPECTED_COUNT agents loaded"
echo "=========================================="

if [ "$ALL_OK" = "true" ]; then
    echo "All agents loaded successfully!"
    echo "  Server PID: $SERVER_PID"
    echo "  URL:        http://$HOST:$PORT/get_observation"
    echo "  Logs:       tail -f $SERVE_LOG"
    echo "  Stop:       kill $SERVER_PID"
    disown $SERVER_PID
    exit 0
else
    echo "WARNING: Some agents failed to load:$FAILED_LIST"
    echo ""
    echo "--- Backend logs ---"
    for f in "$LOG_DIR"/tool_server_backend_*.log; do
        [ -f "$f" ] && echo "=== $f ===" && tail -30 "$f"
    done
    echo ""
    echo "Server is still running (PID=$SERVER_PID) with the agents that loaded."
    echo "To stop: kill $SERVER_PID"
    disown $SERVER_PID
    exit 1
fi
