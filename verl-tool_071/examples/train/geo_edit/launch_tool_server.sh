#!/usr/bin/env bash
# Launch geo_edit tool servers — one per agent + a router on the main port.
#
# Usage:
#   bash launch_tool_server.sh                                          # all agents
#   bash launch_tool_server.sh geo_edit_function geo_chartr1            # specific agents
#   PORT=30888 bash launch_tool_server.sh geo_edit_function geo_chartr1
#
# Monitor: tail -f tool-server-logs/*.log
# Stop:    pkill -f "verl_tool.servers"
set -x

LOG_DIR="${LOG_DIR:-tool-server-logs}"
mkdir -p "$LOG_DIR"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30888}"

# Agent list: from arguments, or default all
if [ $# -gt 0 ]; then
    AGENT_LIST=("$@")
else
    AGENT_LIST=(
        geo_edit_function
        geo_paddleocr
        geo_sam3
        geo_chartr1
        geo_grounding_dino
    )
fi

# Kill old processes
pkill -9 -f "verl_tool.servers" 2>/dev/null || true
lsof -ti tcp:$PORT | xargs -r kill -9 2>/dev/null || true
sleep 2

echo "Launching ${#AGENT_LIST[@]} tool servers..."

# Start each agent on its own port
BACKEND_PORT=$((PORT + 1))
BACKEND_URLS=""

for agent in "${AGENT_LIST[@]}"; do
    lsof -ti tcp:$BACKEND_PORT | xargs -r kill -9 2>/dev/null || true
    python3 -m verl_tool.servers.tool_server \
        --tool_type "$agent" \
        --host 127.0.0.1 --port "$BACKEND_PORT" \
        --workers_per_tool 8 --max_concurrent_requests 128 --use_ray True \
        > "$LOG_DIR/${agent}.log" 2>&1 &
    echo "$agent  port=$BACKEND_PORT  pid=$!"
    BACKEND_URLS="${BACKEND_URLS:+$BACKEND_URLS,}\"http://127.0.0.1:$BACKEND_PORT\""
    BACKEND_PORT=$((BACKEND_PORT + 1))
done

# Start router on the main port, forwarding to all backends
sleep 5
export VT_WORKER_BASE_URLS="[$BACKEND_URLS]"
python3 -c "
import uvicorn
from verl_tool.servers.serve import router_factory
uvicorn.run(router_factory(), host='$HOST', port=$PORT, log_level='info', access_log=False)
" > "$LOG_DIR/router.log" 2>&1 &
echo "router     port=$PORT  pid=$!"

echo ""
echo "tool_server_url=http://$HOST:$PORT/get_observation"
echo "Logs: tail -f $LOG_DIR/*.log"
echo "Stop: pkill -f 'verl_tool.servers'"

wait
