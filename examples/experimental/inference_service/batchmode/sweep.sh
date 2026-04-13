#!/bin/bash
set -euo pipefail

CONTAINER="${CONTAINER}"
PROJECT="${PROJECT}"
AREAL_PATCH="${AREAL_PATCH}"
MODEL_PATH="${MODEL_PATH}"

ADMIN_KEY="dummy:0"
ROUTER_PORT=8081
DATAPROXY_PORT=8082
GATEWAY_PORT=30098
SGLANG_PORT=30000
USER_ENDPOINT="http://<node>:30001/v1"

CONCURRENCIES="${1:-5,10,15,20,25,30}"
NUM_TASKS="${2:-50}"
NUM_TRIALS="${3:-4}"
SWEEP_TAG="${4:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_BASE="$PROJECT/trajectories/sweep_${SWEEP_TAG}"

singularity exec --writable-tmpfs --no-home --nv \
    -B /storage/openpsi \
    "$CONTAINER" bash -c '
set -euo pipefail
cd /AReaL && source .venv/bin/activate
export PATH="/root/.fnm/aliases/default/bin:$PATH"

python3 -m ensurepip 2>/dev/null
python3 -m pip install -q pdm-backend toml litellm 2>/dev/null
python3 -m pip install -e '"$PROJECT"' 2>/dev/null
python3 -m pip install -e ${TAU2_DIR} 2>/dev/null

cp -a '"$AREAL_PATCH"'/areal/experimental/inference_service/* /AReaL/areal/experimental/inference_service/
cp -a '"$AREAL_PATCH"'/areal/experimental/openai/* /AReaL/areal/experimental/openai/
cp '"$AREAL_PATCH"'/areal/api/cli_args.py /AReaL/areal/api/cli_args.py
find /AReaL/areal/experimental/inference_service /AReaL/areal/experimental/openai /AReaL/areal/api -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

which openclaw || { echo "ERROR: openclaw not found"; exit 1; }

cd '"$PROJECT"'

PIDS=()
cleanup() {
    echo "Cleaning up IS processes..."
    for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    wait 2>/dev/null || true
}
trap cleanup EXIT

python3 -m areal.experimental.inference_service.router \
    --port '"$ROUTER_PORT"' --admin-api-key "'"$ADMIN_KEY"'" --log-level warning &
PIDS+=($!)
for i in $(seq 1 60); do
    curl -sf --max-time 3 http://127.0.0.1:'"$ROUTER_PORT"'/health >/dev/null 2>&1 && { echo "✓ Router OK (${i}s)"; break; }
    [ "$i" -eq 60 ] && { echo "✗ Router FAILED"; exit 1; }
    sleep 1
done

python3 -m areal.experimental.inference_service.data_proxy \
    --port '"$DATAPROXY_PORT"' \
    --backend-addr http://127.0.0.1:'"$SGLANG_PORT"' \
    --backend-type sglang \
    --tokenizer-path '"$MODEL_PATH"' \
    --admin-api-key "'"$ADMIN_KEY"'" \
    --request-timeout 600 --log-level warning &
PIDS+=($!)
for i in $(seq 1 60); do
    curl -sf --max-time 3 http://127.0.0.1:'"$DATAPROXY_PORT"'/health >/dev/null 2>&1 && { echo "✓ DataProxy OK (${i}s)"; break; }
    [ "$i" -eq 60 ] && { echo "✗ DataProxy FAILED"; exit 1; }
    sleep 1
done

curl -sf -X POST http://127.0.0.1:'"$ROUTER_PORT"'/register \
    -H "Authorization: Bearer '"$ADMIN_KEY"'" \
    -H "Content-Type: application/json" \
    -d '"'"'{"worker_addr": "http://127.0.0.1:'"$DATAPROXY_PORT"'"}'"'"' >/dev/null
echo "✓ DataProxy registered"

python3 -m areal.experimental.inference_service.gateway \
    --port '"$GATEWAY_PORT"' \
    --router-addr http://127.0.0.1:'"$ROUTER_PORT"' \
    --admin-api-key "'"$ADMIN_KEY"'" \
    --forward-timeout 600 --log-level warning &
PIDS+=($!)
for i in $(seq 1 60); do
    curl -sf --max-time 3 http://127.0.0.1:'"$GATEWAY_PORT"'/health >/dev/null 2>&1 && { echo "✓ Gateway OK (${i}s)"; break; }
    [ "$i" -eq 60 ] && { echo "✗ Gateway FAILED"; exit 1; }
    sleep 1
done

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Inference Service ready                                 ║"
echo "║  Sweep: concurrencies='"$CONCURRENCIES"'               ║"
echo "║  Tasks: '"$NUM_TASKS"' × '"$NUM_TRIALS"' trials         ║"
echo "║  Output: '"$RESULTS_BASE"'                              ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

export TAU2_DATA_DIR=${TAU2_DIR}/data
SWEEP_START=$(date +%s)

IFS="," read -ra CONCS <<< "'"$CONCURRENCIES"'"
for C in "${CONCS[@]}"; do
    for TRIAL in $(seq 1 '"$NUM_TRIALS"'); do
        RUN_DIR="'"$RESULTS_BASE"'/c${C}/trial_${TRIAL}"
        echo ""
        echo "================================================================"
        echo "  Concurrency=${C}  Trial=${TRIAL}  →  ${RUN_DIR}"
        echo "================================================================"

        python3 '"$PROJECT"'/scripts/collect_trajectories.py \
            --gateway-url http://127.0.0.1:'"$GATEWAY_PORT"' \
            --admin-api-key "'"$ADMIN_KEY"'" \
            --user-endpoint '"$USER_ENDPOINT"' \
            --model Qwen3-235B-A22B-Instruct-2507 \
            --domain airline \
            --concurrency "$C" \
            --num-tasks '"$NUM_TASKS"' \
            --max-steps 200 \
            --max-errors 10 \
            --seed 300 \
            --openclaw-cli $(which openclaw) \
            --openclaw-timeout 3000 \
            --output-dir "$RUN_DIR" || {
                echo "  ✗ FAILED: c=${C} trial=${TRIAL}"
                continue
            }
    done
done

SWEEP_END=$(date +%s)
SWEEP_DUR=$(( SWEEP_END - SWEEP_START ))

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Sweep Complete                                          ║"
echo "║  Duration: $(( SWEEP_DUR / 3600 ))h $(( (SWEEP_DUR % 3600) / 60 ))m         ║"
echo "║  Results: '"$RESULTS_BASE"'                              ║"
echo "╚══════════════════════════════════════════════════════════╝"

python3 -c "
import json, os, glob

base = \"'"$RESULTS_BASE"'\"
print()
print(\"Concurrency | Trial | Tasks | Pass | Fail | Error | Rate   | Dur(s) | tasks/min\")
print(\"-\" * 85)
for summary_path in sorted(glob.glob(os.path.join(base, \"*/*/collection_summary.json\"))):
    with open(summary_path) as f:
        s = json.load(f)
    parts = summary_path.split(\"/\")
    c_dir = [p for p in parts if p.startswith(\"c\")][0] if any(p.startswith(\"c\") for p in parts) else \"?\"
    t_dir = [p for p in parts if p.startswith(\"trial\")][0] if any(p.startswith(\"trial\") for p in parts) else \"?\"
    print(f\"{c_dir:>11} | {t_dir:>5} | {s[\"completed\"]:>5} | {s[\"passed\"]:>4} | {s[\"failed\"]:>4} | {s[\"errors\"]:>5} | {s[\"pass_rate\"]:>5.1%} | {s[\"total_time_s\"]:>6.0f} | {s[\"tasks_per_min\"]:>9.1f}\")
print()
"
'
