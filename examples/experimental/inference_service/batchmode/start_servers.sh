#!/bin/bash
set -euo pipefail

# ────────────────────────────────────────────────────────────────────────────
# start_servers.sh — Launch Agent + User SGLang servers via Slurm
#
# Both servers use Qwen3-235B with production-verified flags.
#
# Usage:
#   bash scripts/start_servers.sh              # default 4 hours
#   bash scripts/start_servers.sh --hours 8    # 8 hours
#   bash scripts/start_servers.sh --agent-only  # agent server only
# ────────────────────────────────────────────────────────────────────────────

CONTAINER="${CONTAINER}"
MODEL_PATH="${MODEL_PATH}"
MODEL_NAME="Qwen3-235B-A22B-Instruct-2507"
LOG_DIR="${PROJECT}/logs"

AGENT_PORT=30000
USER_PORT=30001
HOURS=48
CONTEXT_LENGTH=262144
AGENT_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hours)          HOURS="$2";          shift 2 ;;
        --agent-port)     AGENT_PORT="$2";     shift 2 ;;
        --user-port)      USER_PORT="$2";      shift 2 ;;
        --context-length) CONTEXT_LENGTH="$2"; shift 2 ;;
        --agent-only)     AGENT_ONLY=true;     shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR"

# ════════════════════════════════════════════════════════════════════════════
# Agent SGLang — benchmark target
#   - --disable-radix-cache: simulate no-cache scenario for pressure testing
#   - --tool-call-parser qwen25: REQUIRED for Qwen3 tool calling
#   - --context-length 262144: Qwen3-235B max supported
# ════════════════════════════════════════════════════════════════════════════
echo "════════════════════════════════════════════════════════════════"
echo "  Agent SGLang: ${MODEL_NAME}"
echo "  TP=8, port=${AGENT_PORT}, context=${CONTEXT_LENGTH}"
echo "  Flags: --disable-radix-cache --tool-call-parser qwen25"
echo "════════════════════════════════════════════════════════════════"

AGENT_JOBID=$(sbatch --parsable \
    --job-name=agent-sglang \
    --nodes=1 \
    --cpus-per-task=100 \
    --gres=gpu:8 \
    --mem=1500G \
    --time="${HOURS}:00:00" \
    --output="${LOG_DIR}/agent-sglang-%j.log" \
    --wrap "singularity exec --nv --no-home --writable-tmpfs --bind /storage:/storage ${CONTAINER} bash -c '
python -m sglang.launch_server \
    --model-path ${MODEL_PATH} \
    --served-model-name ${MODEL_NAME} \
    --tp 8 \
    --port ${AGENT_PORT} \
    --host 0.0.0.0 \
    --context-length ${CONTEXT_LENGTH} \
    --tool-call-parser qwen25 \
    --disable-radix-cache \
    --enable-metrics \
    --enable-deterministic-inference
'")
echo "  Job submitted: ${AGENT_JOBID}"

# ════════════════════════════════════════════════════════════════════════════
# User Sim SGLang — NOT benchmarked (radix cache ON)
# ════════════════════════════════════════════════════════════════════════════
USER_JOBID=""
if [[ "$AGENT_ONLY" == "false" ]]; then
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  User SGLang: ${MODEL_NAME}"
    echo "  TP=8, port=${USER_PORT}, context=${CONTEXT_LENGTH}"
    echo "  Flags: --tool-call-parser qwen25 (radix cache ON)"
    echo "════════════════════════════════════════════════════════════════"

    USER_JOBID=$(sbatch --parsable \
        --job-name=user-sglang \
        --nodes=1 \
        --cpus-per-task=100 \
        --gres=gpu:8 \
        --mem=1500G \
        --time="${HOURS}:00:00" \
        --output="${LOG_DIR}/user-sglang-%j.log" \
        --wrap "singularity exec --nv --no-home --writable-tmpfs --bind /storage:/storage ${CONTAINER} bash -c '
python -m sglang.launch_server \
    --model-path ${MODEL_PATH} \
    --served-model-name ${MODEL_NAME} \
    --tp 8 \
    --port ${USER_PORT} \
    --host 0.0.0.0 \
    --context-length ${CONTEXT_LENGTH} \
    --tool-call-parser qwen25 \
    --enable-metrics \
    --enable-deterministic-inference \
    --disable-radix-cache
'")
    echo "  Job submitted: ${USER_JOBID}"
fi

# ════════════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  SGLang Servers Submitted                                   ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Agent:  job=${AGENT_JOBID}  (TP=8, port=${AGENT_PORT}, --disable-radix-cache)"
if [[ -n "$USER_JOBID" ]]; then
echo "║  User:   job=${USER_JOBID}  (TP=8, port=${USER_PORT}, radix-cache ON)"
fi
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Model:   ${MODEL_NAME}"
echo "║  Context:  ${CONTEXT_LENGTH}"
echo "║  Hours:    ${HOURS}"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Next steps:                                                ║"
echo "║  1. Wait for jobs: squeue -u \$(whoami)                     ║"
echo "║  2. Get node names from squeue output                      ║"
echo "║  3. Run benchmark:                                         ║"
echo "║     bash scripts/srun_baseline.sh \\                         ║"
echo "║       --agent-jobid ${AGENT_JOBID} \\                       "
echo "║       --user-endpoint http://<user-node>:${USER_PORT}/v1   "
echo "╚══════════════════════════════════════════════════════════════╝"
