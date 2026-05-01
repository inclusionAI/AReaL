#!/usr/bin/env bash
# ==============================================================================
# Benchmark Qwen3-VL Thinking models in TOOL mode (enable_tools = map general)
# on the 6-dataset eval suite.
#
# Hardware: 8x NVIDIA L20X (~144GB each).
#
# Default targets:
#   - Qwen3-VL-8B-Thinking       (17GB weights):  DP=8 TP=1
#   - Qwen3-VL-30B-A3B-Thinking  (58GB weights):  DP=8 TP=1
#
# Usage:
#   bash geo_edit/scripts/run_qwen3_thinking_tool_bench.sh          # both
#   bash geo_edit/scripts/run_qwen3_thinking_tool_bench.sh 8b       # subset
# ==============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

: "${JUDGE_API_KEY:?JUDGE_API_KEY must be set}"
: "${JUDGE_API_BASE:?JUDGE_API_BASE must be set}"

DATASETS="visual_probe_easy visual_probe_medium visual_probe_hard map_trace reason_map reason_map_plus"
MODE="tool"
PORT=8000

# Model registry: tag -> "path|dp|tp|max_model_len|gpu_mem_util|extra_args"
declare -A MODEL_CFG=(
  [8b]="/storage/openpsi/models/Qwen3-VL-8B-Thinking|8|1|40960|0.85|"
  [30b]="/storage/openpsi/models/Qwen3-VL-30B-A3B-Thinking|8|1|40960|0.85|"
)

# Default: run 8b then 30b.
TARGETS=("$@")
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=(8b 30b)

LOG_DIR=/tmp/log/qwen3_thinking_tool_bench
mkdir -p "$LOG_DIR"

run_one() {
  local tag="$1"
  local cfg="${MODEL_CFG[$tag]:-}"
  if [ -z "$cfg" ]; then
    echo "[ERROR] Unknown model tag: $tag"; return 1
  fi
  IFS='|' read -r MODEL_PATH DP_SIZE TP_SIZE MAX_MODEL_LEN GPU_MEM_UTIL EXTRA_VLLM_ARGS <<< "$cfg"
  local MODEL_NAME
  MODEL_NAME=$(basename "$MODEL_PATH")
  local model_log="$LOG_DIR/${MODEL_NAME}.log"

  echo "================================================================"
  echo "  [$(date '+%F %T')] Running model: $MODEL_NAME  (mode=tool, tools=map general)"
  echo "    path=$MODEL_PATH"
  echo "    DP=$DP_SIZE  TP=$TP_SIZE  max_model_len=$MAX_MODEL_LEN  gpu_mem=$GPU_MEM_UTIL"
  echo "    log=$model_log"
  echo "================================================================"

  pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  pkill -9 -f "EngineCore" 2>/dev/null || true
  sleep 5

  DP_SIZE="$DP_SIZE" TP_SIZE="$TP_SIZE" \
  MAX_MODEL_LEN="$MAX_MODEL_LEN" GPU_MEM_UTIL="$GPU_MEM_UTIL" \
  EXTRA_VLLM_ARGS="$EXTRA_VLLM_ARGS" \
  bash "$SCRIPT_DIR/run_model_benchmark.sh" \
      --model "$MODEL_PATH" \
      --mode "$MODE" \
      --datasets "$DATASETS" \
      --no-image-compression \
      --judge-api-key "$JUDGE_API_KEY" \
      --judge-api-base "$JUDGE_API_BASE" \
      --gpu-mem-util "$GPU_MEM_UTIL" \
      --port "$PORT" \
      --max-concurrent 32 \
      2>&1 | tee "$model_log"
  local rc=${PIPESTATUS[0]}

  echo "[$(date '+%F %T')] Finished $MODEL_NAME (exit=$rc)"
  pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  pkill -9 -f "EngineCore" 2>/dev/null || true
  sleep 5
  return $rc
}

OVERALL_RC=0
for tag in "${TARGETS[@]}"; do
  if ! run_one "$tag"; then
    echo "[WARN] tag=$tag failed; continuing to next model."
    OVERALL_RC=1
  fi
done

echo "================================================================"
echo "  ALL DONE. Logs: $LOG_DIR"
echo "  Eval outputs:  /storage/openpsi/data/lcy_image_edit/eval_output/<ds>/<MODEL>_tool"
echo "================================================================"
exit $OVERALL_RC
