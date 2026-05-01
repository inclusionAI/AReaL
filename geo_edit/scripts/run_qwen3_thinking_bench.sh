#!/usr/bin/env bash
# ==============================================================================
# Benchmark Qwen3-VL Thinking models (30B-A3B, 32B, 235B-A22B) on the 6-dataset
# direct-generate eval suite. Adjusts DP/TP per model to fit 8x L20X (144GB).
#
# Hardware: 8x NVIDIA L20X (~144GB each).
#
# Per-model placement:
#   - Qwen3-VL-30B-A3B-Thinking   (58GB weights):  DP=8 TP=1
#   - Qwen3-VL-32B-Thinking       (63GB weights):  DP=4 TP=2  (KV-cache headroom)
#   - Qwen3-VL-235B-A22B-Thinking (439GB weights): DP=1 TP=8 + expert parallel
#
# Usage:
#   bash geo_edit/scripts/run_qwen3_thinking_bench.sh           # all 3
#   bash geo_edit/scripts/run_qwen3_thinking_bench.sh 30b 32b   # subset
# ==============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

: "${JUDGE_API_KEY:?JUDGE_API_KEY must be set}"
: "${JUDGE_API_BASE:?JUDGE_API_BASE must be set}"

DATASETS="visual_probe_easy visual_probe_medium visual_probe_hard map_trace reason_map reason_map_plus"
MODE="direct"
PORT=8000
# Long thinking outputs need long context; keep 40960 to be safe on KV cache.
MAX_MODEL_LEN_DEFAULT=40960

# Model registry: tag -> "path|dp|tp|max_model_len|gpu_mem_util|extra_args"
declare -A MODEL_CFG=(
  [30b]="/storage/openpsi/models/Qwen3-VL-30B-A3B-Thinking|8|1|40960|0.85|"
  [32b]="/storage/openpsi/models/Qwen3-VL-32B-Thinking|4|2|40960|0.85|"
  [235b]="/storage/openpsi/models/Qwen3-VL-235B-A22B-Thinking|1|8|40960|0.90|--enable-expert-parallel"
)

# Default: run all
TARGETS=("$@")
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=(30b 32b 235b)

LOG_DIR=/tmp/log/qwen3_thinking_bench
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
  echo "  [$(date '+%F %T')] Running model: $MODEL_NAME"
  echo "    path=$MODEL_PATH"
  echo "    DP=$DP_SIZE  TP=$TP_SIZE  max_model_len=$MAX_MODEL_LEN  gpu_mem=$GPU_MEM_UTIL"
  echo "    extra_args=$EXTRA_VLLM_ARGS"
  echo "    log=$model_log"
  echo "================================================================"

  # Hard kill any leftover vllm before each run.
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
  # Cleanup before next model.
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
echo "  ALL DONE. Logs under: $LOG_DIR"
echo "  Eval outputs:        /storage/openpsi/data/lcy_image_edit/eval_output/<ds>/<MODEL>_direct"
echo "================================================================"
exit $OVERALL_RC
