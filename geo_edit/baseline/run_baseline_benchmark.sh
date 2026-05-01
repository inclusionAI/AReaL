#!/usr/bin/env bash
set -uo pipefail
cd /storage/openpsi/users/lichangye.lcy/antoinegg1/AReaL

MODEL_PATH="${1:?MODEL_PATH required}"
INFERENCE_KIND="${2:?KIND required: vtool_r1 | mini_o3 | direct}"
PORT="${3:-8000}"

export MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"
NAME=$(basename "$MODEL_PATH")
LOG_ROOT=/tmp/log/bench
LOG=${LOG_ROOT}/${NAME}.log
mkdir -p "$LOG_ROOT"

API_BASE="http://127.0.0.1:${PORT}"
RESULT_ROOT=/storage/openpsi/data/lcy_image_edit/eval_results
EVAL_ROOT=/storage/openpsi/data/lcy_image_edit/eval_output
SUFFIX_KIND="${INFERENCE_KIND}"
[ "$INFERENCE_KIND" = "direct" ] && SUFFIX_KIND="direct"

declare -A DS_PATH=(
  [visual_probe_easy]=/storage/openpsi/data/VisualProbe_Easy/val.parquet
  [visual_probe_medium]=/storage/openpsi/data/VisualProbe_Medium/val.parquet
  [visual_probe_hard]=/storage/openpsi/data/VisualProbe_Hard/val.parquet
  [map_trace]=/storage/openpsi/data/lcy_image_edit/maptrace_val_2851.parquet
  [reason_map]=/storage/openpsi/data/ReasonMap/reasonmap_base_validation_dataset.parquet
  [reason_map_plus]=/storage/openpsi/data/ReasonMap_plus/reasonmap_plus_test.parquet
)
declare -A DS_NAME=(
  [visual_probe_easy]=visual_probe
  [visual_probe_medium]=visual_probe
  [visual_probe_hard]=visual_probe
  [map_trace]=map_trace
  [reason_map]=reason_map
  [reason_map_plus]=reason_map_plus
)
DATASETS="visual_probe_easy visual_probe_medium visual_probe_hard map_trace reason_map reason_map_plus"

log() { echo "[$(date '+%F %T')] $*"; }

kill_vllm() {
  ps aux | grep -iE "vllm|api_server|EngineCore" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u | xargs -r kill -9 2>/dev/null
  sleep 5
}

wait_vllm() {
  local max=600 waited=0
  until curl -sf "${API_BASE}/v1/models" > /dev/null 2>&1; do
    sleep 5; waited=$((waited+5))
    [ $waited -ge $max ] && { log "vLLM not ready in ${max}s"; tail -30 /tmp/log/vllm_api.log; exit 1; }
  done
  log "vLLM ready (${waited}s)"
}

run_inference_direct() {
  local ds_key=$1
  local out_dir="${RESULT_ROOT}/${ds_key}/${NAME}_direct"
  python -m geo_edit.scripts.direct_generate \
    --api_base "$API_BASE" \
    --dataset_path "${DS_PATH[$ds_key]}" \
    --dataset_name "${DS_NAME[$ds_key]}" \
    --output_dir "$out_dir" \
    --model_name_or_path "$MODEL_PATH" \
    --model_type vLLM --api_mode chat_completions \
    --max_concurrent_requests 64 --sample_rate 1.0 \
    --no_image_compression
  RESULT_DIR="$out_dir"
}

run_inference_baseline() {
  local kind=$1 ds_key=$2
  local raw_dir="${RESULT_ROOT}/${ds_key}/${NAME}_${kind}_raw"
  local meta_dir="${RESULT_ROOT}/${ds_key}/${NAME}_${kind}"
  mkdir -p "$raw_dir"
  python -m geo_edit.baseline.${kind}.inference \
    --dataset_path "${DS_PATH[$ds_key]}" \
    --dataset_name "${DS_NAME[$ds_key]}" \
    --model_name_or_path "$MODEL_PATH" \
    --api_base "$API_BASE" \
    --output_dir "$raw_dir" \
    --sample_rate 1.0
  local jsonl=$(ls -t "$raw_dir"/${NAME}_${DS_NAME[$ds_key]}_*.jsonl 2>/dev/null | head -1)
  if [ -z "$jsonl" ]; then log "No jsonl found in $raw_dir"; RESULT_DIR=""; return 1; fi
  python -m geo_edit.baseline.convert_to_meta_info --input "$jsonl" --output_dir "$meta_dir"
  RESULT_DIR="$meta_dir"
}

run_eval() {
  local ds_key=$1 result_path=$2
  local out=${EVAL_ROOT}/${ds_key}/${NAME}_${SUFFIX_KIND}
  local judge_args="--use_judge --judge_model gpt-4.1-mini-2025-04-14 --judge_api_key $JUDGE_API_KEY --judge_api_base $JUDGE_API_BASE"
  [ "$ds_key" = "map_trace" ] && judge_args=""
  python -m geo_edit.evaluation.eval_unified \
    --dataset_name "${DS_NAME[$ds_key]}" \
    --result_path "$result_path" \
    --output_path "$out" \
    $judge_args
}

main_run() {
  log "=== Pipeline START: $NAME ($INFERENCE_KIND) MAX_MODEL_LEN=$MAX_MODEL_LEN ==="
  kill_vllm
  log "Launching vLLM..."
  GPU_MEM_UTIL="$GPU_MEM_UTIL" bash geo_edit/scripts/launch_vllm_generate.sh "$MODEL_PATH" "$PORT" &
  wait_vllm

  declare -A ACC
  for ds_key in $DATASETS; do
    log "[$ds_key] inference"
    RESULT_DIR=""
    if [ "$INFERENCE_KIND" = "direct" ]; then
      run_inference_direct "$ds_key"
    else
      run_inference_baseline "$INFERENCE_KIND" "$ds_key"
    fi
    if [ -z "$RESULT_DIR" ]; then log "[$ds_key] inference failed, skip eval"; continue; fi
    log "[$ds_key] eval -> $RESULT_DIR"
    out_text=$(run_eval "$ds_key" "$RESULT_DIR" 2>&1) || true
    acc=$(echo "$out_text" | grep "^Accuracy:" | head -1)
    ACC[$ds_key]="${acc:-N/A}"
    echo "$out_text" | grep -E "^(Dataset|Accuracy|NDTW|Judge)" | head -8
    echo "---"
  done

  echo "================================================================"
  echo "  Model:  $NAME"
  echo "  Mode:   $INFERENCE_KIND"
  echo "================================================================"
  for ds_key in $DATASETS; do
    printf "%-25s %s\n" "$ds_key" "${ACC[$ds_key]:-N/A}"
  done
  log "=== DONE ==="
}

main_run 2>&1 | tee -a "$LOG"
