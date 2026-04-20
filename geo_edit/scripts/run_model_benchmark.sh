#!/usr/bin/env bash
# ==============================================================================
# Model Benchmark Pipeline: vLLM launch → inference → eval_unified → summary
#
# One-command evaluation on 7 geo/map datasets with LLM judge.
#
# Usage:
#   bash geo_edit/scripts/run_model_benchmark.sh \
#     --model /path/to/model \
#     [--mode tool|direct]              # default: tool
#     [--datasets "ds1 ds2 ..."]        # default: all 7
#     [--judge-model MODEL]             # default: gpt-4.1-mini-2025-04-14
#     [--judge-api-key KEY]             # or env JUDGE_API_KEY
#     [--judge-api-base URL]            # or env JUDGE_API_BASE
#     [--no-image-compression]          # disable client-side 4MB base64 compression
#     [--gpu-mem-util 0.7]              # vLLM GPU memory utilization (default: 0.8)
#     [--skip-vllm]                     # assume vLLM already running
#     [--skip-inference]                # only run eval on existing results
#     [--skip-eval]                     # only run inference, no eval
#     [--port 8000]                     # vLLM port (default: 8000)
#     [--max-concurrent 64]             # parallel requests (default: 64)
#     [--sample-rate 1.0]               # sample rate (default: 1.0)
#     [--output-root DIR]               # inference output root
#     [--eval-output-root DIR]          # eval output root
#
# Available datasets:
#   visual_probe_easy  visual_probe_medium  visual_probe_hard
#   map_trace  reason_map  reason_map_plus  mapqa
#
# Examples:
#   # Full pipeline, all 7 datasets, tool mode, no compression
#   bash geo_edit/scripts/run_model_benchmark.sh \
#     --model /storage/openpsi/models/Qwen3-VL-8B-Thinking \
#     --no-image-compression
#
#   # Only mapqa in direct mode, skip vLLM launch
#   bash geo_edit/scripts/run_model_benchmark.sh \
#     --model /path/to/model --mode direct \
#     --datasets "mapqa" --skip-vllm
#
#   # Only eval on existing inference results
#   bash geo_edit/scripts/run_model_benchmark.sh \
#     --model /path/to/model --skip-vllm --skip-inference
# ==============================================================================
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL_PATH=""
MODE="tool"
PORT=8000
MAX_CONCURRENT=64
SAMPLE_RATE="1.0"
MAX_TOOL_CALLS=10
GPU_MEM_UTIL="0.8"
NO_IMAGE_COMPRESSION=""
SKIP_VLLM=false
SKIP_INFERENCE=false
SKIP_EVAL=false
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4.1-mini-2025-04-14}"
JUDGE_API_KEY="${JUDGE_API_KEY:-}"
JUDGE_API_BASE="${JUDGE_API_BASE:-}"
OUTPUT_ROOT="/storage/openpsi/data/lcy_image_edit/eval_results"
EVAL_OUTPUT_ROOT="/storage/openpsi/data/lcy_image_edit/eval_output"
ALL_DATASETS="visual_probe_easy visual_probe_medium visual_probe_hard map_trace reason_map reason_map_plus mapqa"
DATASETS=""

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)                MODEL_PATH="$2"; shift 2 ;;
        --mode)                 MODE="$2"; shift 2 ;;
        --datasets)             DATASETS="$2"; shift 2 ;;
        --judge-model)          JUDGE_MODEL="$2"; shift 2 ;;
        --judge-api-key)        JUDGE_API_KEY="$2"; shift 2 ;;
        --judge-api-base)       JUDGE_API_BASE="$2"; shift 2 ;;
        --no-image-compression) NO_IMAGE_COMPRESSION=1; shift ;;
        --gpu-mem-util)         GPU_MEM_UTIL="$2"; shift 2 ;;
        --skip-vllm)            SKIP_VLLM=true; shift ;;
        --skip-inference)       SKIP_INFERENCE=true; shift ;;
        --skip-eval)            SKIP_EVAL=true; shift ;;
        --port)                 PORT="$2"; shift 2 ;;
        --max-concurrent)       MAX_CONCURRENT="$2"; shift 2 ;;
        --sample-rate)          SAMPLE_RATE="$2"; shift 2 ;;
        --output-root)          OUTPUT_ROOT="$2"; shift 2 ;;
        --eval-output-root)     EVAL_OUTPUT_ROOT="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,/^set -/p' "$0" | head -n -1; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: --model is required"; exit 1
fi

DATASETS="${DATASETS:-$ALL_DATASETS}"
MODEL_NAME=$(basename "$MODEL_PATH")
API_BASE="http://127.0.0.1:${PORT}"

# ── Dataset registry ──────────────────────────────────────────────────────────
declare -A DATASET_PATHS=(
    ["visual_probe_easy"]="/storage/openpsi/data/VisualProbe_Easy/val.parquet"
    ["visual_probe_medium"]="/storage/openpsi/data/VisualProbe_Medium/val.parquet"
    ["visual_probe_hard"]="/storage/openpsi/data/VisualProbe_Hard/val.parquet"
    ["map_trace"]="/storage/openpsi/data/lcy_image_edit/maptrace_val_2851.parquet"
    ["reason_map"]="/storage/openpsi/data/ReasonMap/reasonmap_base_validation_dataset.parquet"
    ["reason_map_plus"]="/storage/openpsi/data/ReasonMap_plus/reasonmap_plus_test.parquet"
    ["mapqa"]="/storage/openpsi/data/lcy_image_edit/MapQA_all/mapqa_test_0418.parquet"
)

declare -A DATASET_EVAL_NAMES=(
    ["visual_probe_easy"]="visual_probe"
    ["visual_probe_medium"]="visual_probe"
    ["visual_probe_hard"]="visual_probe"
    ["map_trace"]="map_trace"
    ["reason_map"]="reason_map"
    ["reason_map_plus"]="reason_map_plus"
    ["mapqa"]="mm_mapqa"
)

# map_trace uses NDTW scoring, no LLM judge needed
declare -A SKIP_JUDGE=(
    ["map_trace"]=1
)

# ── Helpers ───────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }

kill_vllm() {
    log "Killing existing vLLM/GPU processes..."
    ps aux | grep -E "vllm|api_server" | grep -v grep \
        | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
        | sort -u | xargs -r kill -9 2>/dev/null || true
    sleep 3
}

wait_vllm() {
    local max_wait=300 waited=0
    until curl -sf "${API_BASE}/v1/models" > /dev/null 2>&1; do
        sleep 5; waited=$((waited + 5))
        if [ $waited -ge $max_wait ]; then
            log "ERROR: vLLM not ready after ${max_wait}s"
            tail -20 /tmp/log/vllm_api.log 2>/dev/null; exit 1
        fi
    done
    log "vLLM ready (${waited}s)"
}

# ── Stage 1: Launch vLLM ─────────────────────────────────────────────────────
if [ "$SKIP_VLLM" = false ]; then
    kill_vllm
    log "Launching vLLM: model=$MODEL_NAME port=$PORT gpu_mem=$GPU_MEM_UTIL"
    GPU_MEM_UTIL="$GPU_MEM_UTIL" \
        bash geo_edit/scripts/launch_vllm_generate.sh "$MODEL_PATH" "$PORT"
else
    log "Skipping vLLM launch (--skip-vllm)"
    wait_vllm
fi

# ── Stage 2: Inference ────────────────────────────────────────────────────────
if [ "$SKIP_INFERENCE" = false ]; then
    COMPRESS_FLAG=""
    [ -n "$NO_IMAGE_COMPRESSION" ] && COMPRESS_FLAG="--no_image_compression"

    total=$(echo $DATASETS | wc -w)
    idx=0
    log "=== Inference: $total datasets, mode=$MODE ==="

    for ds_key in $DATASETS; do
        idx=$((idx + 1))
        ds_path="${DATASET_PATHS[$ds_key]:-}"
        ds_name="${DATASET_EVAL_NAMES[$ds_key]:-}"
        output_dir="${OUTPUT_ROOT}/${ds_key}/${MODEL_NAME}_${MODE}"

        if [ -z "$ds_path" ]; then
            log "[$idx/$total] SKIP unknown dataset: $ds_key"; continue
        fi
        if [ ! -f "$ds_path" ]; then
            log "[$idx/$total] SKIP missing file: $ds_path"; continue
        fi

        log "[$idx/$total] $ds_key ($MODE)"

        if [ "$MODE" = "tool" ]; then
            python -m geo_edit.scripts.async_generate_with_tool_call_api \
                --api_base "$API_BASE" \
                --dataset_path "$ds_path" \
                --dataset_name "$ds_name" \
                --output_dir "$output_dir" \
                --model_name_or_path "$MODEL_PATH" \
                --model_type vLLM \
                --sample_rate "$SAMPLE_RATE" \
                --use_tools auto \
                --max_concurrent_requests "$MAX_CONCURRENT" \
                --max_tool_calls "$MAX_TOOL_CALLS" \
                --enable_tools map general \
                $COMPRESS_FLAG
        else
            python -m geo_edit.scripts.direct_generate \
                --api_base "$API_BASE" \
                --dataset_path "$ds_path" \
                --dataset_name "$ds_name" \
                --output_dir "$output_dir" \
                --model_name_or_path "$MODEL_PATH" \
                --model_type vLLM \
                --api_mode chat_completions \
                --max_concurrent_requests "$MAX_CONCURRENT" \
                --sample_rate "$SAMPLE_RATE" \
                $COMPRESS_FLAG
        fi

        log "[$idx/$total] Done: $ds_key"
    done
    log "=== All inference completed ==="
else
    log "Skipping inference (--skip-inference)"
fi

# ── Stage 3: Evaluation ──────────────────────────────────────────────────────
if [ "$SKIP_EVAL" = false ]; then
    total=$(echo $DATASETS | wc -w)
    idx=0
    log "=== Evaluation: $total datasets, judge=$JUDGE_MODEL ==="

    declare -A EVAL_ACCURACY

    for ds_key in $DATASETS; do
        idx=$((idx + 1))
        ds_name="${DATASET_EVAL_NAMES[$ds_key]:-}"
        result_path="${OUTPUT_ROOT}/${ds_key}/${MODEL_NAME}_${MODE}"
        eval_output="${EVAL_OUTPUT_ROOT}/${ds_key}/${MODEL_NAME}_${MODE}"

        if [ ! -d "$result_path" ]; then
            log "[$idx/$total] SKIP: no results at $result_path"; continue
        fi

        log "[$idx/$total] Eval: $ds_key"

        judge_args=""
        if [ "${SKIP_JUDGE[$ds_key]:-0}" != "1" ] && [ -n "$JUDGE_API_KEY" ]; then
            judge_args="--use_judge --judge_model $JUDGE_MODEL --judge_api_key $JUDGE_API_KEY"
            [ -n "$JUDGE_API_BASE" ] && judge_args="$judge_args --judge_api_base $JUDGE_API_BASE"
        fi

        eval_output_text=$(python -m geo_edit.evaluation.eval_unified \
            --dataset_name "$ds_name" \
            --result_path "$result_path" \
            --output_path "$eval_output" \
            $judge_args 2>&1) || true

        # Extract accuracy line
        acc_line=$(echo "$eval_output_text" | grep "^Accuracy:" | head -1)
        [ -z "$acc_line" ] && acc_line=$(echo "$eval_output_text" | grep "Accuracy:" | tail -1)
        EVAL_ACCURACY[$ds_key]="${acc_line:-N/A}"

        # Print detailed output
        echo "$eval_output_text" | grep -E "^(Dataset|Total|Correct|Accuracy|Avg NDTW|Median|LLM Judge|Rule-only|Per Question|  Counting|  TorF)" || true
        echo "---"
    done

    # ── Summary table ─────────────────────────────────────────────────────────
    echo ""
    echo "================================================================"
    echo "  BENCHMARK SUMMARY"
    echo "  Model:  $MODEL_NAME"
    echo "  Mode:   $MODE"
    echo "  Judge:  $JUDGE_MODEL"
    echo "  Time:   $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================"
    printf "%-25s %s\n" "Dataset" "Accuracy"
    printf "%-25s %s\n" "-------------------------" "-------------------"
    for ds_key in $DATASETS; do
        printf "%-25s %s\n" "$ds_key" "${EVAL_ACCURACY[$ds_key]:-N/A}"
    done
    echo "================================================================"
    echo "Inference: ${OUTPUT_ROOT}/*/${MODEL_NAME}_${MODE}"
    echo "Eval:      ${EVAL_OUTPUT_ROOT}/*/${MODEL_NAME}_${MODE}"
else
    log "Skipping evaluation (--skip-eval)"
fi

log "Pipeline complete."
