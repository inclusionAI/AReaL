#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

DATASET_RAW_DIR="/storage/openpsi/data/mm_mapqa/data"
DATASET_PROCESSED="/storage/openpsi/data/mm_mapqa/processed/mm_mapqa_flat.parquet"
DATASET_NAME="mm_mapqa"

MODEL_THINKING="/storage/openpsi/models/Qwen3-VL-8B-Thinking"
MODEL_INSTRUCT="/storage/openpsi/models/Qwen3-VL-8B-Instruct"

OUTPUT_BASE="/storage/openpsi/data/mm_mapqa/results"
PORT="${PORT:-8000}"
MAX_CONCURRENT="${MAX_CONCURRENT:-32}"
SAMPLE_RATE="${SAMPLE_RATE:-1.0}"

if [ ! -f "$DATASET_PROCESSED" ]; then
    echo "=== Step 1: Preprocessing mm_mapqa ==="
    python -m geo_edit.data_preprocess.preprocess_mm_mapqa \
        --input_dir "$DATASET_RAW_DIR" \
        --output_dir "$(dirname "$DATASET_PROCESSED")"
else
    echo "=== Step 1: Preprocessed data already exists, skipping ==="
fi

stop_vllm() {
    echo "Stopping vLLM..."
    if [ -f /tmp/log/vllm.pid ]; then
        local pid
        pid=$(cat /tmp/log/vllm.pid)
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
        rm -f /tmp/log/vllm.pid
    fi
    sleep 5
}

run_inference() {
    local model_path="$1"
    local model_tag="$2"
    local output_dir="${OUTPUT_BASE}/${model_tag}"

    echo "=== Running inference: ${model_tag} ==="
    python -m geo_edit.scripts.direct_generate \
        --dataset_path "$DATASET_PROCESSED" \
        --output_dir "$output_dir" \
        --dataset_name "$DATASET_NAME" \
        --model_name_or_path "$model_path" \
        --api_base "http://localhost:${PORT}" \
        --model_type vLLM \
        --api_mode chat_completions \
        --max_concurrent_requests "$MAX_CONCURRENT" \
        --sample_rate "$SAMPLE_RATE"
    echo "Inference done: ${output_dir}"
}

run_eval() {
    local model_tag="$1"
    local result_dir="${OUTPUT_BASE}/${model_tag}"
    local eval_dir="${OUTPUT_BASE}/eval_${model_tag}"

    echo "=== Evaluating: ${model_tag} ==="
    python -m geo_edit.evaluation.eval_mm_mapqa \
        --result_path "$result_dir" \
        --output_path "$eval_dir"
    echo "=== Results for ${model_tag}: ==="
    cat "${eval_dir}/summary.txt"
    echo ""
}

trap stop_vllm EXIT

echo "=== Step 2: Qwen3-VL-8B-Thinking ==="
bash "${SCRIPT_DIR}/launch_vllm_generate.sh" "$MODEL_THINKING" "$PORT"
run_inference "$MODEL_THINKING" "qwen3vl8b_thinking"
stop_vllm
run_eval "qwen3vl8b_thinking"

echo "=== Step 3: Qwen3-VL-8B-Instruct ==="
bash "${SCRIPT_DIR}/launch_vllm_generate.sh" "$MODEL_INSTRUCT" "$PORT"
run_inference "$MODEL_INSTRUCT" "qwen3vl8b_instruct"
stop_vllm
run_eval "qwen3vl8b_instruct"

echo ""
echo "=========================================="
echo "  All done. Results summary:"
echo "=========================================="
echo ""
echo "--- Thinking ---"
cat "${OUTPUT_BASE}/eval_qwen3vl8b_thinking/summary.txt"
echo ""
echo "--- Instruct ---"
cat "${OUTPUT_BASE}/eval_qwen3vl8b_instruct/summary.txt"
