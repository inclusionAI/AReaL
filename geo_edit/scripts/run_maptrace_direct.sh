#!/usr/bin/env bash
set -x

API_BASE=${API_BASE:-"http://127.0.0.1:8000"}
MODEL_PATH=${MODEL_PATH:-"/storage/openpsi/models/Qwen3-VL-8B-Thinking/"}
MODEL_TYPE=${MODEL_TYPE:-"vLLM"}
OUTPUT_BASE=${OUTPUT_BASE:-"/storage/openpsi/data/lcy_image_edit/maptrace_test_0417"}
DATASET_PATH=${DATASET_PATH:-"/storage/openpsi/data/lcy_image_edit/maptrace_val_2851.parquet"}
MAX_CONCURRENT=${MAX_CONCURRENT:-64}
SAMPLE_RATE=${SAMPLE_RATE:-1.0}

MODE=${MODE:-"direct"}

MODEL_NAME=$(basename "$MODEL_PATH")
SUFFIX=$([ "$MODE" = "tool" ] && echo "tool" || echo "direct")
OUTPUT_DIR="${OUTPUT_BASE}/${MODEL_NAME}_${SUFFIX}"

echo "=== MapTrace Val (${MODE}) ==="

if [ "$MODE" = "tool" ]; then
    python -m geo_edit.scripts.async_generate_with_tool_call_api \
        --api_base "$API_BASE" \
        --dataset_path "$DATASET_PATH" \
        --dataset_name map_trace \
        --output_dir "$OUTPUT_DIR" \
        --model_name_or_path "$MODEL_PATH" \
        --model_type "$MODEL_TYPE" \
        --sample_rate "$SAMPLE_RATE" \
        --use_tools auto \
        --max_concurrent_requests "$MAX_CONCURRENT" \
        --max_tool_calls 10 \
        --enable_tools map general
else
    python -m geo_edit.scripts.direct_generate \
        --api_base "$API_BASE" \
        --dataset_path "$DATASET_PATH" \
        --dataset_name map_trace \
        --output_dir "$OUTPUT_DIR" \
        --model_name_or_path "$MODEL_PATH" \
        --model_type "$MODEL_TYPE" \
        --api_mode chat_completions \
        --max_concurrent_requests "$MAX_CONCURRENT" \
        --sample_rate "$SAMPLE_RATE"
fi

echo "Done: ${OUTPUT_DIR}"
