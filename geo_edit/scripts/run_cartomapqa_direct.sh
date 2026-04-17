#!/usr/bin/env bash
set -x

API_BASE=${API_BASE:-"http://127.0.0.1:8000"}
MODEL_PATH=${MODEL_PATH:-"/storage/openpsi/models/Qwen3-VL-8B-Thinking/"}
MODEL_TYPE=${MODEL_TYPE:-"vLLM"}
OUTPUT_BASE=${OUTPUT_BASE:-"/storage/openpsi/data/lcy_image_edit/cartomapqa_test_0417"}
DATASET_ROOT="/storage/openpsi/data/lcy_image_edit/CartoMapQA_parquet"
MAX_CONCURRENT=${MAX_CONCURRENT:-64}
SAMPLE_RATE=${SAMPLE_RATE:-1.0}
MAX_TOOL_CALLS=${MAX_TOOL_CALLS:-10}
ENABLE_TOOLS=${ENABLE_TOOLS:-"map general"}

# Mode: "direct" (no tools) or "tool" (with tool calls)
MODE=${MODE:-"direct"}

# Each entry: "parquet_filename:dataset_name"
SUBSETS=(
    "MFS:cartomapqa_mfs"
    "MML:cartomapqa_mml"
    "MTMF:cartomapqa_mtmf"
    "RLE:cartomapqa_rle"
    "SRN:cartomapqa_srn"
    "STMF_counting:cartomapqa_stmf_counting"
    "STMF_name_listing:cartomapqa_stmf_name_listing"
    "STMF_presence:cartomapqa_stmf_presence"
)

total=${#SUBSETS[@]}
idx=0

for entry in "${SUBSETS[@]}"; do
    idx=$((idx + 1))
    parquet_name="${entry%%:*}"
    dataset_name="${entry##*:}"
    parquet_path="${DATASET_ROOT}/${parquet_name}.parquet"
    output_dir="${OUTPUT_BASE}/${parquet_name}"

    echo "=== [$idx/$total] ${parquet_name} (${dataset_name}) [${MODE}] ==="

    if [ ! -f "$parquet_path" ]; then
        echo "  SKIP: $parquet_path not found"
        continue
    fi

    if [ "$MODE" = "tool" ]; then
        python -m geo_edit.scripts.async_generate_with_tool_call_api \
            --api_base "$API_BASE" \
            --dataset_path "$parquet_path" \
            --dataset_name "$dataset_name" \
            --output_dir "$output_dir" \
            --model_name_or_path "$MODEL_PATH" \
            --model_type "$MODEL_TYPE" \
            --sample_rate "$SAMPLE_RATE" \
            --use_tools auto \
            --max_concurrent_requests "$MAX_CONCURRENT" \
            --max_tool_calls "$MAX_TOOL_CALLS" \
            --enable_tools $ENABLE_TOOLS
    else
        python -m geo_edit.scripts.direct_generate \
            --api_base "$API_BASE" \
            --dataset_path "$parquet_path" \
            --dataset_name "$dataset_name" \
            --output_dir "$output_dir" \
            --model_name_or_path "$MODEL_PATH" \
            --model_type "$MODEL_TYPE" \
            --api_mode chat_completions \
            --max_concurrent_requests "$MAX_CONCURRENT" \
            --sample_rate "$SAMPLE_RATE"
    fi

    echo "  DONE: $parquet_name -> $output_dir"
done

echo "All $total subsets completed."
