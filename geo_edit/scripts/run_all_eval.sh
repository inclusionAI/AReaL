#!/usr/bin/env bash
set -x

MODEL_PATH=${MODEL_PATH:?'MODEL_PATH is required'}
MODE=${MODE:-"tool"}  # "tool" or "direct"

API_BASE=${API_BASE:-"http://127.0.0.1:8000"}
MODEL_TYPE=${MODEL_TYPE:-"vLLM"}
MAX_CONCURRENT=${MAX_CONCURRENT:-64}
SAMPLE_RATE=${SAMPLE_RATE:-1.0}
MAX_TOOL_CALLS=${MAX_TOOL_CALLS:-10}
NO_IMAGE_COMPRESSION=${NO_IMAGE_COMPRESSION:-""}
OUTPUT_ROOT=${OUTPUT_ROOT:-"/storage/openpsi/data/lcy_image_edit/eval_results"}

MODEL_NAME=$(basename "$MODEL_PATH")

declare -A DATASET_PATHS
DATASET_PATHS=(
    ["visual_probe_easy"]="/storage/openpsi/data/VisualProbe_Easy/val.parquet"
    ["visual_probe_medium"]="/storage/openpsi/data/VisualProbe_Medium/val.parquet"
    ["visual_probe_hard"]="/storage/openpsi/data/VisualProbe_Hard/val.parquet"
    ["map_trace"]="/storage/openpsi/data/lcy_image_edit/maptrace_val_2851.parquet"
    ["reason_map"]="/storage/openpsi/data/ReasonMap/reasonmap_base_validation_dataset.parquet"
    ["reason_map_plus"]="/storage/openpsi/data/ReasonMap_plus/reasonmap_plus_test.parquet"
)

declare -A DATASET_NAMES
DATASET_NAMES=(
    ["visual_probe_easy"]="visual_probe"
    ["visual_probe_medium"]="visual_probe"
    ["visual_probe_hard"]="visual_probe"
    ["map_trace"]="map_trace"
    ["reason_map"]="reason_map"
    ["reason_map_plus"]="reason_map_plus"
    ["mapqa"]="mm_mapqa"
)

DATASETS=${DATASETS:-"visual_probe_easy visual_probe_medium visual_probe_hard map_trace reason_map reason_map_plus mapqa"}

EXTRA_ARGS=""
if [ -n "$NO_IMAGE_COMPRESSION" ]; then
    EXTRA_ARGS="--no_image_compression"
fi

total=$(echo $DATASETS | wc -w)
idx=0

for ds_key in $DATASETS; do
    idx=$((idx + 1))
    ds_path="${DATASET_PATHS[$ds_key]}"
    ds_name="${DATASET_NAMES[$ds_key]}"
    output_dir="${OUTPUT_ROOT}/${ds_key}/${MODEL_NAME}_${MODE}"

    echo "=== [$idx/$total] ${ds_key} (${MODE}) ==="

    if [ ! -f "$ds_path" ]; then
        echo "  SKIP: $ds_path not found"
        continue
    fi

    if [ "$MODE" = "tool" ]; then
        python -m geo_edit.scripts.async_generate_with_tool_call_api \
            --api_base "$API_BASE" \
            --dataset_path "$ds_path" \
            --dataset_name "$ds_name" \
            --output_dir "$output_dir" \
            --model_name_or_path "$MODEL_PATH" \
            --model_type "$MODEL_TYPE" \
            --sample_rate "$SAMPLE_RATE" \
            --use_tools auto \
            --max_concurrent_requests "$MAX_CONCURRENT" \
            --max_tool_calls "$MAX_TOOL_CALLS" \
            --enable_tools map general \
            $EXTRA_ARGS
    else
        python -m geo_edit.scripts.direct_generate \
            --api_base "$API_BASE" \
            --dataset_path "$ds_path" \
            --dataset_name "$ds_name" \
            --output_dir "$output_dir" \
            --model_name_or_path "$MODEL_PATH" \
            --model_type "$MODEL_TYPE" \
            --api_mode chat_completions \
            --max_concurrent_requests "$MAX_CONCURRENT" \
            --sample_rate "$SAMPLE_RATE" \
            $EXTRA_ARGS
    fi

    echo "  DONE: ${ds_key} -> ${output_dir}"
done

echo "All $total datasets completed."
echo "Results in: ${OUTPUT_ROOT}/*/${MODEL_NAME}_${MODE}"
