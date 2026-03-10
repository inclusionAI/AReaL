#!/usr/bin/env bash
# Batch ChartQA tool experiments
set -euo pipefail

# Require OPENAI_API_KEY environment variable
if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "Error: OPENAI_API_KEY environment variable is required"
    echo "Usage: OPENAI_API_KEY=your_key $0"
    exit 1
fi

DATASET_PATH="/storage/openpsi/data/lcy_image_edit/chartqa_test.parquet"
DATASET_NAME="chartqa"
OUTPUT_DIR="/storage/openpsi/data/lcy_image_edit/chartqa_exp"

# Experiment definitions: "tools|rounds"
EXPERIMENTS=(
    # Phase A: Single tools (1 round baseline)
    "chart_data_extract|1"
    "chart_trend_analysis|1"
    "chart_text_ocr|1"
    "chartmoe|1"
    "text_ocr|1"
    # Phase B: Crop + multi-round (2 rounds)
    "image_crop chart_data_extract|2"
    "image_crop chart_trend_analysis|2"
    "image_crop chart_text_ocr|2"
    # Phase C: Multi-round (3 rounds)
    "image_crop chart_data_extract chart_trend_analysis|3"
    "image_crop chart_text_ocr chart_data_extract|3"
)

for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r tools rounds <<< "$exp"
    tools_tag=$(echo "$tools" | tr ' ' '_')
    exp_dir="${OUTPUT_DIR}/${DATASET_NAME}_${tools_tag}_${rounds}r"

    echo "=== Running: $tools ($rounds rounds) ==="

    python -m geo_edit.scripts.separated_reasoning_generate \
        --api_base "https://matrixllm.alipay.com/v1" \
        --api_key "$OPENAI_API_KEY" \
        --dataset_path "$DATASET_PATH" \
        --dataset_name "$DATASET_NAME" \
        --output_dir "$exp_dir" \
        --model_name_or_path "gpt-5-2025-08-07" \
        --model_type "OpenAI" \
        --max_tool_calls "$rounds" \
        --enable_tools $tools \
        --max_concurrent_requests 16 \
        --sample_rate 0.25
done

echo "=== All experiments completed (${#EXPERIMENTS[@]} total) ==="
