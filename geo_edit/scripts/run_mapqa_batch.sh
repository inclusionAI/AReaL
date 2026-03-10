#!/usr/bin/env bash
# Batch MapQA tool experiments
set -euo pipefail

# Require OPENAI_API_KEY environment variable
if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "Error: OPENAI_API_KEY environment variable is required"
    echo "Usage: OPENAI_API_KEY=your_key $0 <dataset_path> [dataset_name] [output_dir]"
    exit 1
fi

DATASET_PATH="/storage/openpsi/data/lcy_image_edit/mapeval_visual.parquet"
DATASET_NAME="mapeval_visual"
OUTPUT_DIR="/storage/openpsi/data/lcy_image_edit/mapeval_0309/mapqa_batch"

# Experiment definitions: "tools|rounds"
EXPERIMENTS=(
    # Phase A: Single tools (1 round baseline)
    # "text_ocr|1"  # done
    # "text_spotting|1"  # done
    "grounding_dino|1"
    "auto_segment|1"
    "bbox_segment|1"
    "map_text_ocr|1"  # new: filtered OCR for maps
    # Phase B: Crop + multi-round (2 rounds)
    "image_crop text_ocr|2"
    "image_crop text_spotting|2"
    "image_crop grounding_dino|2"
    "image_crop auto_segment|2"
    "image_crop map_text_ocr|2"  # new: crop + filtered OCR
    # Phase B: Crop + multi-round (3 rounds)
    "image_crop text_ocr|3"
    "image_crop grounding_dino text_spotting|3"
    "image_crop text_ocr text_spotting|3"
    "image_crop grounding_dino auto_segment|3"
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
