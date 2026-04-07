#!/usr/bin/env bash
# Run iterative sampling on all 8 CartoMapQA sub-datasets sequentially.
#
# Prerequisites: set API_KEY environment variables.
#
# Usage:
#   API_KEY=your_key bash geo_edit/scripts/run_cartomapqa.sh
set -euo pipefail

if [ -z "${API_KEY:-}" ]; then
    echo "Error: API_KEY is required"
    exit 1
fi

DATASET_ROOT="/storage/openpsi/data/lcy_image_edit/CartoMapQA_parquet"
OUTPUT_ROOT="/storage/openpsi/data/lcy_image_edit/CartoMapQA_iterative_gpt5_0406"

# Each entry: "parquet_filename:dataset_name"
SUBSETS=(
    # "MFS:cartomapqa_mfs"
    # "MML:cartomapqa_mml"
    # "MTMF:cartomapqa_mtmf" #需要结构化judge
    "RLE:cartomapqa_rle"
    # "SRN:cartomapqa_srn"
    # "STMF_counting:cartomapqa_stmf_counting"
    # "STMF_name_listing:cartomapqa_stmf_name_listing"
    # "STMF_presence:cartomapqa_stmf_presence"
)

echo "============================================"
echo "CartoMapQA Iterative Sampling"
echo "Output: $OUTPUT_ROOT"
echo "Subsets: ${#SUBSETS[@]} tasks"
echo "============================================"

total=${#SUBSETS[@]}
idx=0

for entry in "${SUBSETS[@]}"; do
    idx=$((idx + 1))
    parquet_name="${entry%%:*}"
    dataset_name="${entry##*:}"
    parquet_path="${DATASET_ROOT}/${parquet_name}.parquet"
    output_dir="${OUTPUT_ROOT}/${parquet_name}"

    echo ""
    echo "=== [$idx/$total] Running: $parquet_name ($dataset_name) ==="

    if [ ! -f "$parquet_path" ]; then
        echo "  SKIP: $parquet_path not found"
        continue
    fi

    python -m geo_edit.scripts.iterative_sampling_generate \
        --api_key "$API_KEY" \
        --api_base "https://matrixllm.alipay.com/v1" \
        --model_name_or_path "gpt-5-2025-08-07" \
        --model_type "OpenAI" \
        --dataset_path "$parquet_path" \
        --dataset_name "$dataset_name" \
        --output_dir "$output_dir" \
        --sample_rate 1.0 \
        --max_concurrent_requests 16 \
        --max_iterative_rounds 5 \
        --judge_model "gpt-5-mini-2025-08-07" \
        --judge_api_key "$API_KEY" \
        --judge_api_base "https://matrixllm.alipay.com/v1" \
        --enable_tools general map

    echo "  DONE: $parquet_name -> $output_dir"
done

echo ""
echo "============================================"
echo "All $total subsets completed."
echo "Results: $OUTPUT_ROOT"
echo "============================================"
