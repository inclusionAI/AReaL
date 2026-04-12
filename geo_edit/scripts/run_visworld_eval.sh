#!/usr/bin/env bash
# Run iterative sampling on all 7 VisWorld-Eval sub-datasets sequentially.
#
# Prerequisites: set API_KEY and JUDGE_API_KEY environment variables.
#
# Usage:
#   API_KEY=your_key JUDGE_API_KEY=your_judge_key bash geo_edit/scripts/run_visworld_eval.sh
set -euo pipefail

if [ -z "${API_KEY:-}" ]; then 
    echo "Error: API_KEY is required"
    exit 1
fi

DATASET_ROOT="/storage/openpsi/data/lcy_image_edit/VisWorld-Eval"
OUTPUT_ROOT="/storage/openpsi/data/lcy_image_edit/visworld_iterative_gpt5_0403"
#DONE SUBSETS=(ballgame multihop)
SUBSETS=(mmsi cube paperfolding)

echo "============================================"
echo "VisWorld-Eval Iterative Sampling"
echo "Output: $OUTPUT_ROOT"
echo "Subsets: ${SUBSETS[*]}"
echo "============================================"

total=${#SUBSETS[@]}
idx=0

for subset in "${SUBSETS[@]}"; do
    idx=$((idx + 1))
    parquet_path="${DATASET_ROOT}/${subset}/${subset}.parquet"
    output_dir="${OUTPUT_ROOT}/${subset}"

    echo ""
    echo "=== [$idx/$total] Running: $subset ==="

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
        --dataset_name visworld_eval \
        --output_dir "$output_dir" \
        --sample_rate 1.0 \
        --max_concurrent_requests 16 \
        --max_iterative_rounds 5 \
        --judge_model "gpt-5-mini-2025-08-07" \
        --judge_api_key "$API_KEY" \
        --judge_api_base "https://matrixllm.alipay.com/v1" \
        --enable_tools general map math_latex_ocr math_image_describe formula_ocr

    echo "  DONE: $subset -> $output_dir"
done

echo ""
echo "============================================"
echo "All $total subsets completed."
echo "Results: $OUTPUT_ROOT"
echo "============================================"
