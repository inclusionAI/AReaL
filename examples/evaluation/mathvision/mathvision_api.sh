#!/usr/bin/env bash
set -euo pipefail



model_name="${1:-}"
api_key="${2:-}"
split="${3:-test}"

if [ -z "$model_name" ] || [ -z "$api_key" ]; then
    echo "Usage: $0 <model_name> <api_key> [split]"
    echo "Example: $0 claude-3-7-sonnet-20250219 sk-myapikey test"
    exit 1
fi

echo "Model: $model_name"
echo "Split: $split"

# 2. 准备输出目录
out_dir="outputs/${model_name}"
pred_file="${out_dir}/mathvision_preds.jsonl"
score_file="${out_dir}/mathvision_score.json"

echo "Results will be saved to: $out_dir"



python examples/evaluation/mathvision/mathvision_api.py \
  --dataset "/storage/openpsi/data/MathVision/" \
  --model "$model_name" \
  --base-url "https://matrixllm.alipay.com/v1" \
  --api-key "$api_key" \
  --split "$split" \
  --concurrency 64 \
  --output "$pred_file" \
  --score-json "$score_file"

echo "Evaluation finished."