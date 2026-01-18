#!/bin/bash
set -euo pipefail

source .sglang/bin/activate

# Configuration
MODEL_PATH=${1:-"Qwen/Qwen3-8B"} # Replace with actual 8B path if available
PORT=30009
OUTPUT_FILE="qwen3-aime.md"

# Run Evaluation
echo "Running evaluation..."
python eval_aime.py \
    --model-path "$MODEL_PATH" \
    --output-file "$OUTPUT_FILE" \
    --tp-size 8 # Matching gpus-per-node
