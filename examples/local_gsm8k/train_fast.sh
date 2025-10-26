#!/bin/bash
# Fast training with reduced dataset and shorter sequence length

source venv/bin/activate

echo "Starting fast training with optimized settings..."
echo "- Model: Using smaller 0.5B model"
echo "- Max length: 128 tokens"
echo "- Batch size: 2"
echo "- Device: Auto (will try MPS first)"
echo ""

python examples/local_gsm8k/train_local_simple.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --max-time 1800 \
    --batch-size 2 \
    --gradient-accumulation-steps 16 \
    --max-length 128 \
    --device auto \
    --output-dir ./outputs/gsm8k-fast \
    --wandb

