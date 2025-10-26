#!/bin/bash
# Script to run training on CPU (most memory-efficient option)

source venv/bin/activate

echo "Starting training on CPU (most stable for 32GB RAM)..."
echo "- Batch size: 1"
echo "- Gradient accumulation: 32 (effective batch size: 32)"
echo "- Max length: 256"
echo "- Device: CPU"
echo ""

# Run on CPU to avoid MPS memory issues
python examples/local_gsm8k/train_local_simple.py \
    --model ./models/DeepSeek-R1-Distill-Qwen-1.5B \
    --max-time 1800 \
    --batch-size 1 \
    --gradient-accumulation-steps 32 \
    --max-length 256 \
    --device cpu \
    --wandb

