#!/bin/bash
# Script to run training with proper memory settings

source venv/bin/activate

# Reduce MPS memory pressure (use 40% of available memory instead of default)
# Note: This ratio must be less than 1.0
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.4

echo "Starting training with memory-optimized settings..."
echo "- Batch size: 1"
echo "- Gradient accumulation: 32 (effective batch size: 32)"
echo "- Max length: 256"
echo "- Device: MPS (Mac GPU)"
echo "- Memory watermark: 40%"
echo ""

# Run with reduced batch size
python examples/local_gsm8k/train_local_simple.py \
    --model ./models/DeepSeek-R1-Distill-Qwen-1.5B \
    --max-time 1800 \
    --batch-size 1 \
    --gradient-accumulation-steps 32 \
    --max-length 256 \
    --wandb
