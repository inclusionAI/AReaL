#!/bin/bash
# Simple dependency installation for local training

set -e

echo "Installing required packages for local GSM8K training..."

# Activate virtual environment
source venv/bin/activate

# Install basic dependencies
pip install torch transformers datasets accelerate tqdm wandb numpy

echo "Installation complete!"
echo ""
echo "To run training, use:"
echo "  python examples/local_gsm8k/train_local_simple.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

