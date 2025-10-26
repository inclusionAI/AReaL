#!/bin/bash
# Download model manually using HuggingFace CLI

echo "Downloading DeepSeek-R1-Distill-Qwen-1.5B model..."

# Make sure you're authenticated (optional but recommended for private repos)
# You can skip this for public models

# Download the model
huggingface-cli download \
    deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --local-dir ./models/DeepSeek-R1-Distill-Qwen-1.5B \
    --local-dir-use-symlinks False

echo "Model downloaded to ./models/DeepSeek-R1-Distill-Qwen-1.5B"
echo ""
echo "Now you can train with:"
echo "  python examples/local_gsm8k/train_local_simple.py \\"
echo "    --model ./models/DeepSeek-R1-Distill-Qwen-1.5B \\"
echo "    --max-time 1800"

