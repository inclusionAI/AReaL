#!/bin/bash
# Simple training script for GSM8K SFT using HuggingFace Trainer
#
# Usage:
#   bash examples/local_gsm8k/train.sh
#
# Or with custom arguments:
#   bash examples/local_gsm8k/train.sh --max-samples 1000 --batch-size 4

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Default training parameters
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/gsm8k-training}"
MAX_SAMPLES="${MAX_SAMPLES:-1500}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-3e-5}"
MAX_LENGTH="${MAX_LENGTH:-128}"
MAX_TIME="${MAX_TIME:-7200}"  # 2 hours in seconds
SAVE_STEPS="${SAVE_STEPS:-100}"
USE_WANDB="${USE_WANDB:-true}"

# Parse command line arguments (override defaults)
WANDB_FLAG=""
if [ "$USE_WANDB" = "false" ] || [ "$USE_WANDB" = "0" ]; then
    WANDB_FLAG="--no-wandb"
fi

# Build command
CMD="python examples/local_gsm8k/train_hf_trainer.py \
    --model $MODEL \
    --output-dir $OUTPUT_DIR \
    --max-samples $MAX_SAMPLES \
    --batch-size $BATCH_SIZE \
    --gradient-accumulation-steps $GRADIENT_ACCUMULATION_STEPS \
    --num-epochs $NUM_EPOCHS \
    --learning-rate $LEARNING_RATE \
    --max-length $MAX_LENGTH \
    --max-time $MAX_TIME \
    --save-steps $SAVE_STEPS"

# Add W&B flag if needed
if [ -n "$WANDB_FLAG" ]; then
    CMD="$CMD $WANDB_FLAG"
fi

# Add any additional arguments passed to the script
if [ $# -gt 0 ]; then
    CMD="$CMD $@"
fi

# Print configuration
echo "=========================================="
echo "GSM8K SFT Training (HuggingFace Trainer)"
echo "=========================================="
echo "Model: $MODEL"
echo "Output: $OUTPUT_DIR"
echo "Samples: $MAX_SAMPLES"
echo "Batch size: $BATCH_SIZE"
echo "Gradient accumulation: $GRADIENT_ACCUMULATION_STEPS"
echo "Epochs: $NUM_EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "Max length: $MAX_LENGTH"
echo "Max time: ${MAX_TIME}s (~$(($MAX_TIME / 3600)) hours)"
echo "Save steps: $SAVE_STEPS"
echo "W&B: $USE_WANDB"
echo "=========================================="
echo ""

# Run training
eval $CMD

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
echo "Model saved to: $OUTPUT_DIR"
echo "=========================================="

# Run comparison test (Base vs Trained)
echo ""
echo "=========================================="
echo "Running model comparison (Base vs Trained) - 50 samples"
echo "=========================================="

python examples/local_gsm8k/test_model.py \
    --compare \
    --base-model "$MODEL" \
    --trained-model "$OUTPUT_DIR" \
    --max-samples 50

echo ""
echo "=========================================="
echo "Process completed!"
echo "=========================================="

