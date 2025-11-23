#!/bin/bash
# Run GRPO training in Docker container
# Usage: bash examples/docker_gsm8k/run_training.sh [config_file]
# Example: bash examples/docker_gsm8k/run_training.sh examples/docker_gsm8k/gsm8k_grpo_reasoning_fast.yaml

set -e

cd /workspace/AReaL

# Configuration
CONFIG_FILE="${1:-examples/docker_gsm8k/gsm8k_grpo_fast.yaml}"
# Extract experiment name from filename (e.g., gsm8k_grpo_reasoning_fast)
EXPERIMENT_NAME=$(basename "$CONFIG_FILE" .yaml | sed 's/_/-/')
# Or just use a fixed one for docker runs if we want consistency, but dynamic is better
EXPERIMENT_NAME="${EXPERIMENT_NAME}-docker"
TRIAL_NAME="trial0"

# Load WandB API key from wandb folder if available
if [ -f "wandb/.wandb_api_key" ]; then
    # Trim whitespace (leading/trailing) from API key
    export WANDB_API_KEY=$(cat wandb/.wandb_api_key | tr -d '\r\n' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    echo "Loaded WandB API key from wandb/.wandb_api_key"
elif [ -z "$WANDB_API_KEY" ]; then
    echo "Warning: WANDB_API_KEY not set and wandb/.wandb_api_key not found"
    echo "WandB logging will be disabled"
fi

# Trim whitespace from environment variable if set
if [ -n "$WANDB_API_KEY" ]; then
    export WANDB_API_KEY=$(echo "$WANDB_API_KEY" | tr -d '\r\n' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
fi

# Verify AReaL is installed
echo "Checking AReaL installation..."
python3 -c "import areal; print(f'AReaL version: {areal.__version__}')" || {
    echo "AReaL not found. Installing..."
    pip install -e .
}

# Verify GPU
echo "Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv

# Run training
echo ""
echo "Starting GRPO training..."
echo "Config: $CONFIG_FILE"
echo "Experiment: $EXPERIMENT_NAME"
echo "Trial: $TRIAL_NAME"
echo ""

python3 -m areal.launcher.local examples/docker_gsm8k/gsm8k_grpo_train.py \
    --config "$CONFIG_FILE" \
    experiment_name="$EXPERIMENT_NAME" \
    trial_name="$TRIAL_NAME"

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Training completed successfully. Running validation..."
    echo "=========================================="
    
    # Find latest checkpoint
    CHECKPOINT_BASE="./outputs/grpo/checkpoints/root/${EXPERIMENT_NAME}/${TRIAL_NAME}/default"
    
    if [ -d "$CHECKPOINT_BASE" ]; then
        LATEST_CHECKPOINT=$(ls -td "$CHECKPOINT_BASE"/*/ | head -1)
        
        if [ -n "$LATEST_CHECKPOINT" ]; then
            echo "Found latest checkpoint: $LATEST_CHECKPOINT"
            
            # Determine test script based on experiment name or config
            if [[ "$CONFIG_FILE" == *"reasoning"* ]]; then
                echo "Detected reasoning model."
                echo "------------------------------------------"
                echo "Testing BASELINE model (Qwen/Qwen2.5-0.5B-Instruct) - 50 samples..."
                python3 examples/docker_gsm8k/test_reasoning_model.py \
                    --model-path "Qwen/Qwen2.5-0.5B-Instruct" \
                    --max-samples 50 \
                    --max-new-tokens 1024 \
                    --model-name "Baseline"

                echo "------------------------------------------"
                echo "Testing TRAINED model - 50 samples..."
                python3 examples/docker_gsm8k/test_reasoning_model.py \
                    --model-path "$LATEST_CHECKPOINT" \
                    --max-samples 50 \
                    --max-new-tokens 1024 \
                    --model-name "Trained"
            else
                echo "Running standard validation (Base vs Trained Model)"
                echo "------------------------------------------"
                echo "Testing BASELINE model (Qwen/Qwen2.5-0.5B-Instruct) - 50 samples..."
                python3 examples/docker_gsm8k/test_trained_model.py \
                    --model-path "Qwen/Qwen2.5-0.5B-Instruct" \
                    --max-samples 50

                echo "------------------------------------------"
                echo "Testing TRAINED model - 50 samples..."
                python3 examples/docker_gsm8k/test_trained_model.py \
                    --model-path "$LATEST_CHECKPOINT" \
                    --max-samples 50
            fi
        else
            echo "WARNING: No checkpoint subdirectories found in $CHECKPOINT_BASE"
        fi
    else
        echo "WARNING: Checkpoint directory not found: $CHECKPOINT_BASE"
    fi
fi

exit $TRAIN_EXIT_CODE
