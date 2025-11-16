#!/bin/bash
# Run GRPO training in Docker container
# Usage: Inside Docker container: bash examples/docker_gsm8k/run_training.sh

set -e

cd /workspace/AReaL

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
echo "Starting GRPO training (single-GPU mode with disk-based weight updates)..."
echo "Config: examples/docker_gsm8k/gsm8k_grpo_fast.yaml"
echo "Experiment: gsm8k-grpo-docker"
echo ""

python3 -m areal.launcher.local examples/docker_gsm8k/gsm8k_grpo_train.py \
    --config examples/docker_gsm8k/gsm8k_grpo_fast.yaml \
    experiment_name=gsm8k-grpo-docker \
    trial_name=trial0
