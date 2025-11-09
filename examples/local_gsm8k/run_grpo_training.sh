#!/bin/bash
# Run GRPO training inside Docker container
# This script should be executed inside the Docker container

set -e

cd /workspace/AReaL

# Read WandB API key from wandb folder (if exists)
if [ -f "/workspace/AReaL/wandb/.wandb_api_key" ]; then
    export WANDB_API_KEY=$(cat /workspace/AReaL/wandb/.wandb_api_key)
    echo "Loaded WandB API key from wandb/.wandb_api_key"
else
    echo "Warning: wandb/.wandb_api_key not found, using environment variable"
fi

# Verify AReaL is installed
echo "Checking AReaL installation..."
if ! python3 -c "import areal" 2>/dev/null; then
    echo "AReaL not found. Installing..."
    pip install -e .
fi

# Verify GPU
echo "Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv

# Run training
echo ""
echo "Starting GRPO training..."
echo "Config: examples/local_gsm8k/train_grpo.yaml"
echo "Experiment: gsm8k-grpo-local"
echo ""

python3 -m areal.launcher.local examples/math/gsm8k_grpo.py \
    --config examples/local_gsm8k/train_grpo.yaml \
    experiment_name=gsm8k-grpo-local \
    trial_name=trial0

