#!/bin/bash
# Quick script to run GRPO training inside Docker container

set -e

cd /workspace/AReaL

# Set WandB API key (optional - can also set in train_grpo.yaml)
export WANDB_API_KEY=${WANDB_API_KEY:-5cd583e967c0e092a7f7be82e0479c1f71eeeab9}

# Verify AReaL is installed
echo "Checking AReaL installation..."
python -c "import areal; print(f'AReaL version: {areal.__version__}')" || {
    echo "AReaL not found. Installing..."
    pip install -e .
}

# Verify GPU
echo "Checking GPU..."
nvidia-smi

# Run training
echo "Starting GRPO training..."
python -m areal.launcher.local examples/math/gsm8k_grpo.py \
    --config examples/local_gsm8k/train_grpo.yaml \
    experiment_name=gsm8k-grpo-local \
    trial_name=trial0

