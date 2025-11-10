#!/bin/bash
# Cloud-optimized training script for GRPO
# Automatically detects environment and uses appropriate configuration
#
# Usage:
#   bash examples/cloud_gsm8k/run_training_cloud.sh [config_name]
#
# Config options:
#   - fast: Fast training (20-30 min, 200 samples, 1 epoch)
#   - 1hour: 1-hour training (500 samples, 2 epochs) [default]
#   - 3hour: 3-hour training (1000 samples, 3 epochs)
#   - full: Full training (all samples, 5 epochs) - takes days

set -e

# Configuration
CONFIG_NAME="${1:-1hour}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Check if we're in the right directory
if [ ! -f "examples/cloud_gsm8k/gsm8k_grpo_cloud.yaml" ]; then
    echo "ERROR: Not in AReaL project root or cloud_gsm8k files not found"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Check WandB API key
if [ -z "$WANDB_API_KEY" ]; then
    echo "WARNING: WANDB_API_KEY not set. WandB logging will be disabled."
    echo "Set it with: export WANDB_API_KEY=your-api-key"
fi

# Verify AReaL is installed
echo "Checking AReaL installation..."
if ! python3 -c "import areal" 2>/dev/null; then
    echo "AReaL not found. Installing..."
    pip install -e .
fi

# Verify GPU
echo "Checking GPU..."
if ! nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null; then
    echo "WARNING: nvidia-smi not available. GPU may not be accessible."
fi

# Select configuration
case "$CONFIG_NAME" in
    fast)
        CONFIG_FILE="examples/cloud_gsm8k/gsm8k_grpo_fast.yaml"
        TRAIN_SCRIPT="examples/docker_gsm8k/gsm8k_grpo_fast.py"
        EXPERIMENT_NAME="gsm8k-grpo-cloud-fast"
        echo "Using FAST training configuration (20-30 minutes)"
        ;;
    1hour)
        CONFIG_FILE="examples/cloud_gsm8k/gsm8k_grpo_1hour.yaml"
        TRAIN_SCRIPT="examples/docker_gsm8k/gsm8k_grpo_1hour.py"
        EXPERIMENT_NAME="gsm8k-grpo-cloud-1hour"
        echo "Using 1-HOUR training configuration (~1-2 hours)"
        echo "Note: Uses limited dataset (500 samples) from docker_gsm8k script"
        ;;
    3hour)
        CONFIG_FILE="examples/cloud_gsm8k/gsm8k_grpo_3hour.yaml"
        TRAIN_SCRIPT="examples/docker_gsm8k/gsm8k_grpo_3hour.py"
        EXPERIMENT_NAME="gsm8k-grpo-cloud-3hour"
        echo "Using 3-HOUR training configuration (~3-4 hours)"
        echo "Note: Uses limited dataset (1000 samples) from docker_gsm8k script"
        ;;
    full)
        CONFIG_FILE="examples/cloud_gsm8k/gsm8k_grpo_cloud.yaml"
        TRAIN_SCRIPT="examples/cloud_gsm8k/gsm8k_grpo_cloud.py"
        EXPERIMENT_NAME="gsm8k-grpo-cloud-full"
        echo "Using FULL training configuration (~5 days)"
        ;;
    *)
        echo "ERROR: Unknown config name: $CONFIG_NAME"
        echo "Valid options: fast, 1hour, 3hour, full"
        exit 1
        ;;
esac

# Check if config file exists, fallback to docker_gsm8k versions
if [ ! -f "$CONFIG_FILE" ]; then
    DOCKER_CONFIG="examples/docker_gsm8k/$(basename "$CONFIG_FILE")"
    if [ -f "$DOCKER_CONFIG" ]; then
        echo "Using config from docker_gsm8k: $DOCKER_CONFIG"
        CONFIG_FILE="$DOCKER_CONFIG"
    else
        echo "ERROR: Config file not found: $CONFIG_FILE"
        exit 1
    fi
fi

# Check if training script exists, fallback to docker_gsm8k versions
if [ ! -f "$TRAIN_SCRIPT" ]; then
    DOCKER_SCRIPT="examples/docker_gsm8k/$(basename "$TRAIN_SCRIPT")"
    if [ -f "$DOCKER_SCRIPT" ]; then
        echo "Using script from docker_gsm8k: $DOCKER_SCRIPT"
        TRAIN_SCRIPT="$DOCKER_SCRIPT"
    else
        echo "ERROR: Training script not found: $TRAIN_SCRIPT"
        exit 1
    fi
fi

# Generate trial name with timestamp
TRIAL_NAME="trial_$(date +%Y%m%d_%H%M%S)"

echo ""
echo "=========================================="
echo "Starting GRPO Training (Cloud)"
echo "=========================================="
echo "Config: $CONFIG_NAME"
echo "Config file: $CONFIG_FILE"
echo "Training script: $TRAIN_SCRIPT"
echo "Experiment: $EXPERIMENT_NAME"
echo "Trial: $TRIAL_NAME"
echo "WandB API key: ${WANDB_API_KEY:0:10}..." 
echo "=========================================="
echo ""

# Run training
python3 -m areal.launcher.local "$TRAIN_SCRIPT" \
    --config "$CONFIG_FILE" \
    experiment_name="$EXPERIMENT_NAME" \
    trial_name="$TRIAL_NAME"

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
echo "Checkpoints: outputs/grpo/checkpoints/$EXPERIMENT_NAME/$TRIAL_NAME"
echo "Logs: outputs/grpo/logs/root/$EXPERIMENT_NAME/$TRIAL_NAME"
echo "WandB: https://wandb.ai (project: gsm8k-grpo-local)"
echo "=========================================="

