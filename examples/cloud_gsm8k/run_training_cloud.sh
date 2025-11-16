#!/bin/bash
# Cloud-optimized training script for GRPO
#
# Usage:
#   bash examples/cloud_gsm8k/run_training_cloud.sh [config_name]
#
# Config options:
#   - fast: Fast training (20-30 min, 200 samples, 1 epoch)
#   - 1hour: 1-hour training (500 samples, 2 epochs) [default]
#   - 3hour: 3-hour training (1000 samples, 3 epochs)
#   - full: Full training (all samples, 5 epochs) - REQUIRES H200/H100/A100-80GB or equivalent
#
# All configs use memory-optimized settings that work on all GPUs.

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

# Verify AReaL is installed (only install if not already installed)
echo "Checking AReaL installation..."
if ! python3 -c "import areal" 2>/dev/null; then
    echo "AReaL not found. Installing..."
    pip install -e .
else
    echo "AReaL already installed. Skipping installation."
fi

# Get GPU information
echo "Checking GPU..."
GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)
if [ -z "$GPU_INFO" ]; then
    echo "WARNING: nvidia-smi not available. GPU may not be accessible."
    GPU_NAME=""
    GPU_MEMORY=""
else
    echo "$GPU_INFO"
    GPU_NAME=$(echo "$GPU_INFO" | cut -d',' -f1 | xargs)
    GPU_MEMORY=$(echo "$GPU_INFO" | cut -d',' -f2 | xargs | grep -oE '[0-9]+' | head -1)
    # Set PyTorch memory allocator for better memory management
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    echo "Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True for better memory management"
fi

# Function to check if GPU is suitable for full training
check_full_training_gpu() {
    if [ -z "$GPU_NAME" ]; then
        echo "ERROR: Cannot detect GPU. Full training requires H200/H100/A100-80GB or equivalent."
        return 1
    fi
    
    # Check for high-end GPUs suitable for full training
    if echo "$GPU_NAME" | grep -qiE "H200|H100|A100.*80|A100.*80GB"; then
        return 0
    fi
    
    # Check memory (H200-class GPUs have 80GB+)
    if [ -n "$GPU_MEMORY" ] && [ "$GPU_MEMORY" -ge 80000 ]; then
        return 0
    fi
    
    return 1
}


# Select configuration
case "$CONFIG_NAME" in
    fast)
        CONFIG_FILE="examples/cloud_gsm8k/gsm8k_grpo_fast.yaml"
        TRAIN_SCRIPT="examples/cloud_gsm8k/gsm8k_grpo_train.py"
        EXPERIMENT_NAME="gsm8k-grpo-cloud-fast"
        echo "Using FAST training configuration (20-30 minutes)"
        ;;
    1hour)
        CONFIG_FILE="examples/cloud_gsm8k/gsm8k_grpo_1hour.yaml"
        TRAIN_SCRIPT="examples/cloud_gsm8k/gsm8k_grpo_train.py"
        EXPERIMENT_NAME="gsm8k-grpo-cloud-1hour"
        echo "Using 1-HOUR training configuration (~1-2 hours)"
        echo "Note: Uses limited dataset (500 samples)"
        ;;
    3hour)
        CONFIG_FILE="examples/cloud_gsm8k/gsm8k_grpo_3hour.yaml"
        TRAIN_SCRIPT="examples/cloud_gsm8k/gsm8k_grpo_train.py"
        EXPERIMENT_NAME="gsm8k-grpo-cloud-3hour"
        echo "Using 3-HOUR training configuration (~3-4 hours)"
        echo "Note: Uses limited dataset (1000 samples)"
        ;;
    full)
        # Full training requires high-end GPUs
        if ! check_full_training_gpu; then
            echo "ERROR: Full training requires H200, H100, A100-80GB, or equivalent GPU (80GB+ memory)"
            echo "Detected GPU: $GPU_NAME ($GPU_MEMORY MB)"
            echo ""
            echo "For full training, please use:"
            echo "  - H200 (141GB memory)"
            echo "  - H100 (80GB memory)"
            echo "  - A100 80GB (80GB memory)"
            echo ""
            echo "For other GPUs, use: fast, 1hour, or 3hour configs"
            exit 1
        fi
        
        CONFIG_FILE="examples/cloud_gsm8k/gsm8k_grpo_cloud.yaml"
        TRAIN_SCRIPT="examples/cloud_gsm8k/gsm8k_grpo_train.py"
        EXPERIMENT_NAME="gsm8k-grpo-cloud-full"
        echo "Using FULL training configuration (full dataset, 5 epochs)"
        echo "GPU: $GPU_NAME ($GPU_MEMORY MB) - suitable for full training"
        echo "Estimated time: ~5 days"
        ;;
    *)
        echo "ERROR: Unknown config name: $CONFIG_NAME"
        echo "Valid options: fast, 1hour, 3hour, full"
        exit 1
        ;;
esac

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Check if training script exists
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "ERROR: Training script not found: $TRAIN_SCRIPT"
    exit 1
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
echo "GPU: $GPU_NAME ($GPU_MEMORY MB)"
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
