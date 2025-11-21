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
#   - reasoning_fast: Reasoning model fast training (20-30 min, 200 samples, 1 epoch)
#   - reasoning_1hour: Reasoning model 1-hour training (500 samples, 2 epochs)
#   - reasoning_3hour: Reasoning model 3-hour training (1000 samples, 3 epochs)
#   - reasoning_2000samples_4GPUs: Reasoning model with 2000 samples using 4x A40 GPUs
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
    GPU_COUNT=0
else
    echo "$GPU_INFO"
    GPU_NAME=$(echo "$GPU_INFO" | cut -d',' -f1 | xargs)
    GPU_MEMORY=$(echo "$GPU_INFO" | cut -d',' -f2 | xargs | grep -oE '[0-9]+' | head -1)
    GPU_COUNT=$(echo "$GPU_INFO" | wc -l | xargs)
    # Set PyTorch memory allocator for better memory management
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    echo "Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True for better memory management"
    echo "Detected $GPU_COUNT GPU(s)"
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
    reasoning_fast)
        CONFIG_FILE="examples/cloud_gsm8k/gsm8k_grpo_reasoning_fast.yaml"
        TRAIN_SCRIPT="examples/cloud_gsm8k/gsm8k_grpo_train.py"
        EXPERIMENT_NAME="gsm8k-grpo-reasoning-fast"
        echo "Using REASONING FAST training configuration (20-30 minutes)"
        echo "Note: Trains reasoning model with XML format"
        ;;
    reasoning_1hour)
        CONFIG_FILE="examples/cloud_gsm8k/gsm8k_grpo_reasoning_1hour.yaml"
        TRAIN_SCRIPT="examples/cloud_gsm8k/gsm8k_grpo_train.py"
        EXPERIMENT_NAME="gsm8k-grpo-reasoning-1hour"
        echo "Using REASONING 1-HOUR training configuration (~1-2 hours)"
        echo "Note: Trains reasoning model with XML format (500 samples)"
        ;;
    reasoning_3hour)
        CONFIG_FILE="examples/cloud_gsm8k/gsm8k_grpo_reasoning_3hour.yaml"
        TRAIN_SCRIPT="examples/cloud_gsm8k/gsm8k_grpo_train.py"
        EXPERIMENT_NAME="gsm8k-grpo-reasoning-3hour"
        echo "Using REASONING 3-HOUR training configuration (~3-4 hours)"
        echo "Note: Trains reasoning model with XML format (1000 samples)"
        ;;
    reasoning_2000samples_4GPUs)
        # Check GPU count
        if [ -z "$GPU_COUNT" ] || [ "$GPU_COUNT" -lt 4 ]; then
            echo "ERROR: This config requires 4 GPUs"
            echo "Detected: $GPU_COUNT GPU(s)"
            echo ""
            echo "This config is optimized for 4x A40 GPUs (48GB each)"
            echo "Please use a pod with 4 GPUs or use a different config"
            exit 1
        fi
        CONFIG_FILE="examples/cloud_gsm8k/gsm8k_grpo_reasoning_2000samples_4GPUs.yaml"
        TRAIN_SCRIPT="examples/cloud_gsm8k/gsm8k_grpo_train.py"
        EXPERIMENT_NAME="gsm8k-grpo-reasoning-4gpu-2000samples"
        echo "Using REASONING 2000 SAMPLES 4 GPUs training configuration"
        echo "Note: Trains reasoning model with XML format (2000 samples, 4x A40 GPUs)"
        echo "GPU count: $GPU_COUNT (required: 4)"
        ;;
    *)
        echo "ERROR: Unknown config name: $CONFIG_NAME"
        echo "Valid options: fast, 1hour, 3hour, full, reasoning_fast, reasoning_1hour, reasoning_3hour, reasoning_2000samples_4GPUs"
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

TRAINING_EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
else
    echo "⚠️  Training exited with code: $TRAINING_EXIT_CODE"
fi
echo "=========================================="
echo "Checkpoints: outputs/grpo/checkpoints/$EXPERIMENT_NAME/$TRIAL_NAME"
echo "Logs: outputs/grpo/logs/root/$EXPERIMENT_NAME/$TRIAL_NAME"
echo "WandB: https://wandb.ai (project: gsm8k-grpo-local)"
echo "=========================================="

# Create completion marker to prevent re-running
# This is critical to prevent RunPod from wasting money by re-running the script
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    COMPLETION_MARKER="/workspace/outputs/training_completed_$(date +%Y%m%d_%H%M%S).marker"
    echo "Training completed successfully at $(date)" > "$COMPLETION_MARKER"
    echo "Experiment: $EXPERIMENT_NAME" >> "$COMPLETION_MARKER"
    echo "Trial: $TRIAL_NAME" >> "$COMPLETION_MARKER"
    echo "Config: $CONFIG_NAME" >> "$COMPLETION_MARKER"
    echo ""
    echo "✅ Created completion marker: $COMPLETION_MARKER"
    echo "   This prevents the script from re-running if the container restarts."
fi

echo ""
echo "=========================================="
echo "⚠️  IMPORTANT: Stop the pod to save costs!"
echo "=========================================="
echo "Training has finished. To prevent wasting money:"
echo ""
echo "1. Go to RunPod dashboard: https://www.runpod.io/console/pods"
echo "2. Find your pod and click 'Stop'"
echo ""
echo "Your checkpoints are safe in the network volume!"
echo "They will persist even after the pod stops."
echo ""
echo "If the pod auto-restarts, the completion marker will"
echo "prevent the training script from running again."
echo "=========================================="
echo ""

# Exit with the training exit code
exit $TRAINING_EXIT_CODE
