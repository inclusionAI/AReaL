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

# Clean up any leftover processes that might be using the GPU
echo "Cleaning up any leftover GPU processes..."

# Install cleanup tools if missing
if ! command -v fuser &> /dev/null || ! command -v lsof &> /dev/null; then
    echo "Installing cleanup tools (psmisc, lsof)..."
    apt-get update && apt-get install -y psmisc lsof || echo "WARNING: Failed to install cleanup tools"
fi

# Kill any Python processes that might be holding the GPU
pkill -9 -f "sglang" 2>/dev/null || true
pkill -9 -f "areal.launcher" 2>/dev/null || true
pkill -9 -f "torchrun" 2>/dev/null || true
pkill -9 -f "python.*gsm8k_grpo" 2>/dev/null || true
pkill -9 -f "ray" 2>/dev/null || true

# Wait a moment for processes to terminate
sleep 3

# Aggressive cleanup: kill any processes holding /dev/nvidia* open
echo "Checking for processes holding GPU device files..."
if command -v lsof &> /dev/null; then
    # Find processes using nvidia devices
    GPU_PIDS=$(lsof /dev/nvidia* 2>/dev/null | grep -v "PID" | awk '{print $2}' | sort -u)
    if [ -n "$GPU_PIDS" ]; then
        echo "Found processes holding GPU devices: $GPU_PIDS"
        for pid in $GPU_PIDS; do
            # Don't kill ourself
            if [ "$pid" != "$$" ]; then
                echo "Killing process $pid..."
                kill -9 "$pid" 2>/dev/null || true
            fi
        done
        sleep 2
    fi
fi

if command -v fuser &> /dev/null; then
    echo "Using fuser to kill processes on GPU devices..."
    for gpu_id in /dev/nvidia*; do
        if [ -e "$gpu_id" ]; then
            fuser -k -9 "$gpu_id" 2>/dev/null || true
        fi
    done
    sleep 2
fi

# Get GPU information and check status
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
    
    # Check GPU utilization and memory usage
    echo "Checking GPU status..."
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,compute_mode --format=csv,nounits || true
    
    # Warn if we are in Exclusive Process mode (which makes "busy" errors more likely)
    if nvidia-smi --query-gpu=compute_mode --format=csv,noheader | grep -i "Exclusive_Process" > /dev/null; then
        echo "WARNING: GPUs are in Exclusive Process mode. Any zombie process will block access."
    fi
    
    # Final check: ensure no Python processes are holding CUDA contexts
    # This is important because CUDA contexts can persist even after processes exit
    PYTHON_PIDS=$(pgrep -f "python.*torch|python.*cuda" 2>/dev/null || echo "")
    if [ -n "$PYTHON_PIDS" ]; then
        echo "WARNING: Found Python processes that might have CUDA contexts: $PYTHON_PIDS"
        echo "Waiting 10 seconds for contexts to release..."
        sleep 10
    fi
    
    # Verify GPU is actually accessible before starting
    echo "Verifying GPU accessibility..."
    python3 << 'EOF'
import torch
import sys
import time

max_retries = 5
retry_delay = 3

for attempt in range(max_retries):
    try:
        if torch.cuda.is_available():
            # Try to create a tensor on GPU 0
            device = torch.device("cuda:0")
            test_tensor = torch.zeros(1, device=device)
            del test_tensor
            torch.cuda.empty_cache()
            print(f"GPU accessibility verified on attempt {attempt + 1}")
            sys.exit(0)
        else:
            print(f"CUDA not available on attempt {attempt + 1}")
    except Exception as e:
        print(f"GPU accessibility check failed on attempt {attempt + 1}: {e}")
        if attempt < max_retries - 1:
            print(f"Waiting {retry_delay} seconds before retry...")
            time.sleep(retry_delay)
        else:
            print("ERROR: GPU is not accessible after multiple attempts")
            print("This usually means the GPU driver state is corrupted.")
            print("Please restart the pod to reset the GPU state.")
            sys.exit(1)

sys.exit(1)
EOF
    
    GPU_CHECK_EXIT=$?
    if [ $GPU_CHECK_EXIT -ne 0 ]; then
        echo "ERROR: GPU accessibility check failed!"
        echo "The GPU appears to be in a corrupted state."
        echo "Please restart the pod (Stop then Start) to reset the GPU driver state."
        exit 1
    fi
    
    echo "Starting training..."
    # Additional delay to ensure GPU is fully ready
    sleep 3
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

# Set CUDA environment variables for better error reporting
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Try to reverse GPU order to avoid potentially bad GPU 0
# This maps visible devices 0,1,2,3 to physical GPUs 3,2,1,0
# If physical GPU 0 is the zombie, this moves it to the last device
if [ "$GPU_COUNT" -eq 4 ]; then
    echo "Reversing GPU order to avoid potential issues with GPU 0..."
    export CUDA_VISIBLE_DEVICES=3,2,1,0
fi

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
    echo ""
    
    # Run validation on Base and Trained models
    echo "=========================================="
    echo "Running validation (Base vs Trained Model)"
    echo "=========================================="
    
    CHECKPOINT_BASE="/workspace/outputs/grpo/checkpoints/root/${EXPERIMENT_NAME}/${TRIAL_NAME}/default"
    if [ -d "$CHECKPOINT_BASE" ]; then
        LATEST_CHECKPOINT=$(ls -td "$CHECKPOINT_BASE"/*/ | head -1)
        if [ -n "$LATEST_CHECKPOINT" ]; then
            echo "Found latest checkpoint: $LATEST_CHECKPOINT"
            
            if [[ "$EXPERIMENT_NAME" == *"reasoning"* ]]; then
                echo "Detected reasoning model."
                echo "------------------------------------------"
                echo "Testing BASELINE model (Qwen/Qwen2.5-0.5B-Instruct) - 50 samples..."
                python3 examples/cloud_gsm8k/test_reasoning_model_cloud.py \
                    --model-path "Qwen/Qwen2.5-0.5B-Instruct" \
                    --max-samples 50 \
                    --max-new-tokens 1024 \
                    --model-name "Baseline"

                echo "------------------------------------------"
                echo "Testing TRAINED model - 50 samples..."
                python3 examples/cloud_gsm8k/test_reasoning_model_cloud.py \
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
                    --max-samples 50 \
                    --log-dir "/workspace/outputs/grpo/test_logs"
                
                echo "------------------------------------------"
                echo "Testing TRAINED model - 50 samples..."
                python3 examples/docker_gsm8k/test_trained_model.py \
                    --model-path "$LATEST_CHECKPOINT" \
                    --max-samples 50 \
                    --log-dir "/workspace/outputs/grpo/test_logs"
            fi
        fi
    else
         echo "WARNING: Checkpoint directory not found: $CHECKPOINT_BASE"
    fi
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
