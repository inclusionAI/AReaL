#!/bin/bash
# Multi-Session Full Dataset Training Script
# Processes ALL GSM8K training samples across multiple sessions
# Automatically resumes from last checkpoint if interrupted

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Configuration
EXPERIMENT_NAME="${EXPERIMENT_NAME:-gsm8k-grpo-full-local}"
TRIAL_NAME="${TRIAL_NAME:-trial0}"
CONFIG_FILE="${CONFIG_FILE:-examples/docker_gsm8k/gsm8k_grpo_full.yaml}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-examples/docker_gsm8k/gsm8k_grpo_train.py}"

# Session configuration (optional - can be overridden)
# Set MAX_STEPS_PER_SESSION to limit steps per session (0 = no limit)
MAX_STEPS_PER_SESSION="${MAX_STEPS_PER_SESSION:-0}"
# Set MAX_TIME_PER_SESSION to limit time per session in seconds (0 = no limit)
MAX_TIME_PER_SESSION="${MAX_TIME_PER_SESSION:-0}"

echo "=========================================="
echo "Multi-Session Full Dataset Training"
echo "=========================================="
echo "Experiment: $EXPERIMENT_NAME"
echo "Trial: $TRIAL_NAME"
echo "Config: $CONFIG_FILE"
echo "Script: $TRAIN_SCRIPT"
echo ""
echo "Session Limits:"
echo "  Max steps per session: ${MAX_STEPS_PER_SESSION:-unlimited}"
echo "  Max time per session: ${MAX_TIME_PER_SESSION:-unlimited}"
echo "=========================================="
echo ""

# Check WandB API key
# Trim whitespace from environment variable if set
if [ -n "$WANDB_API_KEY" ]; then
    export WANDB_API_KEY=$(echo "$WANDB_API_KEY" | tr -d '\r\n' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
fi

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
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "WARNING: nvidia-smi not available. GPU may not be accessible."
fi

echo ""
echo "Starting training session..."
echo "Note: Training will automatically resume from last checkpoint if interrupted."
echo "Press Ctrl+C to stop current session (checkpoint will be saved)."
echo ""

# Run training
# The script will automatically:
# 1. Load last checkpoint if it exists (recover.mode: auto)
# 2. Continue training from where it left off
# 3. Save checkpoints frequently (every 50 steps or 30 minutes)
# 4. Can be interrupted and resumed anytime

python3 -m areal.launcher.local "$TRAIN_SCRIPT" \
    --config "$CONFIG_FILE" \
    experiment_name="$EXPERIMENT_NAME" \
    trial_name="$TRIAL_NAME"

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Training completed successfully. Running quick validation..."
    echo "=========================================="
    
    # Find latest checkpoint
    # Note: In Docker, user is usually 'root'
    CHECKPOINT_BASE="./outputs/grpo/checkpoints/root/${EXPERIMENT_NAME}/${TRIAL_NAME}/default"
    
    if [ -d "$CHECKPOINT_BASE" ]; then
        # Get latest checkpoint directory (sort by time descending)
        LATEST_CHECKPOINT=$(ls -td "$CHECKPOINT_BASE"/*/ | head -1)
        
        if [ -n "$LATEST_CHECKPOINT" ]; then
            echo "Found latest checkpoint: $LATEST_CHECKPOINT"
            
            # Determine test script based on experiment name or config
            if [[ "$EXPERIMENT_NAME" == *"reasoning"* ]] || [[ "$CONFIG_FILE" == *"reasoning"* ]]; then
                echo "Detected reasoning model. Running test_reasoning_model.py (30 samples)..."
                python3 examples/docker_gsm8k/test_reasoning_model.py \
                    --model-path "$LATEST_CHECKPOINT" \
                    --max-samples 30 \
                    --max-new-tokens 1024
            else
                echo "Running standard test_trained_model.py (30 samples)..."
                python3 examples/docker_gsm8k/test_trained_model.py \
                    --config "$CONFIG_FILE" \
                    --max-samples 30
            fi
        else
            echo "WARNING: No checkpoint subdirectories found in $CHECKPOINT_BASE"
        fi
    else
        echo "WARNING: Checkpoint directory not found: $CHECKPOINT_BASE"
    fi
fi

echo ""
echo "=========================================="
echo "Training session completed or stopped."
echo "=========================================="
echo ""
echo "To resume training, simply run this script again:"
echo "  bash $0"
echo ""
echo "The training will automatically resume from the last checkpoint."
echo ""
