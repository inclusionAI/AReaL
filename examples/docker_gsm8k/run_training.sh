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

# Record training start time to verify checkpoints are from this run
TRAINING_START_TIME=$(date +%s)
TRAINING_START_TIME_READABLE=$(date -d "@$TRAINING_START_TIME" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date '+%Y-%m-%d %H:%M:%S')

# Run training
echo ""
echo "Starting GRPO training..."
echo "Config: $CONFIG_FILE"
echo "Experiment: $EXPERIMENT_NAME"
echo "Trial: $TRIAL_NAME"
echo "Training started at: $TRAINING_START_TIME_READABLE"
echo ""

# Temporarily disable exit-on-error so we can check for checkpoints even if training exits with error
set +e
python3 -m areal.launcher.local examples/docker_gsm8k/gsm8k_grpo_train.py \
    --config "$CONFIG_FILE" \
    experiment_name="$EXPERIMENT_NAME" \
    trial_name="$TRIAL_NAME"

TRAIN_EXIT_CODE=$?
TRAINING_END_TIME=$(date +%s)
set -e  # Re-enable exit-on-error for the rest of the script

# Function to check if a checkpoint is from the current training run
# Returns 0 (success) if checkpoint is recent enough, 1 (failure) otherwise
checkpoint_is_from_current_run() {
    local checkpoint_path="$1"
    if [ ! -d "$checkpoint_path" ]; then
        return 1
    fi
    
    # Get checkpoint modification time (in seconds since epoch)
    # Use the most recent file in the checkpoint directory as a proxy
    local checkpoint_mtime=$(find "$checkpoint_path" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1)
    
    if [ -z "$checkpoint_mtime" ]; then
        # Fallback: use directory modification time
        checkpoint_mtime=$(stat -c %Y "$checkpoint_path" 2>/dev/null || stat -f %m "$checkpoint_path" 2>/dev/null || echo "0")
    fi
    
    # Convert to integer (remove decimal part)
    checkpoint_mtime=${checkpoint_mtime%.*}
    
    # Check if checkpoint was modified after training started (with 5 minute buffer for safety)
    local time_diff=$((checkpoint_mtime - TRAINING_START_TIME))
    if [ $time_diff -ge -300 ]; then  # Allow 5 minutes before start time (for safety)
        return 0  # Checkpoint is from current run
    else
        return 1  # Checkpoint is too old
    fi
}

# Check if training completed by looking for checkpoints
# (The launcher may exit with non-zero code even on successful completion)
CHECKPOINT_BASE="./outputs/grpo/checkpoints/root/${EXPERIMENT_NAME}/${TRIAL_NAME}/default"

echo ""
echo "Checking for checkpoints..."
echo "Experiment name: $EXPERIMENT_NAME"
echo "Checkpoint base path: $CHECKPOINT_BASE"
echo "Training exit code: $TRAIN_EXIT_CODE"
echo "Training started at: $TRAINING_START_TIME_READABLE"

if [ -d "$CHECKPOINT_BASE" ]; then
    echo "Checkpoint directory exists!"
    LATEST_CHECKPOINT=$(ls -td "$CHECKPOINT_BASE"/*/ 2>/dev/null | head -1)
    
    if [ -n "$LATEST_CHECKPOINT" ]; then
        LATEST_CHECKPOINT=$(echo "$LATEST_CHECKPOINT" | sed 's|/$||')  # Remove trailing slash
        
        # Verify checkpoint is from current run
        if checkpoint_is_from_current_run "$LATEST_CHECKPOINT"; then
            CHECKPOINT_MTIME_READABLE=$(stat -c %y "$LATEST_CHECKPOINT" 2>/dev/null | cut -d'.' -f1 || echo "unknown")
            echo ""
            echo "=========================================="
            echo "Training completed. Running validation..."
            echo "=========================================="
            echo "Found latest checkpoint: $LATEST_CHECKPOINT"
            echo "Checkpoint modified at: $CHECKPOINT_MTIME_READABLE"
            echo "✓ Checkpoint verified as from current training run"
        
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
            CHECKPOINT_MTIME_READABLE=$(stat -c %y "$LATEST_CHECKPOINT" 2>/dev/null | cut -d'.' -f1 || echo "unknown")
            echo ""
            echo "⚠️  WARNING: Latest checkpoint is from a previous training run!"
            echo "Checkpoint: $LATEST_CHECKPOINT"
            echo "Checkpoint modified at: $CHECKPOINT_MTIME_READABLE"
            echo "Training started at: $TRAINING_START_TIME_READABLE"
            echo ""
            echo "Skipping tests to avoid testing old checkpoints."
            echo "If training completed successfully, checkpoints may not have been saved yet."
        fi
    else
        echo "WARNING: No checkpoint subdirectories found in $CHECKPOINT_BASE"
        echo "Training may have failed or not produced checkpoints."
    fi
else
    echo "WARNING: Checkpoint directory not found: $CHECKPOINT_BASE"
    echo "Training may have failed or not produced checkpoints."
fi

# Exit with training exit code (even if tests ran)
exit $TRAIN_EXIT_CODE
