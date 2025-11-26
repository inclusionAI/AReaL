#!/bin/bash
# Run GRPO training in Docker container
# Usage: bash examples/docker_gsm8k/run_training.sh [config_name|config_file]
#
# Config presets:
#   - fastest: Fastest training (~5-10 min, 20 samples, 1 epoch) - Pipeline testing only
#   - fast: Fast training (20-30 min, 200 samples, 1 epoch)
#   - 1hour: 1-hour training (500 samples, 2 epochs)
#   - 3hour: 3-hour training (1000 samples, 3 epochs)
#   - full: Full training (all samples, 5 epochs)
#   - reasoning_fastest: Reasoning model fastest training (~5-10 min, 20 samples, 1 epoch) - Pipeline testing only
#   - reasoning_fast: Reasoning model fast training (20-30 min, 200 samples, 1 epoch)
#   - reasoning_1hour: Reasoning model 1-hour training (500 samples, 2 epochs)
#   - reasoning_3hour: Reasoning model 3-hour training (1000 samples, 3 epochs) [NEW]
#   - reasoning_5hour: Reasoning model 5-hour training (2000 samples, 3 epochs)
#
# Or provide full path to any config file:
#   Example: bash examples/docker_gsm8k/run_training.sh examples/docker_gsm8k/gsm8k_grpo_reasoning_fast.yaml

set -e

cd /workspace/AReaL

# Configuration
CONFIG_NAME="${1:-fast}"
TRIAL_NAME="trial0"

# Check if CONFIG_NAME is a preset or a file path
if [[ "$CONFIG_NAME" == *".yaml" ]] || [[ "$CONFIG_NAME" == *".yml" ]]; then
    # It's a file path - use it directly
    CONFIG_FILE="$CONFIG_NAME"
    # Extract experiment name from filename
    EXPERIMENT_NAME=$(basename "$CONFIG_FILE" .yaml | sed 's/_/-/')
    EXPERIMENT_NAME="${EXPERIMENT_NAME}-docker"
else
    # It's a preset name - map to config file
    case "$CONFIG_NAME" in
        fastest)
            CONFIG_FILE="examples/docker_gsm8k/gsm8k_grpo_fastest.yaml"
            EXPERIMENT_NAME="gsm8k-grpo-fastest-docker"
            echo "Using FASTEST training configuration (~5-10 minutes)"
            echo "Note: Minimal training for pipeline testing only (20 samples, 1 epoch)"
            ;;
        fast)
            CONFIG_FILE="examples/docker_gsm8k/gsm8k_grpo_fast.yaml"
            EXPERIMENT_NAME="gsm8k-grpo-fast-docker"
            echo "Using FAST training configuration (20-30 minutes)"
            ;;
        1hour)
            CONFIG_FILE="examples/docker_gsm8k/gsm8k_grpo_1hour.yaml"
            EXPERIMENT_NAME="gsm8k-grpo-1hour-docker"
            echo "Using 1-HOUR training configuration (~1-2 hours)"
            echo "Note: Uses limited dataset (500 samples)"
            ;;
        3hour)
            CONFIG_FILE="examples/docker_gsm8k/gsm8k_grpo_3hour.yaml"
            EXPERIMENT_NAME="gsm8k-grpo-3hour-docker"
            echo "Using 3-HOUR training configuration (~3-4 hours)"
            echo "Note: Uses limited dataset (1000 samples)"
            ;;
        full)
            CONFIG_FILE="examples/docker_gsm8k/gsm8k_grpo_full.yaml"
            EXPERIMENT_NAME="gsm8k-grpo-full-docker"
            echo "Using FULL training configuration (full dataset, 5 epochs)"
            ;;
        reasoning_fastest)
            CONFIG_FILE="examples/docker_gsm8k/gsm8k_grpo_reasoning_fastest.yaml"
            EXPERIMENT_NAME="gsm8k-grpo-reasoning-fastest-docker"
            echo "Using REASONING FASTEST training configuration (~5-10 minutes)"
            echo "Note: Minimal training for pipeline testing only (20 samples, 1 epoch)"
            echo "Note: Trains reasoning model with XML format"
            ;;
        reasoning_fast)
            CONFIG_FILE="examples/docker_gsm8k/gsm8k_grpo_reasoning_fast.yaml"
            EXPERIMENT_NAME="gsm8k-grpo-reasoning-fast-docker"
            echo "Using REASONING FAST training configuration (20-30 minutes)"
            echo "Note: Trains reasoning model with XML format"
            ;;
        reasoning_1hour)
            CONFIG_FILE="examples/docker_gsm8k/gsm8k_grpo_reasoning_1hour.yaml"
            EXPERIMENT_NAME="gsm8k-grpo-reasoning-1hour-docker"
            echo "Using REASONING 1-HOUR training configuration (~1-2 hours)"
            echo "Note: Trains reasoning model with XML format (500 samples)"
            ;;
        reasoning_3hour)
            CONFIG_FILE="examples/docker_gsm8k/gsm8k_grpo_reasoning_3hour.yaml"
            EXPERIMENT_NAME="gsm8k-grpo-reasoning-3hour-docker"
            echo "Using REASONING 3-HOUR training configuration (~3-4 hours)"
            echo "Note: Trains reasoning model with XML format (1000 samples, 3 epochs)"
            echo "Optimized for RTX 4080S (16GB VRAM)"
            ;;
        reasoning_5hour)
            CONFIG_FILE="examples/docker_gsm8k/gsm8k_grpo_reasoning_5hour.yaml"
            EXPERIMENT_NAME="gsm8k-grpo-reasoning-5hour-docker"
            echo "Using REASONING 5-HOUR training configuration (~5-6 hours)"
            echo "Note: Trains reasoning model with XML format (2000 samples, 3 epochs)"
            ;;
        *)
            echo "ERROR: Unknown config preset: $CONFIG_NAME"
            echo "Valid presets: fastest, fast, 1hour, 3hour, full, reasoning_fastest, reasoning_fast, reasoning_1hour, reasoning_3hour, reasoning_5hour"
            echo "Or provide full path to a config file (e.g., examples/docker_gsm8k/gsm8k_grpo_fast.yaml)"
            exit 1
            ;;
    esac
fi

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
