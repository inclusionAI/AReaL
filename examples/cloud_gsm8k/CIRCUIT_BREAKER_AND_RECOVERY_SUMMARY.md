# Circuit Breaker and Recovery Implementation Summary

## What Was Implemented

### 1. Circuit Breaker in Training Script

**File**: `examples/cloud_gsm8k/gsm8k_grpo_cloud.py`

**Features**:
- Monitors `grpo_actor/task_reward/avg` after each training step
- Tracks consecutive zero-reward steps
- **Stops training** if reward is zero for **10 consecutive steps**
- Saves a checkpoint before stopping
- Provides detailed error message with recovery instructions

**Configuration**:
```python
CIRCUIT_BREAKER_THRESHOLD = 10  # Stop after 10 consecutive zero-reward steps
CIRCUIT_BREAKER_ENABLED = True
```

**Benefits**:
- Prevents model corruption from training on invalid data
- Stops training immediately when SGLang server fails
- Saves checkpoint before stopping for easy recovery

### 2. Checkpoint Listing Tool

**File**: `examples/cloud_gsm8k/list_checkpoints.py`

**Features**:
- Lists all available checkpoints for an experiment/trial
- Can filter by maximum global step (find checkpoints before a crash)
- Shows checkpoint details (epoch, step, global_step, path)
- Verifies checkpoint completeness (checks for model files)

**Usage**:
```bash
# List all checkpoints
python examples/cloud_gsm8k/list_checkpoints.py \
    --experiment-name gsm8k-grpo-cloud-h200 \
    --trial-name trial_20251112_203112

# List checkpoints before step 188
python examples/cloud_gsm8k/list_checkpoints.py \
    --experiment-name gsm8k-grpo-cloud-h200 \
    --trial-name trial_20251112_203112 \
    --max-step 188
```

### 3. Recovery Setup Tool

**File**: `examples/cloud_gsm8k/setup_recovery.py`

**Features**:
- Finds the latest checkpoint before a specified step
- Copies checkpoint to `recover_checkpoint` directory
- Prepares checkpoint for automatic recovery
- Supports dry-run mode

**Usage**:
```bash
# Automatically find and set up latest checkpoint before step 188
python examples/cloud_gsm8k/setup_recovery.py \
    --experiment-name gsm8k-grpo-cloud-h200 \
    --trial-name trial_20251112_203112 \
    --before-step 188
```

### 4. Recovery Guide Tool

**File**: `examples/cloud_gsm8k/resume_training.py`

**Features**:
- Interactive tool to find and resume from checkpoints
- Shows available checkpoints
- Provides exact commands to resume training
- Checks for recover checkpoint info

**Usage**:
```bash
python examples/cloud_gsm8k/resume_training.py \
    --experiment-name gsm8k-grpo-cloud-h200 \
    --trial-name trial_20251112_203112 \
    --before-step 188
```

## Current Checkpoint Strategy

### Checkpoint Saving

From `gsm8k_grpo_cloud.yaml`:
```yaml
saver:
  freq_epochs: 1  # Save after each epoch
  freq_steps: null
  freq_secs: null
```

**Checkpoint Frequency**:
- Saves after each epoch completion
- With 935 steps per epoch, checkpoints are ~935 steps apart
- Checkpoint path: `epoch{epoch}epochstep{step}globalstep{global_step}/`

### Recovery Info Saving

```yaml
recover:
  mode: disabled  # Currently disabled (set to 'auto' to enable)
  freq_epochs: 1
  freq_steps: null
  freq_secs: 3600  # Save recovery info every hour
```

**Recovery Info**:
- Saved to `recover_checkpoint/` directory
- Contains step info, dataloader state, optimizer state, etc.
- Updated every hour during training

## Recovery Workflow for H200

### Step 1: List Checkpoints

```bash
cd /workspace/AReaL
python examples/cloud_gsm8k/list_checkpoints.py \
    --experiment-name gsm8k-grpo-cloud-h200 \
    --trial-name trial_20251112_203112 \
    --before-step 188 \
    --recover-info
```

### Step 2: Set Up Recovery

```bash
python examples/cloud_gsm8k/setup_recovery.py \
    --experiment-name gsm8k-grpo-cloud-h200 \
    --trial-name trial_20251112_203112 \
    --before-step 188
```

This will:
1. Find the latest checkpoint with `global_step < 188`
2. Copy it to `recover_checkpoint/` directory
3. Prepare it for automatic recovery

### Step 3: Resume Training

```bash
python -m areal.launcher.local examples/cloud_gsm8k/gsm8k_grpo_cloud.py \
    --config examples/cloud_gsm8k/gsm8k_grpo_cloud.yaml \
    experiment_name=gsm8k-grpo-cloud-h200 \
    trial_name=trial_20251112_203112 \
    recover.mode=auto
```

## How Recovery Works

When `recover.mode=auto` is set:

1. **AReaL checks** for `recover_checkpoint/` directory
2. **Loads recovery info** from `step_info.json` and other metadata files
3. **Restores model weights** from the checkpoint
4. **Restores optimizer state** (if saved)
5. **Restores dataloader state** (to continue from same data position)
6. **Resumes training** from `global_step + 1`

## Important Notes

1. **Checkpoint Frequency**: Checkpoints are saved after each epoch (~935 steps). If you need more frequent checkpoints, you can:
   - Set `saver.freq_steps: 100` to save every 100 steps
   - Or `saver.freq_secs: 3600` to save every hour

2. **Recovery Mode**: Currently set to `disabled` in config. To enable automatic recovery:
   - Set `recover.mode: auto` in the config
   - Or override via CLI: `recover.mode=auto`

3. **Circuit Breaker**: The circuit breaker will now prevent the issue you experienced (500 steps of zero reward). It stops after 10 consecutive zero-reward steps.

4. **Trial Name**: Make sure to use the correct trial name. Check your logs or WandB to find the exact trial name used.

## Files Created

1. **`list_checkpoints.py`** - List and filter checkpoints
2. **`setup_recovery.py`** - Set up recovery from a checkpoint
3. **`resume_training.py`** - Interactive recovery guide
4. **`CHECKPOINT_AND_RECOVERY_GUIDE.md`** - Detailed documentation
5. **`RECOVERY_QUICK_START.md`** - Quick reference guide
6. **`CIRCUIT_BREAKER_AND_RECOVERY_SUMMARY.md`** - This file

## Modified Files

1. **`gsm8k_grpo_cloud.py`** - Added circuit breaker logic

## Next Steps

1. **On RunPod**, run `list_checkpoints.py` to see what checkpoints are available
2. **Use `setup_recovery.py`** to prepare recovery from before step 188
3. **Resume training** with `recover.mode=auto`
4. **Monitor WandB** to ensure task reward recovers properly

