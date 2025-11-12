# Checkpoint and Recovery Guide for H200 Training

## Current Checkpoint Strategy

Based on `gsm8k_grpo_cloud.yaml`:

```yaml
saver:
  freq_epochs: 1  # Save after each epoch
  freq_steps: null
  freq_secs: null

recover:
  mode: disabled  # Currently disabled
  freq_epochs: 1
  freq_steps: null
  freq_secs: 3600  # Save recovery info every hour
```

### Checkpoint Location

Checkpoints are saved to:
```
/workspace/outputs/grpo/checkpoints/{user}/{experiment_name}/{trial_name}/default/epoch{epoch}epochstep{step}globalstep{global_step}/
```

For H200 training:
```
/workspace/outputs/grpo/checkpoints/root/gsm8k-grpo-cloud-h200/trial_20251112_203112/default/epoch{epoch}epochstep{step}globalstep{global_step}/
```

### Recovery Info Location

Recovery metadata is saved to:
```
/workspace/outputs/grpo/checkpoints/{user}/{experiment_name}/{trial_name}/recover_checkpoint/
```

## Circuit Breaker

A circuit breaker has been added to `gsm8k_grpo_cloud.py` that:

- **Monitors** `grpo_actor/task_reward/avg` after each training step
- **Stops training** if task reward is zero for **10 consecutive steps**
- **Saves a checkpoint** before stopping
- **Provides clear error message** with recovery instructions

### Configuration

You can adjust the circuit breaker in the training script:

```python
CIRCUIT_BREAKER_THRESHOLD = 10  # Stop after 10 consecutive zero-reward steps
CIRCUIT_BREAKER_ENABLED = True
```

## Listing Available Checkpoints

Use `list_checkpoints.py` to see what checkpoints are available:

```bash
# List all checkpoints
python examples/cloud_gsm8k/list_checkpoints.py \
    --experiment-name gsm8k-grpo-cloud-h200 \
    --trial-name trial_20251112_203112

# List checkpoints before step 188 (before crash)
python examples/cloud_gsm8k/list_checkpoints.py \
    --experiment-name gsm8k-grpo-cloud-h200 \
    --trial-name trial_20251112_203112 \
    --max-step 188

# Show only latest checkpoint
python examples/cloud_gsm8k/list_checkpoints.py \
    --experiment-name gsm8k-grpo-cloud-h200 \
    --trial-name trial_20251112_203112 \
    --latest
```

## Resuming Training

### Step 1: Find Available Checkpoints

```bash
# On RunPod, check what checkpoints exist
python examples/cloud_gsm8k/list_checkpoints.py \
    --experiment-name gsm8k-grpo-cloud-h200 \
    --trial-name trial_20251112_203112 \
    --before-step 188
```

This will show you the latest checkpoint before step 188.

### Step 2: Set Up Recovery

Use `setup_recovery.py` to copy a checkpoint to the `recover_checkpoint` directory:

```bash
# Automatically find and set up latest checkpoint before step 188
python examples/cloud_gsm8k/setup_recovery.py \
    --experiment-name gsm8k-grpo-cloud-h200 \
    --trial-name trial_20251112_203112 \
    --before-step 188

# Or use a specific checkpoint path
python examples/cloud_gsm8k/setup_recovery.py \
    --experiment-name gsm8k-grpo-cloud-h200 \
    --trial-name trial_20251112_203112 \
    --checkpoint-path /workspace/outputs/grpo/checkpoints/root/gsm8k-grpo-cloud-h200/trial_20251112_203112/default/epoch0epochstep187globalstep187
```

### Step 3: Resume Training

After setting up recovery, resume training with:

```bash
python -m areal.launcher.local examples/cloud_gsm8k/gsm8k_grpo_cloud.py \
    --config examples/cloud_gsm8k/gsm8k_grpo_cloud.yaml \
    experiment_name=gsm8k-grpo-cloud-h200 \
    trial_name=trial_20251112_203112 \
    recover.mode=auto
```

### Alternative: Use `resume_training.py`

This script provides a guided recovery process:

```bash
python examples/cloud_gsm8k/resume_training.py \
    --experiment-name gsm8k-grpo-cloud-h200 \
    --trial-name trial_20251112_203112 \
    --before-step 188
```

It will:
1. Find the latest checkpoint before step 188
2. Show you the checkpoint details
3. Provide the exact command to resume training

## Complete Recovery Workflow for H200

### 1. Check Available Checkpoints

```bash
cd /workspace/AReaL
python examples/cloud_gsm8k/list_checkpoints.py \
    --experiment-name gsm8k-grpo-cloud-h200 \
    --trial-name trial_20251112_203112 \
    --before-step 188 \
    --recover-info
```

### 2. Set Up Recovery

```bash
python examples/cloud_gsm8k/setup_recovery.py \
    --experiment-name gsm8k-grpo-cloud-h200 \
    --trial-name trial_20251112_203112 \
    --before-step 188
```

### 3. Verify Recovery Setup

```bash
# Check that recover_checkpoint directory exists and has files
ls -lh /workspace/outputs/grpo/checkpoints/root/gsm8k-grpo-cloud-h200/trial_20251112_203112/recover_checkpoint/
```

### 4. Resume Training

```bash
python -m areal.launcher.local examples/cloud_gsm8k/gsm8k_grpo_cloud.py \
    --config examples/cloud_gsm8k/gsm8k_grpo_cloud.yaml \
    experiment_name=gsm8k-grpo-cloud-h200 \
    trial_name=trial_20251112_203112 \
    recover.mode=auto
```

## What Gets Recovered

When resuming with `recover.mode=auto`, AReaL will:

1. **Load model weights** from the checkpoint
2. **Restore optimizer state** (if saved)
3. **Restore dataloader state** (to continue from same data position)
4. **Restore training step counter** (starts from `global_step + 1`)
5. **Restore stats logger state** (for WandB continuity)

## Important Notes

1. **Checkpoint Frequency**: Currently saves after each epoch. For 935 steps per epoch, this means checkpoints are ~935 steps apart.

2. **Recovery Info**: Recovery metadata is saved every hour (`freq_secs: 3600`), so you may have recovery info even if no epoch checkpoint exists.

3. **Circuit Breaker**: The circuit breaker will now prevent training from continuing with zero rewards for too long, protecting your model.

4. **Trial Name**: Make sure to use the correct trial name. For H200, it's likely `trial_20251112_203112` based on the logs.

## Troubleshooting

### No Checkpoints Found

If `list_checkpoints.py` finds no checkpoints:

1. **Verify path**: Check that the fileroot is correct (`/workspace/outputs/grpo`)
2. **Check user**: The checkpoints are saved under the user who ran training (usually `root`)
3. **Verify experiment/trial names**: Make sure they match exactly

### Recovery Fails

If recovery fails:

1. **Check recover_checkpoint directory**: Make sure it has the necessary files
2. **Verify checkpoint integrity**: Check that model files exist in the checkpoint
3. **Check logs**: Look for errors in the recovery process

### Circuit Breaker Too Sensitive

If the circuit breaker triggers too early:

- Increase `CIRCUIT_BREAKER_THRESHOLD` in `gsm8k_grpo_cloud.py`
- Or disable it temporarily: `CIRCUIT_BREAKER_ENABLED = False`

