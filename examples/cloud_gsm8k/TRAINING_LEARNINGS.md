# Training Learnings and Best Practices

This document consolidates all key learnings, fixes, and best practices from training GRPO on cloud GPUs (RunPod).

## Table of Contents

1. [Spot Instances](#spot-instances)
2. [Checkpoint and Recovery](#checkpoint-and-recovery)
3. [Circuit Breaker](#circuit-breaker)
4. [GPU Configuration](#gpu-configuration)
5. [Common Issues and Fixes](#common-issues-and-fixes)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)

## Spot Instances

### Understanding Spot Instances

**Spot instances** on RunPod are 50-70% cheaper but can be **interrupted** at any time when:
- Someone else claims the GPU
- Provider needs the GPU for a regular customer
- Spot price changes

### What Happens During Interruption

- ✅ Training stops immediately (no graceful shutdown)
- ✅ Checkpoints are saved (if using network volume)
- ⚠️ WandB shows "running" (no clean shutdown signal)
- ⚠️ No error messages (external interruption, not training error)

### Spot vs Regular Instances

| Type | Cost | Stability | Best For |
|------|------|-----------|----------|
| **Spot** | 50-70% cheaper | Can be interrupted | Experiments, testing, short runs |
| **Regular** | Full price | Guaranteed | Long training runs, production |

**Example**: RTX 4090 spot ($0.09/hour) vs regular ($0.29/hour) = 69% savings

### Best Practices for Spot Instances

1. **Always use network volumes** - Mount to `/workspace/outputs` for checkpoint persistence
2. **Frequent checkpoints** - Ensure config saves checkpoints regularly
3. **Monitor training** - Check WandB regularly to detect interruptions
4. **Set up recovery** - Know how to resume from checkpoints
5. **Use circuit breaker** - Training script includes automatic error detection
6. **Test with short runs** - Use `fast` or `1hour` configs first

### Detecting Spot Interruptions

**Signs**:
- WandB run shows "running" but hasn't updated in hours
- RunPod pod status shows "Stopped" or "Failed"
- Training was healthy (no errors) before stopping
- Last step was mid-training (not at epoch end)

**Verify**:
```bash
# Check WandB run status
python examples/cloud_gsm8k/check_wandb_run.py --trial-name trial_YYYYMMDD_HHMMSS
```

## Checkpoint and Recovery

### Checkpoint Strategy

**Current Configuration**:
```yaml
saver:
  freq_epochs: 1  # Save after each epoch
  freq_steps: null
  freq_secs: null

recover:
  mode: disabled  # Set to 'auto' to enable
  freq_epochs: 1
  freq_steps: null
  freq_secs: 3600  # Save recovery info every hour
```

**Checkpoint Location**:
```
/workspace/outputs/grpo/checkpoints/{user}/{experiment_name}/{trial_name}/default/epoch{epoch}epochstep{step}globalstep{global_step}/
```

**Recovery Info Location**:
```
/workspace/outputs/grpo/checkpoints/{user}/{experiment_name}/{trial_name}/recover_checkpoint/
```

### Recovery Tools

#### 1. List Checkpoints

```bash
# List all checkpoints
python examples/cloud_gsm8k/list_checkpoints.py \
    --experiment-name gsm8k-grpo-cloud-1hour \
    --trial-name trial_YYYYMMDD_HHMMSS

# List checkpoints before a specific step
python examples/cloud_gsm8k/list_checkpoints.py \
    --experiment-name gsm8k-grpo-cloud-1hour \
    --trial-name trial_YYYYMMDD_HHMMSS \
    --before-step 188
```

#### 2. Set Up Recovery

```bash
# Automatically find and set up latest checkpoint
python examples/cloud_gsm8k/setup_recovery.py \
    --experiment-name gsm8k-grpo-cloud-1hour \
    --trial-name trial_YYYYMMDD_HHMMSS \
    --before-step 188
```

#### 3. Resume Training

```bash
# After setting up recovery, resume with:
python -m areal.launcher.local examples/cloud_gsm8k/gsm8k_grpo_train.py \
    --config examples/cloud_gsm8k/gsm8k_grpo_1hour.yaml \
    experiment_name=gsm8k-grpo-cloud-1hour \
    trial_name=trial_YYYYMMDD_HHMMSS \
    recover.mode=auto
```

#### 4. Interactive Recovery Guide

```bash
python examples/cloud_gsm8k/resume_training.py \
    --experiment-name gsm8k-grpo-cloud-1hour \
    --trial-name trial_YYYYMMDD_HHMMSS \
    --before-step 188
```

### Recovery Workflow

1. **Detect Interruption**: Check WandB or RunPod dashboard
2. **Verify Checkpoints**: Use `list_checkpoints.py` to find saved checkpoints
3. **Set Up Recovery**: Use `setup_recovery.py` to prepare recovery
4. **Resume Training**: Start new pod and resume with `recover.mode=auto`
5. **Monitor**: Watch WandB to ensure training continues from correct step

### What Gets Recovered

When resuming with `recover.mode=auto`:
- Model weights from checkpoint
- Optimizer state (if saved)
- Dataloader state (continues from same data position)
- Training step counter (starts from `global_step + 1`)
- Stats logger state (for WandB continuity)

## Circuit Breaker

### Overview

The training script (`gsm8k_grpo_train.py`) includes a **circuit breaker** that prevents model corruption from training on invalid data.

### How It Works

- **Monitors** `grpo_actor/task_reward/avg` after each training step
- **Tracks** consecutive zero-reward steps
- **Stops training** if reward is zero for **10 consecutive steps**
- **Saves checkpoint** before stopping
- **Provides detailed error message** with recovery instructions

### Configuration

```python
CIRCUIT_BREAKER_THRESHOLD = 10  # Stop after 10 consecutive zero-reward steps
CIRCUIT_BREAKER_ENABLED = True
```

### Benefits

- Prevents model corruption from training on invalid data
- Stops training immediately when SGLang server fails
- Saves checkpoint before stopping for easy recovery
- Provides clear error message with recovery instructions

### Common Causes of Zero Rewards

- SGLang server crashed or disconnected
- Inference server not responding
- Network connectivity issues
- All rollouts failed

## GPU Configuration

### Memory-Optimized Configs

All time-based configs (`1hour`, `3hour`) use memory-optimized settings that work on all GPUs:
- **Batch size**: 4 (reduced from 8)
- **Gradient checkpointing**: Enabled
- **SGLang memory fraction**: 0.5 (reduced from 0.8)
- **Max new tokens**: 256 (reduced from 512)
- **Max tokens per microbatch**: 4096 (reduced from 5120)
- **Max concurrent rollouts**: 16 (reduced from 32)

### GPU Requirements

| Config | GPU Memory | Recommended GPUs |
|--------|------------|------------------|
| `fast` | Any | RTX 4090, RTX 5090, A40, etc. |
| `1hour` | Any | RTX 4090, RTX 5090, A40, etc. |
| `3hour` | Any | RTX 4090, RTX 5090, A40, etc. |
| `full` | 80GB+ | H200, H100, A100-80GB only |

### Attention Backend

- **RTX 5090 (SM 100+)**: Requires `flashinfer` backend
- **Other GPUs**: Work with `flashinfer` (universal compatibility)
- **H200 (SM 90)**: Works with default `fa3` or `flashinfer`

### H200 Full Training

H200 (141GB memory) is ideal for full dataset training:
- **No memory optimizations needed**
- **Gradient checkpointing disabled** (faster training)
- **Larger batch sizes** (better stability)
- **More concurrent rollouts** (higher throughput)
- **Full dataset** (7473 samples, 5 epochs)
- **Training time**: ~5 days

### Memory Optimization Details

**Why Memory Optimization is Needed**:
- Both SGLang inference server and PPO trainer share the same GPU
- Default configs allocate 80% of GPU memory to SGLang (e.g., 36GB on A40)
- This leaves insufficient memory for the trainer, causing OOM errors

**Memory Breakdown (Example: A40 GPU)**:

| Config | SGLang | Trainer | Total | Status |
|--------|--------|---------|-------|--------|
| Default | ~36GB (80%) | ~8GB | ~44GB | ❌ OOM |
| Optimized | ~22GB (50%) | ~6GB | ~28GB | ✅ Fits |

**Performance Impact**:
- **Training speed**: ~20-30% slower due to gradient checkpointing
- **Inference throughput**: ~30-40% lower due to reduced parallelism
- **Total time**: 3-hour config may take ~4-4.5 hours with optimizations

## Common Issues and Fixes

### Issue: Checkpoints Not Saving to Persistent Volume

**Problem**: Checkpoints saved to container disk (lost on pod restart) instead of network volume.

**Root Cause**: Config used relative path `./outputs/grpo` which resolved to `/workspace/AReaL/outputs/grpo` (container disk).

**Fix**: Changed all configs to use absolute path `/workspace/outputs/grpo` (mounted volume).

**Verification**:
```bash
# Check mounted volume
ls -lh /workspace/outputs/grpo/checkpoints/
```

### Issue: CUDA Out of Memory on A40/RTX 5090

**Problem**: OOM errors on GPUs with 32-48GB memory (A40, RTX 5090, RTX 4090).

**Solution**: Use memory-optimized configs (automatically selected by `run_training_cloud.sh`):
- Reduced SGLang memory fraction: 0.8 → 0.5
- Enabled gradient checkpointing
- Reduced batch size: 8 → 4
- Reduced max new tokens: 512 → 256
- Reduced concurrent rollouts: 32 → 16

**Additional Optimizations** (if still OOM):
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Issue: SGLang Server Disconnect (Step 188 Example)

**Problem**: Task reward dropped to zero and stayed zero for hundreds of steps, corrupting the model.

**Root Cause**: SGLang server crashed/disconnected, but training continued with failed rollouts.

**Solution**: Circuit breaker now stops training after 10 consecutive zero-reward steps, preventing model corruption.

**Prevention**:
- Circuit breaker monitors `task_reward/avg` and stops on sustained zero rewards
- Increased `rollout.request_timeout` to 7200 seconds
- Increased `rollout.request_retries` to 5
- Increased `rollout.setup_timeout` to 300 seconds

### Issue: RTX 5090 "Unknown config name" Error

**Problem**: Training failed with "Unknown config name: h200" error on RTX 5090.

**Root Cause**: Script tried to use H200-specific config on RTX 5090.

**Solution**: Consolidated configs - `full` config now validates GPU requirements and works on all high-memory GPUs.

### Issue: Spot Instance Interruption

**Problem**: Training stopped abruptly, WandB shows "running" but no updates.

**Root Cause**: Spot instance was preempted by another user or provider needs.

**Solution**: 
- Always use network volumes for checkpoint persistence
- Resume from checkpoint using recovery workflow
- Monitor WandB to detect interruptions early

## Troubleshooting

### Training Crashes

**Symptoms**:
- Pod status changes to "Failed" or "Stopped"
- Terminal output ends abruptly
- GPU usage drops to zero
- No clear error message

**Solutions**:
1. **Check logs**: Look for `CUDA out of memory` or SGLang server errors
2. **Reduce memory usage**:
   - Reduce `sglang.mem_fraction_static` to 0.5 or 0.7
   - Enable `actor.gradient_checkpointing: true`
   - Reduce `train_dataset.batch_size`
   - Reduce `gconfig.max_new_tokens`
3. **Increase resilience**:
   - Increase `rollout.request_timeout` to 7200
   - Increase `rollout.request_retries` to 5
4. **Check circuit breaker**: May have triggered due to zero rewards

### Checkpoints Not Saving

**Symptoms**:
- No `outputs/grpo/checkpoints` directory after restart
- Cannot resume training from previous state

**Solutions**:
1. **Create network volume**: Mount to `/workspace/outputs`
2. **Verify fileroot**: Ensure config points to `/workspace/outputs/grpo`
3. **Check permissions**: Ensure write access to volume

### WandB Shows "Running" But Training Stopped

**Symptoms**:
- WandB run shows "running" state
- No updates in hours
- Training was healthy before stopping

**Likely Cause**: Spot instance interruption or external termination

**Solutions**:
1. **Check RunPod dashboard**: Verify pod status
2. **Check for checkpoints**: Use `list_checkpoints.py`
3. **Resume from checkpoint**: Use recovery workflow

### No Checkpoints Found

**Solutions**:
1. **Verify path**: Check that fileroot is correct (`/workspace/outputs/grpo`)
2. **Check user**: Checkpoints saved under user who ran training (usually `root`)
3. **Verify experiment/trial names**: Must match exactly
4. **Check network volume**: Ensure volume is mounted correctly

### Recovery Fails

**Solutions**:
1. **Check recover_checkpoint directory**: Must have necessary files
2. **Verify checkpoint integrity**: Check that model files exist
3. **Check logs**: Look for errors in recovery process
4. **Verify step_info.json**: Must contain valid step information

## Best Practices

### Before Training

1. **Set up network volume** - Mount to `/workspace/outputs` for persistence
2. **Configure WandB API key** - Set as environment variable in RunPod
3. **Choose appropriate config** - Match GPU capabilities to config
4. **Test with short run** - Use `fast` config first to verify setup

### During Training

1. **Monitor WandB** - Check training progress regularly
2. **Watch for zero rewards** - Circuit breaker will stop if detected
3. **Check GPU memory** - Ensure not approaching limits
4. **Verify checkpoints** - Confirm checkpoints are being saved

### After Interruption

1. **Check WandB status** - Use `check_wandb_run.py` to diagnose
2. **List checkpoints** - Use `list_checkpoints.py` to find saved checkpoints
3. **Set up recovery** - Use `setup_recovery.py` to prepare recovery
4. **Resume training** - Use `recover.mode=auto` to continue

### Cost Optimization

1. **Use spot instances** - 50-70% savings for experiments
2. **Use regular instances** - For production/long runs
3. **Monitor usage** - Stop pods when not in use
4. **Choose right GPU** - Match GPU to training needs

### Configuration Tips

1. **Memory-constrained GPUs**: Use `1hour` or `3hour` configs (already optimized)
2. **High-memory GPUs**: Can use `full` config for complete training
3. **Adjust checkpoint frequency**: Set `saver.freq_steps` for more frequent saves
4. **Enable recovery**: Set `recover.mode=auto` for automatic recovery

## Recovery Tools

### 1. List Checkpoints

```bash
# List all checkpoints
python examples/cloud_gsm8k/list_checkpoints.py \
    --experiment-name gsm8k-grpo-cloud-1hour \
    --trial-name trial_YYYYMMDD_HHMMSS

# List checkpoints before a specific step
python examples/cloud_gsm8k/list_checkpoints.py \
    --experiment-name gsm8k-grpo-cloud-1hour \
    --trial-name trial_YYYYMMDD_HHMMSS \
    --before-step 188
```

### 2. Set Up Recovery

```bash
# Automatically find and set up latest checkpoint
python examples/cloud_gsm8k/setup_recovery.py \
    --experiment-name gsm8k-grpo-cloud-1hour \
    --trial-name trial_YYYYMMDD_HHMMSS \
    --before-step 188
```

### 3. Interactive Recovery Guide

```bash
# Get step-by-step recovery instructions
python examples/cloud_gsm8k/resume_training.py \
    --experiment-name gsm8k-grpo-cloud-1hour \
    --trial-name trial_YYYYMMDD_HHMMSS \
    --before-step 188
```

### 4. Check WandB Run Status

```bash
# Diagnose WandB run (check if interrupted, find last step, etc.)
python examples/cloud_gsm8k/check_wandb_run.py \
    --trial-name trial_YYYYMMDD_HHMMSS
```

## Summary

✅ **Always use network volumes** for checkpoint persistence
✅ **Monitor training** via WandB to detect issues early
✅ **Use circuit breaker** to prevent model corruption
✅ **Set up recovery** before long training runs
✅ **Choose appropriate config** for your GPU
✅ **Use spot instances** for cost savings (with recovery setup)
✅ **Resume seamlessly** from checkpoints after interruptions
✅ **Memory-optimized configs** work on all GPUs (no GPU-specific configs needed)

