# Quick Start: Recovery from Step 188 Crash

## Current Situation

- Training crashed at step 188 (SGLang server disconnect)
- Task reward dropped to zero and stayed zero for ~500 steps
- Need to resume from a checkpoint before step 188

## Step 1: List Available Checkpoints

On your RunPod, run:

```bash
cd /workspace/AReaL
python examples/cloud_gsm8k/list_checkpoints.py \
    --experiment-name gsm8k-grpo-cloud-h200 \
    --trial-name trial_20251112_203112 \
    --before-step 188
```

This will show you all checkpoints saved before step 188.

## Step 2: Set Up Recovery

Once you know which checkpoint to use, set it up for recovery:

```bash
python examples/cloud_gsm8k/setup_recovery.py \
    --experiment-name gsm8k-grpo-cloud-h200 \
    --trial-name trial_20251112_203112 \
    --before-step 188
```

This will:
- Find the latest checkpoint before step 188
- Copy it to the `recover_checkpoint` directory
- Prepare it for automatic recovery

## Step 3: Resume Training

After setting up recovery, resume training:

```bash
python -m areal.launcher.local examples/cloud_gsm8k/gsm8k_grpo_cloud.py \
    --config examples/cloud_gsm8k/gsm8k_grpo_cloud.yaml \
    experiment_name=gsm8k-grpo-cloud-h200 \
    trial_name=trial_20251112_203112 \
    recover.mode=auto
```

## What Changed

### Circuit Breaker Added

The training script now includes a **circuit breaker** that will:
- Monitor `task_reward/avg` after each step
- Stop training if reward is zero for 10 consecutive steps
- Save a checkpoint before stopping
- Provide clear error message with recovery instructions

This prevents the model from being corrupted by training on invalid data.

### Checkpoint Strategy

- **Saves after each epoch** (`freq_epochs: 1`)
- With 935 steps per epoch, checkpoints are ~935 steps apart
- Recovery info is saved every hour (`freq_secs: 3600`)

## Troubleshooting

### If No Checkpoints Found

Checkpoints are saved to:
```
/workspace/outputs/grpo/checkpoints/root/gsm8k-grpo-cloud-h200/trial_20251112_203112/default/
```

Verify the path exists:
```bash
ls -lh /workspace/outputs/grpo/checkpoints/root/gsm8k-grpo-cloud-h200/trial_20251112_203112/default/
```

### If Recovery Fails

1. Check that `recover_checkpoint` directory has files:
   ```bash
   ls -lh /workspace/outputs/grpo/checkpoints/root/gsm8k-grpo-cloud-h200/trial_20251112_203112/recover_checkpoint/
   ```

2. Verify checkpoint integrity:
   ```bash
   # Should have model files
   ls -lh /workspace/outputs/grpo/checkpoints/root/gsm8k-grpo-cloud-h200/trial_20251112_203112/recover_checkpoint/*.safetensors
   ```

3. Check recovery info:
   ```bash
   cat /workspace/outputs/grpo/checkpoints/root/gsm8k-grpo-cloud-h200/trial_20251112_203112/recover_checkpoint/step_info.json
   ```

## Next Steps After Recovery

1. **Monitor WandB** to ensure task reward recovers
2. **Check SGLang server** is running and healthy
3. **Watch for circuit breaker** - if it triggers again, investigate SGLang server issues

