# Recovering from SGLang Server Crash

## Quick Recovery Steps

### Step 1: Diagnose the Issue

```bash
# Run health check script
bash examples/docker_gsm8k/check_sglang_health.sh
```

This will show:
- SGLang server status
- Connection errors
- GPU/system memory usage
- OOM kills
- Available checkpoints

### Step 2: Find Last Good Checkpoint

```bash
# List checkpoints (sorted by time)
ls -lt outputs/grpo/checkpoints/gsm8k-grpo-full-local/trial0/ | head -10

# Check recovery info
ls -lh outputs/grpo/recover/gsm8k-grpo-full-local/trial0/
```

**Important**: Find a checkpoint from **before step 940** (before the crash).

### Step 3: Apply Memory Fixes (If OOM)

If the crash was due to OOM, the config has been updated with:
- `sglang.mem_fraction_static: 0.6` (reduced from 0.8)
- `rollout.max_concurrent_rollouts: 16` (reduced from 32)
- `rollout.request_timeout: 7200` (increased for reliability)

The updated config is in `gsm8k_grpo_full.yaml`.

### Step 4: Resume Training

```bash
# Simply run the training script again
# It will automatically resume from the last checkpoint
bash examples/docker_gsm8k/run_full_training.sh
```

**The training will:**
- Automatically detect the last checkpoint
- Resume from the exact step where it stopped
- Continue training with the updated (safer) memory settings

## What Happened

Around **step 940**:
1. SGLang inference server crashed (likely OOM)
2. Server stopped responding (`ConnectionRefusedError`)
3. All rollout generation requests failed
4. Task reward dropped to zero (no valid generations)
5. Training continued but learned nothing (garbage data)

## Prevention

The updated `gsm8k_grpo_full.yaml` now includes:
- **Lower memory usage**: `mem_fraction_static: 0.6` (was 0.8)
- **Fewer concurrent requests**: `max_concurrent_rollouts: 16` (was 32)
- **Longer timeouts**: `request_timeout: 7200` (was 3600)
- **More retries**: `request_retries: 5` (was 3)

These changes should prevent future crashes during long training runs.

## Monitoring

After resuming, monitor:
1. **WandB dashboard**: Watch `grpo_actor/task_reward/avg` - should stay above zero
2. **GPU memory**: Should be stable (not increasing)
3. **Connection errors**: Should be zero
4. **Training logs**: No crash/error messages

## If Crash Repeats

If the server crashes again:
1. **Further reduce memory**: `mem_fraction_static: 0.5` or `0.4`
2. **Reduce generation length**: `max_new_tokens: 256` (from 512)
3. **Reduce samples**: `n_samples: 2` (from 4)
4. **Enable gradient checkpointing**: `gradient_checkpointing: true`

## Expected Recovery

After resuming:
- Training should continue from last checkpoint
- Task reward should recover (back to ~20-30% initially)
- Training should progress normally
- No more crashes (with updated memory settings)

## Summary

1. ✅ **Diagnose**: Run `check_sglang_health.sh`
2. ✅ **Resume**: Run `run_full_training.sh` (auto-detects checkpoint)
3. ✅ **Monitor**: Watch WandB for reward recovery
4. ✅ **Prevent**: Updated config has safer memory settings

The training will automatically resume from the last checkpoint before the crash!

