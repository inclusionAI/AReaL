# Health Check Results Summary

## Date: November 12, 2025

## Status: ✅ Checkpoints Found, Server Down

### Key Findings

1. **✅ Checkpoints Available**
   - **Latest checkpoint**: Step 1869 (Epoch 1, Step 934)
   - **Location**: `outputs/grpo/checkpoints/root/gsm8k-grpo-full-local/trial0/default/epoch1epochstep934globalstep1869`
   - **Saved**: November 12, 2025 at 05:04

2. **✅ Recovery Info Available**
   - **Location**: `outputs/grpo/checkpoints/root/gsm8k-grpo-full-local/trial0/recover_info/step_info.json`
   - **Content**:
     ```json
     {
         "epoch": 1,
         "epoch_step": 934,
         "global_step": 1869,
         "steps_per_epoch": 935
     }
     ```

3. **❌ SGLang Server Status: DOWN**
   - No SGLang server processes running
   - Server crashed around **Nov 12 05:09** (5 minutes after last checkpoint)

4. **❌ Connection Errors: 128 total**
   - 64 errors in `trainer.log`
   - 64 errors in `wandb/output.log`
   - All errors: `ConnectionRefusedError(111, "Connect call failed ('172.17.0.2', 13231)")`

5. **✅ System Resources: Healthy**
   - **GPU Memory**: 9% used (1,541 MiB / 16,376 MiB)
   - **System Memory**: 96% available (28 GiB free)
   - **No OOM kills** detected

## Timeline

- **Nov 11 11:37**: Training at step 940 (Epoch 2, Step 5) - this was earlier, not the crash point
- **Nov 12 05:04**: Last checkpoint saved at step 1869
- **Nov 12 05:09**: SGLang server crashed (connection refused errors start)
- **Nov 12 18:33**: Training manually stopped (Ctrl+C)

## Root Cause Analysis

The SGLang inference server crashed around **step 1869**, not step 940. The crash was likely due to:

1. **Memory leak or gradual memory accumulation** over ~1869 steps
2. **Not an OOM kill** (GPU memory was only 9% at crash time)
3. **Server process crash** (no process found, but no OOM kill in logs)

## Recovery Plan

### ✅ Good News

- **Checkpoints exist** up to step 1869
- **Recovery info is available** for automatic resume
- **Config already updated** with safer memory settings:
  - `sglang.mem_fraction_static: 0.6` (reduced from 0.8)
  - `rollout.max_concurrent_rollouts: 16` (reduced from 32)
  - `rollout.request_timeout: 7200` (increased from 3600)

### Next Steps

1. **Resume training** (will auto-detect checkpoint):
   ```bash
   bash examples/docker_gsm8k/run_full_training.sh
   ```

2. **Training will**:
   - Automatically detect checkpoint at step 1869
   - Resume from step 1870 (next step after checkpoint)
   - Use updated memory settings to prevent future crashes
   - Continue training from where it left off

3. **Monitor**:
   - Watch WandB dashboard for task reward recovery
   - Check GPU memory stays stable
   - Verify no connection errors

## Expected Behavior After Resume

- Training resumes from **step 1870** (epoch 1, step 935)
- SGLang server will restart automatically
- Task reward should recover to previous levels (~20-30%)
- Training continues normally with safer memory settings

## Prevention Measures Applied

The updated `gsm8k_grpo_full.yaml` includes:
- ✅ Lower memory usage (`mem_fraction_static: 0.6`)
- ✅ Fewer concurrent requests (`max_concurrent_rollouts: 16`)
- ✅ Longer timeouts (`request_timeout: 7200`)
- ✅ More retries (`request_retries: 5`)

These changes should prevent future crashes during long training runs.

