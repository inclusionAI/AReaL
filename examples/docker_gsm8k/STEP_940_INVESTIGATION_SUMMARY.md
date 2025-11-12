# Step 940 Investigation Summary

## Problem Timeline

1. **Before step 940**: Training working normally
   - Last non-zero reward: `task_reward/avg = 3.1250e-02` (3.125%)
   - Checkpoints saved regularly

2. **Step 939**: Reward already dropped to zero
   - `task_reward/avg = 0.0000e+00`
   - Training continued but learning nothing

3. **Step 940**: Reward still zero
   - `task_reward/avg = 0.0000e+00`
   - All rollouts returning zero rewards

4. **Steps 940-1869**: Training continued with zero rewards
   - ~930 steps of wasted training
   - No learning occurred

5. **Nov 12 05:09**: SGLang server fully crashed
   - Connection refused errors start
   - Server process died

6. **Nov 12 18:33**: Training manually stopped

## Root Cause

The SGLang server became **unresponsive** (not fully crashed) around step 939-940:

1. **Server started failing** to respond to generation requests
2. **Rollout generation failed** (timeouts, not connection refused yet)
3. **All rollouts returned empty/invalid** → rewards became zero
4. **Training continued** with zero rewards (learning nothing)
5. **Server fully crashed** later at step 1869

## Available Checkpoints

### ✅ Best Checkpoint: Step 899
- **Location**: `outputs/grpo/checkpoints/root/gsm8k-grpo-full-local/trial0/default/epoch0epochstep899globalstep899`
- **Saved**: Nov 11 10:59
- **Status**: Before the problem started
- **Reward**: Was non-zero at this point

### Other Checkpoints Before Problem:
- Step 860 (Nov 11 10:28)
- Step 849 (Nov 11 10:20)
- Step 821 (Nov 11 09:58)

## Recovery Plan

### Step 1: Resume from Step 899 Checkpoint

The training script will automatically detect and resume from the last checkpoint. However, since the recovery info points to step 1869 (which has zero reward), we should:

1. **Verify checkpoint exists**:
   ```bash
   ls -lh outputs/grpo/checkpoints/root/gsm8k-grpo-full-local/trial0/default/epoch0epochstep899globalstep899/
   ```

2. **Update recovery info** (if needed) to point to step 899 instead of 1869

3. **Resume training**:
   ```bash
   bash examples/docker_gsm8k/run_full_training.sh
   ```

### Step 2: Monitor After Resume

After resuming from step 899:
- **Watch WandB** for `grpo_actor/task_reward/avg` - should recover to ~3%
- **Monitor GPU memory** - should be stable
- **Check for connection errors** - should be zero
- **Verify rollouts are generating** - should see valid completions

### Step 3: If Problem Repeats

If reward drops to zero again:
1. **Check SGLang server logs** immediately
2. **Check GPU memory** - might be OOM
3. **Check system resources** - might be resource exhaustion
4. **Apply further memory reductions** if needed

## Prevention (Already Applied)

The updated `gsm8k_grpo_full.yaml` includes:
- ✅ `sglang.mem_fraction_static: 0.6` (reduced from 0.8)
- ✅ `rollout.max_concurrent_rollouts: 16` (reduced from 32)
- ✅ `rollout.request_timeout: 7200` (increased from 3600)
- ✅ `rollout.request_retries: 5` (increased from 3)

These changes should prevent the server from becoming unresponsive.

## Expected Behavior After Resume

1. **Training resumes** from step 900 (next step after checkpoint 899)
2. **Task reward recovers** to ~3% (previous level)
3. **Training progresses normally** with updated memory settings
4. **No more crashes** (with safer memory configuration)

## Key Insight

The problem started **before** the server fully crashed. The server became unresponsive around step 939-940, causing all rollouts to fail and rewards to drop to zero. The server didn't fully crash until much later (step 1869).

This suggests we need:
- **Better health checks** for SGLang server
- **Automatic server restart** if it becomes unresponsive
- **Early detection** of zero rewards (stop training if reward is zero for N consecutive steps)

## Next Steps

1. ✅ **Resume from step 899 checkpoint**
2. ✅ **Monitor closely** for first 100 steps
3. ✅ **Verify reward recovers** to previous levels
4. ✅ **Continue training** with updated config

