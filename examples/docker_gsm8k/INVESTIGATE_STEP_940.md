# Investigation: Task Reward Dropped to Zero at Step 940

## Key Finding

**At step 940, `task_reward/avg` is already `0.0000e+00`**

From the logs:
```
20251111-11:37:11.251 StatsLogger INFO: Epoch 2/5 Step 5/935 Train step 940/4675 done.
...
│ grpo_actor/task_reward/avg      │ 0.0000e+00 │
│ grpo_actor/task_reward/min      │ 0.0000e+00 │
│ grpo_actor/task_reward/max      │ 0.0000e+00 │
│ rollout/reward                  │ 0.0000e+00 │
```

## Timeline

- **Step 940**: Task reward is **already zero**
- **Step 1869**: Last checkpoint saved (reward still zero)
- **Nov 12 05:09**: Connection errors start (SGLang server crashed)
- **Nov 12 18:33**: Training manually stopped

## What Happened

1. **Before step 940**: Training was working normally (reward > 0)
2. **At step 940**: Reward dropped to zero
3. **Steps 940-1869**: Training continued but learned nothing (all rewards zero)
4. **Step 1869**: Last checkpoint saved (still zero reward)
5. **Nov 12 05:09**: SGLang server crashed (connection refused errors)

## Root Cause Hypothesis

The reward dropped to zero **before** the SGLang server crashed. This suggests:

1. **SGLang server became unresponsive** around step 940 (but didn't fully crash)
2. **Rollout generation started failing** (connection timeouts, not connection refused)
3. **All rollouts returned empty/invalid** → rewards became zero
4. **Training continued** with zero rewards (learning nothing)
5. **Server fully crashed** later at step 1869 (Nov 12 05:09)

## Recovery Strategy

### Option 1: Resume from Checkpoint Before Step 940

**Best checkpoint**: Step 899 (before the problem started)

```bash
# Check checkpoint exists
ls -lh outputs/grpo/checkpoints/root/gsm8k-grpo-full-local/trial0/default/epoch0epochstep899globalstep899/

# Resume training (will auto-detect and use this checkpoint)
bash examples/docker_gsm8k/run_full_training.sh
```

### Option 2: Check WandB for Exact Step When Reward Dropped

1. Go to WandB dashboard
2. Project: `gsm8k-grpo-local`
3. Run: `gsm8k-grpo-full-local_trial0_train`
4. Look at `grpo_actor/task_reward/avg` metric
5. Find the exact step where it dropped from non-zero to zero
6. Use checkpoint from just before that step

## Next Steps

1. **Check WandB** to find exact step when reward dropped
2. **Find checkpoint** from before that step
3. **Resume training** from that checkpoint
4. **Monitor closely** for first 50 steps after resume
5. **Apply memory fixes** (already done in config)

## Prevention

The updated config (`gsm8k_grpo_full.yaml`) includes:
- ✅ Lower memory usage (`mem_fraction_static: 0.6`)
- ✅ Fewer concurrent requests (`max_concurrent_rollouts: 16`)
- ✅ Longer timeouts (`request_timeout: 7200`)
- ✅ More retries (`request_retries: 5`)

These should prevent the server from becoming unresponsive.

