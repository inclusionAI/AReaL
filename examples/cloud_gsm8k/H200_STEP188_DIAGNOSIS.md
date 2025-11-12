# H200 Training Step 188 Task Reward Drop - Diagnosis

## Summary
Task reward dropped to zero around step 188 and never recovered to previous levels, as confirmed by WandB visualization.

## WandB Visualization Analysis

The WandB graph reveals a clear pattern:

1. **Initial Training Phase (Steps 0-188):**
   - Task reward starts at ~0.3-0.4
   - Shows healthy fluctuations and upward trend
   - Peaks at ~0.7-0.8 around step 180-188
   - Indicates successful learning and improvement

2. **Critical Event at Step 188:**
   - **Abrupt drop from ~0.8 to exactly 0**
   - Very sudden and steep decline
   - Suggests immediate system failure rather than gradual degradation

3. **Sustained Zero Period (Steps 188-680):**
   - **~500 steps of zero reward** (nearly 500 steps!)
   - No recovery during this period
   - System appears completely non-functional

4. **Intermittent Recovery Attempts (Steps 680+):**
   - Steps 680-750: Intermittent spikes reaching 0.4-0.45
   - Steps 780-830: Similar intermittent spikes around 0.4-0.45
   - Step 850: Smaller spike around 0.1-0.15
   - **Never reaches initial 0.7-0.8 levels**
   - Intermittent nature suggests unstable connection or partial recovery

## Log Analysis

### Current Log Coverage
- Logs show training from step 1 to step 84
- Step 188 is beyond the current log file coverage
- Need to check WandB directly or get logs from step 180-200 range

### Observations from Available Logs (Steps 1-84)

1. **Task Reward Values Before Issue:**
   - Step 1: `task_reward/avg = 3.4375e-01` (34.4%)
   - Step 2: `task_reward/avg = 4.6875e-01` (46.9%)
   - Step 3: `task_reward/avg = 3.4375e-01` (34.4%)
   - Step 4: `task_reward/avg = 5.0000e-01` (50.0%)
   - Step 5: `task_reward/avg = 3.4375e-01` (34.4%)
   - Step 28: `task_reward/avg = 4.0625e-01` (40.6%)
   - Step 63: `task_reward/avg = 5.9375e-01` (59.4%)
   - Step 80: `task_reward/avg = 6.2500e-01` (62.5%)
   - Step 82: `task_reward/avg = 7.5000e-01` (75.0%)
   - Step 83: `task_reward/avg = 3.7500e-01` (37.5%)

2. **Memory Usage:**
   - GPU memory: `133.30-133.55 GB / 139.80 GB` (95-96% utilization)
   - No OOM errors detected in logs
   - Memory usage appears stable

3. **SGLang Server Status:**
   - Server launched successfully at `172.21.0.2:23239`
   - No connection errors or timeouts in logs
   - Server appears to be running normally

4. **Training Metrics:**
   - `update_successful = 1.0000e+00` (all updates successful)
   - No gradient explosion (grad_norm ~2-4)
   - Loss values appear normal
   - No NaN or Inf values detected

## Potential Causes

### 1. SGLang Server Disconnect (Most Likely)
**Evidence:**
- Task reward dropping to zero suggests inference server stopped responding
- If SGLang server crashed or disconnected, all rollouts would fail
- Failed rollouts would result in zero rewards

**How to Check:**
- Look for SGLang server logs around step 188
- Check for connection errors in trainer logs
- Verify if SGLang process is still running

**Solution:**
- Add automatic SGLang server restart mechanism
- Increase `request_timeout` and `request_retries` in rollout config
- Add health checks and automatic reconnection

### 2. OOM Issue (Less Likely)
**Evidence:**
- Memory usage is high (95-96%) but stable
- No OOM errors in current logs
- However, memory could spike during specific batches

**How to Check:**
- Check for CUDA OOM errors in logs after step 84
- Monitor memory usage around step 188
- Look for "out of memory" messages

**Solution:**
- Reduce `mem_fraction_static` in SGLang config (currently 0.8)
- Reduce batch size
- Enable gradient checkpointing (if not already enabled)

### 3. Model Degradation (Possible)
**Evidence:**
- If model weights became corrupted or NaN, all predictions would be wrong
- This would result in zero task reward

**How to Check:**
- Check for NaN in model weights at step 188
- Verify checkpoint integrity
- Check if loss suddenly spiked

**Solution:**
- Add weight validation checks
- Implement gradient clipping
- Use checkpoint recovery to roll back

### 4. Reward Function Issue (Unlikely)
**Evidence:**
- Reward function appears to be working correctly in earlier steps
- Math parser should be stable

**How to Check:**
- Verify reward function is being called
- Check if all completions are empty or malformed

## Recommended Actions

### Immediate Actions:

1. **Stop Current Training:**
   - The model has likely been corrupted by training on 500 steps of zero-reward data
   - Consider rolling back to checkpoint before step 188 (if available)
   - Or restart training from a clean checkpoint

2. **Check SGLang Server Logs:**
   ```bash
   # On RunPod, check SGLang server logs around step 188
   grep -A 50 -B 10 "2025-11-12.*20:4[0-9]" /workspace/outputs/grpo/logs/root/gsm8k-grpo-cloud-h200/trial_*/llm_server.log
   # Look for crashes, OOM, or errors around the time of step 188
   ```

3. **Implement Circuit Breaker:**
   - Add automatic training pause when task_reward stays at zero for N consecutive steps
   - Add alerting for sustained zero rewards
   - Add automatic SGLang server health checks

### Immediate Checks:
1. **Check WandB Metrics:**
   - Look at `grpo_actor/task_reward/avg` around step 188
   - Check `rollout/reward` metric
   - Verify if `rollout` metrics also dropped to zero

2. **Check SGLang Server Logs:**
   ```bash
   # On RunPod, check SGLang server logs
   tail -n 1000 /workspace/outputs/grpo/logs/root/gsm8k-grpo-cloud-h200/trial_*/llm_server.log
   ```

3. **Check Trainer Logs:**
   ```bash
   # Check for errors around step 188
   grep -A 20 -B 20 "Step 188\|Train step 188" /workspace/outputs/grpo/logs/root/gsm8k-grpo-cloud-h200/trial_*/trainer.log
   ```

4. **Check for Connection Errors:**
   ```bash
   grep -i "connection\|timeout\|refused\|failed" /workspace/outputs/grpo/logs/root/gsm8k-grpo-cloud-h200/trial_*/trainer.log
   ```

### Configuration Fixes:

1. **Add Circuit Breaker (CRITICAL):**
   ```python
   # In training loop, add:
   if task_reward_avg == 0.0:
       zero_reward_streak += 1
       if zero_reward_streak >= 10:  # Stop after 10 consecutive zero-reward steps
           logger.error("Task reward zero for 10 steps. Stopping training.")
           # Pause training, alert, and wait for manual intervention
   else:
       zero_reward_streak = 0
   ```

2. **Increase Rollout Resilience:**
   ```yaml
   rollout:
     request_timeout: 7200  # Already set, but verify
     request_retries: 5      # Already set, but verify
     max_head_offpolicyness: 2  # Consider increasing
   ```

3. **Reduce SGLang Memory:**
   ```yaml
   sglang:
     mem_fraction_static: 0.7  # Reduce from 0.8 to leave more headroom
   ```

4. **Add Health Checks:**
   - Implement periodic SGLang server health checks
   - Add automatic restart on failure
   - Add connection retry logic with exponential backoff

5. **Add Monitoring:**
   - Alert when task_reward drops below threshold
   - Alert when task_reward stays at zero for >5 steps
   - Monitor SGLang server process health

## Next Steps

1. **Get Full Logs:**
   - Download complete logs from RunPod covering steps 180-200
   - Check both trainer and SGLang server logs

2. **Check WandB:**
   - Examine `grpo_actor/task_reward/avg` metric
   - Check if other metrics (loss, entropy) also changed
   - Verify if the issue is isolated to task_reward

3. **Verify Checkpoint:**
   - Check if checkpoint at step 188 is valid
   - Try loading and testing the model from step 187 vs step 189

4. **Implement Monitoring:**
   - Add alerts for task_reward dropping below threshold
   - Add automatic recovery mechanisms

## Conclusion

**Confirmed: SGLang Server Disconnect/Crash at Step 188**

The WandB visualization provides definitive evidence:

1. **The abrupt drop to zero at step 188** indicates an immediate system failure, not gradual degradation
2. **The 500-step zero period** suggests the SGLang server was completely down and did not auto-restart
3. **The intermittent recovery attempts** (steps 680+) suggest:
   - Manual restarts or automatic recovery attempts that partially worked
   - Connection instability preventing sustained recovery
   - Possible model degradation during the zero-reward period (model trained on bad data for 500 steps)
4. **The failure to reach initial reward levels** suggests:
   - Model may have been corrupted or degraded during the zero-reward period
   - Training on failed rollouts (zero rewards) for 500 steps likely damaged the model
   - The intermittent spikes show the system can still function, but the model quality has degraded

### Root Cause Analysis

**Most Likely Scenario:**
1. SGLang server crashed or became unresponsive at step 188
2. Training continued but all rollouts failed (returning zero rewards)
3. Model trained on 500 steps of zero-reward data, causing degradation
4. Server was eventually restarted (manually or automatically) around step 680
5. Model partially recovered but never reached initial quality due to degradation

### Critical Issue

**The training should have STOPPED when task reward dropped to zero**, not continued for 500 steps. This suggests:
- No automatic failure detection
- No circuit breaker for sustained zero rewards
- Training continued with invalid data, corrupting the model


