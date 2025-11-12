# SGLang Server Crash Diagnosis

## Problem Identified

Training was working until **step 940**, then:
- **Task reward dropped to zero** and stayed at zero
- **Connection refused errors** to SGLang server (`172.17.0.2:13231`)
- Training continued but learned nothing (no valid rollouts)

## Root Cause

The **SGLang inference server crashed or stopped responding** around step 940. When the server is down:
- All rollout generation requests fail
- No valid completions are generated
- Rewards are zero (or garbage)
- Training continues but learns from invalid data

## Error Pattern

```
ConnectionRefusedError(111, "Connect call failed ('172.17.0.2', 13231)")
```

This indicates the SGLang server process:
1. Crashed (OOM, segmentation fault, etc.)
2. Hung/froze
3. Was killed by system
4. Lost network connectivity

## Diagnostic Steps

### 1. Check SGLang Server Logs

```bash
# Inside Docker container
# Find SGLang server logs
find outputs/grpo/logs -name "*llm_server*" -o -name "*sglang*" | head -5

# Check most recent logs
tail -200 outputs/grpo/logs/*/llm_server.log 2>/dev/null | tail -100

# Look for crash/error messages
grep -i "error\|crash\|killed\|segfault\|oom" outputs/grpo/logs/*/llm_server.log 2>/dev/null
```

### 2. Check System Logs for OOM Kills

```bash
# Check for out-of-memory kills
dmesg | grep -i "out of memory\|killed" | tail -20

# Check system journal
journalctl -n 50 --no-pager | grep -i "killed\|oom\|sglang"
```

### 3. Check GPU Memory Usage

```bash
# Check current GPU memory
nvidia-smi

# Check if there are zombie processes
ps aux | grep sglang
ps aux | grep python | grep -i server
```

### 4. Check Training Logs Around Step 940

```bash
# Check trainer logs around step 940
grep -A 10 -B 10 "step.*940\|global_step.*940" outputs/grpo/logs/*/trainer.log 2>/dev/null

# Check for weight update errors
grep -i "weight.*update\|disk.*update" outputs/grpo/logs/*/trainer.log 2>/dev/null | tail -20
```

### 5. Check WandB for Reward Drop

1. Go to WandB dashboard
2. Project: `gsm8k-grpo-local`
3. Look at `grpo_actor/task_reward/avg` metric
4. Check exact step where reward dropped to zero
5. Correlate with any other metrics (loss, GPU memory, etc.)

## Common Causes

### 1. Out of Memory (OOM)

**Symptoms:**
- Server process killed by system
- `dmesg` shows OOM kill
- GPU memory at 100% before crash

**Solution:**
- Reduce `sglang.mem_fraction_static` (0.8 → 0.6 or 0.5)
- Reduce `max_new_tokens` (512 → 256)
- Reduce `max_concurrent_rollouts` (32 → 16)
- Enable gradient checkpointing

### 2. Memory Leak

**Symptoms:**
- Memory usage gradually increases
- Server crashes after many steps
- No OOM kill in logs

**Solution:**
- Restart training from checkpoint before crash
- Reduce memory usage (see OOM solutions)
- Check for memory leaks in SGLang version

### 3. Disk-Based Weight Update Failure

**Symptoms:**
- Server crashes during weight update
- Timeout errors before crash
- Disk I/O errors

**Solution:**
- Increase `rollout.request_timeout` (3600 → 7200)
- Check disk space: `df -h`
- Check disk I/O: `iostat -x 1`

### 4. SGLang Server Bug

**Symptoms:**
- Server crashes with segmentation fault
- Server hangs (no response but process exists)
- Inconsistent crashes

**Solution:**
- Check SGLang version compatibility
- Try different SGLang version
- Report bug to SGLang maintainers

## Recovery Plan

### Option 1: Resume from Checkpoint Before Crash

If you have a checkpoint from before step 940:

```bash
# Resume training from checkpoint (auto-detects)
bash examples/docker_gsm8k/run_full_training.sh
```

The training will automatically resume from the last checkpoint. If the checkpoint is from step 900, it will continue from there.

### Option 2: Resume from Last Good Checkpoint

If you want to resume from a specific checkpoint:

1. **Find last good checkpoint** (before step 940):
   ```bash
   ls -lt outputs/grpo/checkpoints/gsm8k-grpo-full-local/trial0/ | head -10
   ```

2. **Verify checkpoint exists**:
   ```bash
   ls outputs/grpo/checkpoints/gsm8k-grpo-full-local/trial0/checkpoint_epoch_*_step_*/
   ```

3. **Resume training** (auto-detects checkpoint):
   ```bash
   bash examples/docker_gsm8k/run_full_training.sh
   ```

### Option 3: Fix Configuration and Resume

If the crash was due to memory issues:

1. **Update config** to reduce memory:
   ```yaml
   sglang:
     mem_fraction_static: 0.6  # Reduced from 0.8
   
   rollout:
     max_concurrent_rollouts: 16  # Reduced from 32
   
   gconfig:
     max_new_tokens: 256  # Reduced from 512
   ```

2. **Resume training**:
   ```bash
   bash examples/docker_gsm8k/run_full_training.sh
   ```

## Prevention

### 1. Monitor Server Health

Add health checks (future enhancement):
- Periodic health checks to SGLang server
- Automatic restart if server is down
- Alert when server becomes unresponsive

### 2. Reduce Memory Usage

For long training runs, use conservative memory settings:

```yaml
sglang:
  mem_fraction_static: 0.6  # Leave more headroom

rollout:
  max_concurrent_rollouts: 16  # Reduce concurrent requests

gconfig:
  max_new_tokens: 256  # Shorter generations
```

### 3. Frequent Checkpoints

Already configured in `gsm8k_grpo_full.yaml`:
- Every 50 steps
- Every 30 minutes
- After each epoch

This minimizes progress loss if crash occurs.

### 4. Monitor Training Metrics

Watch WandB dashboard for:
- **Task reward**: Should stay above zero
- **GPU memory**: Should be stable
- **Generation errors**: Should be zero
- **Connection errors**: Should be zero

## Immediate Action

1. **Check logs** to identify exact cause (see Diagnostic Steps above)
2. **Resume from checkpoint** before step 940
3. **Apply memory fixes** if OOM was the cause
4. **Monitor closely** for first 100 steps after resume

## Next Steps

After identifying the root cause:
1. Apply appropriate fix (memory reduction, timeout increase, etc.)
2. Resume training from last good checkpoint
3. Monitor for recurrence
4. Consider adding automatic server health checks (future enhancement)

