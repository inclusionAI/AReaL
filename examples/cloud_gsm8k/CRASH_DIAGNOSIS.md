# Training Crash Diagnosis: Step 57 Pattern

## ⚠️ UPDATE: This Was Actually a Container Restart Loop

**The "crash" was actually a container restart loop!** See `CONTAINER_RESTART_LOOP_FIX.md` for the actual issue.

The container was restarting every ~17 seconds because:
1. Docker command tried to `git clone` AReaL
2. AReaL directory already existed
3. `git clone` failed with "destination path already exists"
4. Container exited
5. RunPod auto-restarted container
6. Loop repeated infinitely

**Fix**: Updated Docker command to remove AReaL before cloning: `(rm -rf AReaL || true)`

---

## Original Analysis (For Reference)

Training consistently crashes around step 57 (out of 250 steps) with:
- Terminal closes
- GPU utilization drops to zero
- No error messages in terminal output
- Pattern repeats across multiple runs

## Root Cause Analysis

Based on the logs, the crash happens **after weight updates**:
1. Training step completes successfully
2. Weight update starts: "Loading weights from disk done"
3. Next step begins
4. **Crash occurs** (likely during next weight update or rollout)

## Potential Causes

### 1. Weight Update Timeout (Most Likely)

The disk-based weight update has a **hardcoded 120-second timeout**:
```python
save_timestamp = float(name_resolve.wait(update_name, timeout=120))
```

**Issue**: If the weight update takes longer than 120 seconds, it times out and crashes.

**Solution**: Increase timeout in config or modify code.

### 2. ProcessPoolExecutor Crash

The weight update runs in a `ProcessPoolExecutor` which might be crashing silently:
```python
fut = self.executor.submit(_update_weights_from_disk, ...)
```

**Issue**: Process crashes might not be logged to terminal.

**Solution**: Check system logs, add error handling.

### 3. Memory Leak During Weight Updates

Each weight update:
- Saves model to disk (~1-2GB for Qwen 0.5B)
- Loads model from disk in SGLang server
- Cleans up temp files

**Issue**: Memory might accumulate if cleanup fails.

**Solution**: Check memory usage, verify cleanup.

### 4. Disk I/O Issues

Weight updates involve heavy disk I/O:
- Writing model weights to disk
- Reading model weights from disk
- Network volume might be slow

**Issue**: Slow disk I/O causes timeouts or crashes.

**Solution**: Check disk space, I/O performance.

### 5. SGLang Server Crash

The SGLang server might be crashing during weight loading:
- Loading weights from disk
- Memory pressure during loading
- Server restart required

**Issue**: Server crash kills the training process.

**Solution**: Check SGLang server logs.

## Immediate Diagnostic Steps

### 1. Check System Logs

```bash
# Inside pod, check for crash logs
dmesg | tail -50
journalctl -n 50

# Check for OOM kills
dmesg | grep -i "out of memory"
dmesg | grep -i "killed"
```

### 2. Check Disk Space

```bash
# Check available disk space
df -h /workspace/outputs

# Check for disk errors
dmesg | grep -i "i/o error"
```

### 3. Check Memory Usage

```bash
# Monitor memory during training
watch -n 1 'free -h && nvidia-smi'
```

### 4. Check SGLang Server Logs

```bash
# Check SGLang server logs
tail -100 /workspace/AReaL/outputs/grpo/logs/*/llm_server.log

# Look for errors or crashes
grep -i "error\|crash\|timeout\|killed" /workspace/AReaL/outputs/grpo/logs/*/llm_server.log
```

### 5. Check Weight Update Logs

```bash
# Check trainer logs for weight update errors
tail -100 /workspace/AReaL/outputs/grpo/logs/*/trainer.log

# Look for timeout or error messages
grep -i "timeout\|error\|failed\|exception" /workspace/AReaL/outputs/grpo/logs/*/trainer.log
```

## Solutions

### Solution 1: Increase Timeouts (Quick Fix)

Add to config:
```yaml
rollout:
  request_timeout: 7200  # Increase from default 3600 to 7200 seconds (2 hours)
  request_retries: 5  # Increase retries from default 3 to 5
  setup_timeout: 300  # Increase from default 120 to 300 seconds
```

### Solution 2: Reduce Weight Update Frequency

Update weights less frequently to reduce crash risk:
```yaml
# In training script, update weights every N steps instead of every step
# This reduces the number of weight updates and crash opportunities
```

### Solution 3: Add Error Handling

Modify the training script to catch and log weight update errors:
```python
try:
    actor.update_weights(weight_update_meta)
except Exception as e:
    logger.error(f"Weight update failed: {e}")
    # Continue training with old weights
    pass
```

### Solution 4: Use Regular (Non-Spot) Instance

Spot instances might be getting interrupted:
- Use regular instance to rule out spot interruptions
- If it still crashes, it's not a spot issue

### Solution 5: Check Disk I/O Performance

```bash
# Test disk write speed
dd if=/dev/zero of=/workspace/outputs/test bs=1G count=1 oflag=direct

# Test disk read speed
dd if=/workspace/outputs/test of=/dev/null bs=1G count=1 iflag=direct
```

If disk I/O is slow (< 100 MB/s), this could cause timeouts.

## Recommended Fix

**Immediate action**: Add timeout configuration to A40 config:

```yaml
rollout:
  request_timeout: 7200  # 2 hours (increase from 1 hour default)
  request_retries: 5  # More retries
  setup_timeout: 300  # 5 minutes (increase from 2 minutes)
```

This should prevent timeout-related crashes during weight updates.

## Next Steps

1. **Add timeout config** to `gsm8k_grpo_1hour_a40.yaml`
2. **Check system logs** for crash details
3. **Monitor memory** during training
4. **Check disk I/O** performance
5. **Try regular instance** to rule out spot interruptions

