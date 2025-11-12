# Running the Health Check Script

## Quick Command

**Inside Docker container**, run:

```bash
bash examples/docker_gsm8k/check_sglang_health.sh
```

## If You're Not in Docker Container

### Option 1: Enter Docker Container First

```bash
# On Windows (PowerShell or WSL)
docker exec -it areal-grpo /bin/bash

# Then inside container:
cd /workspace/AReaL
bash examples/docker_gsm8k/check_sglang_health.sh
```

### Option 2: Run Command Directly in Container

```bash
# On Windows (PowerShell or WSL)
docker exec -it areal-grpo bash -c "cd /workspace/AReaL && bash examples/docker_gsm8k/check_sglang_health.sh"
```

## What the Script Checks

1. ✅ SGLang server processes (running or not)
2. ✅ Server logs (last few lines)
3. ✅ Connection errors in logs
4. ✅ GPU memory usage
5. ✅ System memory usage
6. ✅ OOM kills (out-of-memory)
7. ✅ Available checkpoints
8. ✅ Recovery info files

## Expected Output

The script will show:
- Whether SGLang server is running
- Any connection errors found
- Memory usage (GPU and system)
- Available checkpoints for recovery
- Recovery info for auto-resume

## After Running

Based on the output:
1. **If server is down**: Resume training (it will auto-restart server)
2. **If OOM detected**: Config already updated with safer settings
3. **If checkpoints exist**: Training can resume from last checkpoint

