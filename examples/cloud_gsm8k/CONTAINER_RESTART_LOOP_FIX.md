# Container Restart Loop Fix

## Problem Identified

The container was stuck in a **restart loop**, not a training crash. The logs show:

```
fatal: destination path 'AReaL' already exists and is not an empty directory.
```

**What happened:**
1. Container starts
2. Docker command tries to `git clone` AReaL
3. AReaL directory already exists (from previous run)
4. `git clone` fails with error
5. Docker command exits with error
6. Container exits
7. RunPod automatically restarts container
8. **Loop repeats infinitely** (~every 17 seconds)

**System logs show:**
- Container starting repeatedly every ~17 seconds
- No actual training running
- Just restart loop

## Root Cause

The Docker command in the template/run script doesn't handle the case where AReaL already exists:

```bash
# This fails if AReaL exists:
git clone -b DL4Math https://github.com/nexthybrid/AReaL.git
```

## Solution

**Updated Docker command** with smart git handling that preserves code during restarts:

```bash
bash -c "pip config set global.index-url https://pypi.org/simple && pip config set global.extra-index-url '' && cd /workspace && if [ -d AReaL/.git ]; then cd AReaL && git fetch && git checkout DL4Math && git pull || (cd .. && rm -rf AReaL && git clone -b DL4Math https://github.com/nexthybrid/AReaL.git); else rm -rf AReaL && git clone -b DL4Math https://github.com/nexthybrid/AReaL.git; fi && cd AReaL && pip install -e . && export WANDB_API_KEY=\$WANDB_API_KEY && bash examples/cloud_gsm8k/run_training_cloud.sh 1hour"
```

**Smart git handling**:
- ✅ **If AReaL exists and is valid git repo**: Updates with `git pull` (preserves code during container restarts)
- ✅ **If AReaL exists but is invalid**: Removes and clones fresh
- ✅ **If AReaL doesn't exist**: Clones fresh
- ✅ **Prevents restart loops**: Handles all cases gracefully
- ✅ **Safe during training**: Won't delete code if container restarts while training is running

## Why This Approach is Safe

**The smart git handling approach is used because:**

1. **Container restarts during training**: If container restarts (spot interruption, crash, etc.), we want to preserve the code and just update it, not delete and re-clone
2. **First-time setup**: If AReaL doesn't exist, clone fresh
3. **Corrupted repo**: If AReaL exists but git is broken, remove and clone fresh
4. **Code updates**: If repo exists and is valid, update with `git pull` to get latest changes

**This ensures:**
- ✅ Code is preserved during container restarts
- ✅ No unnecessary re-cloning (faster startup)
- ✅ Always gets latest code from your branch
- ✅ Handles all edge cases gracefully

## Why This Happens

**RunPod behavior:**
- When a container exits (even with error), RunPod can auto-restart it
- This is useful for long-running services
- But causes infinite loops if the startup command always fails

**Container disk persistence:**
- RunPod container disk persists across restarts
- `/workspace/AReaL` directory remains from previous run
- Next run tries to clone again → fails → restart loop

## Prevention

1. **Always handle existing directories** in Docker commands
2. **Use `|| true`** for cleanup commands that might fail
3. **Check container logs** if you see restart loops
4. **Use conditional logic** (`if [ -d ... ]`) for better control

## Verification

After applying the fix, check container logs:

```bash
# Should see:
# 1. "Removing AReaL..." (or no error)
# 2. "Cloning into 'AReaL'..."
# 3. "Installing AReaL..."
# 4. Training starts
```

**Not:**
- ❌ "fatal: destination path 'AReaL' already exists"
- ❌ Container restarting every 17 seconds
- ❌ No training output

## Files Updated

- ✅ `runpod_template.json` - Template Docker command
- ✅ `RUNPOD_COMPLETE_GUIDE.md` - Documentation
- ✅ `RUNPOD_QUICK_START.md` - Quick start guide
- ✅ `runpod_deploy.sh` - Deployment script
- ✅ `runpod_setup.md` - Setup guide
- ✅ `DOCKER_COMMANDS.md` - Docker commands reference

## Summary

**Problem**: Container restart loop due to `git clone` failing when AReaL exists  
**Fix**: Remove AReaL directory before cloning: `(rm -rf AReaL || true)`  
**Result**: Container starts cleanly, training runs normally

**Next time**: If you see container restarting repeatedly, check logs for "fatal: destination path already exists" or similar errors!

