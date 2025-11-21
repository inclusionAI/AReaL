# RunPod Container Stop Behavior - Important!

## ⚠️ Critical Issue: Container Auto-Restart Loop

**Problem**: RunPod has a default behavior where containers **automatically restart** when they exit. This means:

1. ✅ Training script runs and completes
2. ✅ Container exits (normal behavior)
3. ❌ **RunPod automatically restarts the container**
4. ❌ **Script runs again** (wasting money!)

This can create an **infinite loop** that wastes money!

## How RunPod Container Lifecycle Works

### Normal Flow:
```
Container Start → Script Runs → Training Completes → Container Exits → Pod Stops
```

### Problem Flow (with auto-restart):
```
Container Start → Script Runs → Training Completes → Container Exits 
    → RunPod Restarts Container → Script Runs Again → (infinite loop!)
```

## Solutions

### Solution 1: Disable Auto-Restart (Recommended)

**In RunPod Pod Settings:**
1. Go to your pod settings
2. Find **"Restart Policy"** or **"Auto-Restart"**
3. Set it to **"Never"** or **"On Failure Only"**
4. This prevents the container from restarting when it exits normally

**Where to find it:**
- Pod settings → Advanced → Restart Policy
- Or in the pod creation/template settings

### Solution 2: Use Completion Marker (Implemented)

The training script now includes a **completion marker** system:

1. **At start**: Script checks for completion markers
   - If found → Script exits immediately (prevents re-running)
   - If not found → Training proceeds normally

2. **At end**: Script creates a completion marker
   - Marker file: `/workspace/outputs/training_completed_YYYYMMDD_HHMMSS.marker`
   - Contains: Completion timestamp, experiment name, trial name

**How it works:**
```bash
# Start of script
if completion_marker_exists:
    print("Training already completed, exiting to save money")
    exit 0

# End of script (after training)
create_completion_marker()
print("Stop the pod manually to save costs!")
exit 0
```

**If container restarts:**
- Script checks for marker → Found → Exits immediately
- No training runs → No money wasted!

### Solution 3: Manual Pod Stop

**After training completes:**
1. Go to RunPod dashboard: https://www.runpod.io/console/pods
2. Find your pod
3. Click **"Stop"** button
4. Pod stops → No more charges

**⚠️ Important**: Do this **immediately** after training completes to avoid charges!

### Solution 4: Enable Auto-Shutdown (If Available)

Some RunPod configurations support **auto-shutdown**:
- Pod stops automatically after being idle for X minutes
- Check pod settings for "Idle Timeout" or "Auto-Shutdown"

## Current Implementation

The `run_training_cloud.sh` script now includes:

1. **Completion marker check** at start (prevents re-running)
2. **Completion marker creation** at end (marks training as done)
3. **Clear instructions** to stop pod manually

## Best Practices

1. ✅ **Disable auto-restart** in pod settings (most important!)
2. ✅ **Stop pod manually** after training completes
3. ✅ **Monitor pod status** in RunPod dashboard
4. ✅ **Use completion markers** (already implemented)
5. ✅ **Set budget alerts** if available in RunPod

## How to Verify

### Check if auto-restart is enabled:
```bash
# In RunPod dashboard, check pod settings
# Look for "Restart Policy" or "Auto-Restart" setting
```

### Check completion markers:
```bash
# Inside pod
ls -lh /workspace/outputs/training_completed_*.marker
```

### Check if script will re-run:
```bash
# The script checks for markers at start
# If markers exist, it will exit immediately
```

## Example: What Happens

### Without Protection (Bad):
```
1. Training completes → Container exits
2. RunPod restarts container
3. Script runs again → Wastes money!
4. Training completes → Container exits
5. (Infinite loop...)
```

### With Protection (Good):
```
1. Training completes → Creates marker → Container exits
2. RunPod restarts container
3. Script checks for marker → Found → Exits immediately
4. No training runs → No money wasted!
5. (But you should still stop the pod manually!)
```

## Manual Pod Stop Instructions

**After training completes, you MUST stop the pod:**

1. **Go to RunPod Dashboard**: https://www.runpod.io/console/pods
2. **Find your pod** (look for the running pod)
3. **Click "Stop"** button
4. **Confirm** the stop action

**Your checkpoints are safe!** They're stored in the network volume and will persist even after the pod stops.

## Summary

- ✅ **Completion markers** prevent script from re-running
- ⚠️ **You still need to stop the pod manually** to avoid charges
- ✅ **Disable auto-restart** in pod settings for best protection
- ✅ **Checkpoints are safe** in network volume

The completion marker system prevents wasted training runs, but **you must still stop the pod manually** to avoid charges for idle time!

