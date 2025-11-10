# Checkpoint Saving Fix

## Problem Identified

Training completed successfully, but **no checkpoints were saved to the volume storage**.

### Root Cause

The config files were using a **relative path** for `fileroot`:
```yaml
cluster:
  fileroot: ./outputs/grpo
```

When the training script runs from `/workspace/AReaL`, this relative path resolves to:
- `/workspace/AReaL/outputs/grpo` ❌ (container disk, not persistent)

Instead of:
- `/workspace/outputs/grpo` ✅ (mounted volume, persistent)

### Why This Happened

1. **Relative paths** are resolved from the current working directory
2. Training script runs from `/workspace/AReaL` (where code is)
3. `./outputs/grpo` becomes `/workspace/AReaL/outputs/grpo`
4. This is on the **container disk** (not the mounted volume)
5. Container disk is **not persistent** - data is lost when pod stops

## Solution

**Changed all config files to use absolute path:**

```yaml
cluster:
  fileroot: /workspace/outputs/grpo  # Absolute path to mounted volume
```

This ensures checkpoints are always saved to the **mounted volume** (`/workspace/outputs`), which persists across pod restarts.

## Files Updated

- ✅ `gsm8k_grpo_1hour_a40.yaml` - Changed to `/workspace/outputs/grpo`
- ✅ `gsm8k_grpo_1hour.yaml` - Changed to `/workspace/outputs/grpo`
- ✅ `gsm8k_grpo_3hour.yaml` - Changed to `/workspace/outputs/grpo`
- ✅ `gsm8k_grpo_fast.yaml` - Changed to `/workspace/outputs/grpo`
- ✅ `gsm8k_grpo_cloud.yaml` - Changed to `/workspace/outputs/grpo`

## Verification

After training, verify checkpoints are saved:

```bash
# Inside pod, check mounted volume
ls -lh /workspace/outputs/grpo/checkpoints/

# Should show:
# gsm8k-grpo-cloud-1hour/
#   └── trial0/
#       └── checkpoint_epoch_1_step_63/
#           ├── actor.pt
#           ├── optimizer.pt
#           └── ...
```

**If you see checkpoints here, they're saved to the persistent volume!** ✅

## Additional Fix: Git Clone Command

Also fixed the Docker command to be more robust:
- Added `set -e` to fail fast on errors
- Improved git branch checkout logic
- Added explicit `cd /workspace/AReaL` to ensure correct directory
- Better error handling for git operations

## Summary

**Problem**: Relative path `./outputs/grpo` saved to container disk (not persistent)  
**Fix**: Absolute path `/workspace/outputs/grpo` saves to mounted volume (persistent)  
**Result**: Checkpoints now persist across pod restarts! ✅

