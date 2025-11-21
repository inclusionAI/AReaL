# Fix for Reasoning Model 4-GPU Training Issue

## Problem

When running the `reasoning_2000samples_4GPUs` config on RunPod with 4x A40 GPUs, the training fails with:

```
torch.AcceleratorError: CUDA error: CUDA-capable device(s) is/are busy or unavailable
```

This error occurs when SGLang tries to initialize on GPU 0, but the GPU is already busy or in an unavailable state.

## Root Cause

1. **Container Restart Issue**: When RunPod restarts the container (due to failure or auto-restart), leftover processes from the previous run may still be holding GPU resources.

2. **GPU State Not Reset**: The GPU driver state may not be fully reset between container restarts, leaving the GPU in a "busy" state.

3. **Process Cleanup**: The local launcher doesn't clean up processes before starting, so if a previous run failed, zombie processes may still be using the GPU.

## Solution

Added GPU cleanup logic to `run_training_cloud.sh` that:

1. **Kills Leftover Processes**: Before starting training, kills any processes that might be using the GPU:
   - `sglang` processes
   - `areal.launcher` processes
   - `torchrun` processes
   - Any Python processes running the training script

2. **Waits for Cleanup**: Adds delays to allow processes to fully terminate.

3. **Checks GPU Utilization**: Before starting, checks if GPUs are busy and waits if needed.

4. **Resets GPU State**: Attempts to reset GPU state using `nvidia-smi --gpu-reset` (if available).

## Changes Made

The cleanup logic was added to `run_training_cloud.sh` right before GPU detection:

```bash
# Clean up any leftover processes that might be using the GPU
echo "Cleaning up any leftover GPU processes..."
# Kill any Python processes that might be holding the GPU
pkill -9 -f "sglang" 2>/dev/null || true
pkill -9 -f "areal.launcher" 2>/dev/null || true
pkill -9 -f "torchrun" 2>/dev/null || true
pkill -9 -f "python.*gsm8k_grpo" 2>/dev/null || true
# Wait a moment for processes to terminate
sleep 3
# Additional cleanup: kill any processes using CUDA devices
if command -v fuser &> /dev/null; then
    for gpu_id in /dev/nvidia*; do
        if [ -e "$gpu_id" ]; then
            fuser -k "$gpu_id" 2>/dev/null || true
        fi
    done
    sleep 1
fi
# Try to reset GPU state (may require root, so ignore errors)
nvidia-smi --gpu-reset -i 0 2>/dev/null || nvidia-smi --gpu-reset 2>/dev/null || true
sleep 2
```

## Testing

After applying this fix, the training should:

1. Clean up any leftover processes before starting
2. Wait for GPUs to become available
3. Start SGLang server successfully on GPU 0
4. Start training processes on GPUs 1-3

## Additional Notes

- The cleanup uses `-9` (SIGKILL) to forcefully terminate processes
- All cleanup commands use `|| true` to prevent script failure if cleanup fails
- The script checks GPU utilization and waits if GPUs are busy
- GPU reset may require root privileges, so errors are ignored

## Updated Fix (v2)

After the initial fix didn't fully resolve the issue, we've made additional improvements:

1. **Removed GPU Reset**: The `nvidia-smi --gpu-reset` command requires root privileges and was causing warnings. Removed it.

2. **Better Process Detection**: Now uses `nvidia-smi --query-compute-apps` to find processes actually using the GPU, not just Python processes.

3. **Removed GPU Test**: The PyTorch GPU accessibility test was creating a CUDA context that interfered with SGLang initialization. Removed it.

4. **Better Diagnostics**: Added checks for GPU utilization, memory usage, and active processes.

5. **CUDA Context Cleanup**: Added checks for Python processes that might have CUDA contexts open, with delays to allow contexts to release.

6. **CUDA Debugging**: Added `CUDA_LAUNCH_BLOCKING=1` and `TORCH_USE_CUDA_DSA=1` for better error messages.

## Key Insight

The GPU accessibility test was passing (PyTorch could see the GPU), but SGLang was still failing. This is because:
- The test creates a CUDA context that doesn't get properly cleaned up
- SGLang needs exclusive access to the GPU when initializing distributed training
- CUDA contexts can persist even after Python processes exit

The fix removes the GPU test and adds delays to ensure any CUDA contexts are fully released before SGLang starts.

## If Issue Persists

If the error still occurs after this fix:

1. **Check RunPod Container**: Ensure the container is fully stopped before restarting. The GPU driver state may not reset properly on container restart.

2. **Manual Cleanup**: SSH into the pod and manually check/kill processes:
   ```bash
   # Check what's using the GPU
   nvidia-smi --query-compute-apps=pid,process_name --format=csv
   
   # Kill any processes
   pkill -9 -f sglang
   pkill -9 -f areal
   
   # Check GPU status
   nvidia-smi
   ```

3. **Restart Pod**: If GPU 0 is consistently problematic, try stopping and restarting the entire pod (not just the container) to fully reset GPU driver state.

4. **Check Allocation Mode**: The current config uses `sglang.d1t1p1+d3t1p1` which allocates GPU 0 to SGLang. If GPU 0 is problematic, you may need to:
   - Stop the pod completely and restart it
   - Or modify the allocation mode (though this is complex and not recommended)

5. **RunPod-Specific Issue**: This might be a RunPod-specific issue where GPU state isn't properly reset between container restarts. Consider:
   - Using a different GPU instance type
   - Contacting RunPod support if the issue persists

## Related Files

- `examples/cloud_gsm8k/run_training_cloud.sh` - Training script with cleanup logic
- `examples/cloud_gsm8k/gsm8k_grpo_reasoning_2000samples_4GPUs.yaml` - 4-GPU reasoning config

