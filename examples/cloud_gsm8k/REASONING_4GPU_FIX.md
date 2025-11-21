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

## If Issue Persists

If the error still occurs after this fix:

1. **Check RunPod Container**: Ensure the container is fully stopped before restarting
2. **Manual Cleanup**: SSH into the pod and manually kill processes:
   ```bash
   pkill -9 -f sglang
   pkill -9 -f areal
   nvidia-smi --gpu-reset
   ```
3. **Check GPU Status**: Verify GPUs are available:
   ```bash
   nvidia-smi
   ```
4. **Try Different GPU**: If GPU 0 is consistently problematic, you may need to modify the allocation mode or launcher to use a different GPU

## Related Files

- `examples/cloud_gsm8k/run_training_cloud.sh` - Training script with cleanup logic
- `examples/cloud_gsm8k/gsm8k_grpo_reasoning_2000samples_4GPUs.yaml` - 4-GPU reasoning config

