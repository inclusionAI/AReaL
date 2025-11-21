# Fix for Reasoning Model 4-GPU Training Issue

## Problem

When running the `reasoning_2000samples_4GPUs` config on RunPod with 4x A40 GPUs, the training failed with:

```
torch.AcceleratorError: CUDA error: CUDA-capable device(s) is/are busy or unavailable
```

This error occurs when SGLang tries to initialize on GPU 0, but the GPU is already busy or in an unavailable state.

## Root Cause

1.  **Zombie GPU 0**: Physical GPU 0 on the RunPod instance appears to be in a persistent "busy" or zombie state, possibly due to Exclusive Process mode or driver issues that persist across container restarts.
2.  **Configuration Error**: The batch size (8) was not divisible by the number of training GPUs (3), causing `ValueError: batch size(8) must be divisible by world_size(3)!`.

## Solution

We implemented a comprehensive fix involving both infrastructure cleanup and configuration adjustments.

### 1. Infrastructure Fixes (in `run_training_cloud.sh`)

*   **Aggressive Cleanup**: Automatically installs `psmisc` and `lsof` to identify and kill any process holding `/dev/nvidia*` device files.
*   **GPU Reordering Workaround**: Detects the 4-GPU setup and applies `CUDA_VISIBLE_DEVICES=3,2,1,0`.
    *   This maps logical device 0 (critical for SGLang) to physical GPU 3.
    *   This moves the "busy" physical GPU 0 to logical device 3 (used by a training actor), bypassing the SGLang initialization failure.

### 2. Configuration Fixes (in `gsm8k_grpo_reasoning_2000samples_4GPUs.yaml`)

*   **Batch Size Adjustment**: Changed `batch_size` from 8 to 6.
    *   The allocation mode `sglang.d1t1p1+d3t1p1` uses 1 GPU for SGLang and 3 GPUs for training.
    *   Data parallelism requires batch size to be divisible by the training world size (3).
    *   6 is divisible by 3, resolving the dataloader error.

## Testing

After applying these fixes:
1.  Cleanup script runs successfully.
2.  GPU order is reversed to avoid GPU 0.
3.  SGLang server initializes successfully on physical GPU 3.
4.  Training actors initialize on physical GPUs 2, 1, and 0.
5.  Training loop starts successfully without batch size errors.

## If Issue Persists

If errors return:

1.  **Check RunPod Container**: Ensure the container is fully stopped before restarting.
2.  **Restart Pod**: If physical GPU 0 causes training actors to fail (e.g., NCCL errors on rank 2), the entire pod must be restarted to reset the GPU driver state.
3.  **Manual Cleanup**: SSH into the pod and run:
    ```bash
    pkill -9 -f sglang
    pkill -9 -f areal
    nvidia-smi --gpu-reset
    ```

## Related Files

-   `examples/cloud_gsm8k/run_training_cloud.sh` - Training script with cleanup and reordering logic.
-   `examples/cloud_gsm8k/gsm8k_grpo_reasoning_2000samples_4GPUs.yaml` - 4-GPU reasoning config with corrected batch size.
