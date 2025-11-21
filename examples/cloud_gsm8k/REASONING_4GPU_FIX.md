# Fix for Reasoning Model 4-GPU Training Issue

## Problem

When running the `reasoning_2000samples_4GPUs` config on RunPod with 4x A40 GPUs, the training failed consistently.

**Initial Error:** `CUDA error: CUDA-capable device(s) is/are busy or unavailable` on SGLang initialization (GPU 0).

**Secondary Error:** After moving SGLang to GPU 3, the error shifted to Training Rank 2 (which landed on GPU 0).

## Root Cause

**Physical GPU 0 is dead/zombie.**
Regardless of cleanup or reordering, any process attempting to initialize a CUDA context on Physical GPU 0 fails with "Device busy or unavailable".

## Solution

We must exclude Physical GPU 0 entirely from the training job.
We have 4 GPUs (3, 2, 1, 0). Since 0 is dead, we will use **3 GPUs (3, 2, 1)**.

### 1. Infrastructure Logic (in `run_training_cloud.sh`)

*   **Reordering**: We keep the `CUDA_VISIBLE_DEVICES=3,2,1,0` logic.
    *   This exposes Physical GPUs [3, 2, 1, 0] as Logical GPUs [0, 1, 2, 3].
*   **Selection**: The launcher will pick the first N logical GPUs required by the config.
    *   If config needs 3 GPUs, it picks Logical 0, 1, 2.
    *   This maps to Physical 3, 2, 1.
    *   Physical 0 (Logical 3) remains unused.

### 2. Configuration Fixes (in `gsm8k_grpo_reasoning_2000samples_4GPUs.yaml`)

*   **Downscale to 3 GPUs**: Changed `allocation_mode` from `sglang.d1t1p1+d3t1p1` (4 GPUs) to `sglang.d1t1p1+d2t1p1` (3 GPUs).
    *   1 GPU for SGLang (Logical 0 -> Physical 3).
    *   2 GPUs for Training (Logical 1, 2 -> Physical 2, 1).
*   **Batch Size**: Kept at 6.
    *   6 is divisible by 2 (new training world size), so no dataloader errors.

## Testing

With these changes:
1.  SGLang initializes on Physical GPU 3.
2.  Training Rank 0 initializes on Physical GPU 2.
3.  Training Rank 1 initializes on Physical GPU 1.
4.  Physical GPU 0 is untouched.
5.  Training should proceed successfully with 3 GPUs.

## Related Files

-   `examples/cloud_gsm8k/gsm8k_grpo_reasoning_2000samples_4GPUs.yaml` - Modified to use 3 GPUs.
-   `examples/cloud_gsm8k/run_training_cloud.sh` - Handles GPU reordering.
