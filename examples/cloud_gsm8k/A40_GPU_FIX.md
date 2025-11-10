# A40 GPU Memory Fix

## Problem

A40 GPU (44GB) runs out of memory when running GRPO training because both the SGLang inference server and trainer share the same GPU.

**Error:**
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 48.00 MiB. 
GPU 0 has a total capacity of 44.42 GiB of which 27.50 MiB is free.
Process 2963820 has 36.34 GiB memory in use.  # SGLang server
Process 2965022 has 8.05 GiB memory in use.   # Trainer
```

## Quick Fix

The training script automatically detects A40 GPU and uses the optimized config:

```bash
bash examples/cloud_gsm8k/run_training_cloud.sh 1hour
```

This will automatically use `gsm8k_grpo_1hour_a40.yaml` which includes all memory optimizations.

## Manual Fix

If you need to manually specify the A40 config:

```bash
python3 -m areal.launcher.local examples/docker_gsm8k/gsm8k_grpo_1hour.py \
    --config examples/cloud_gsm8k/gsm8k_grpo_1hour_a40.yaml \
    experiment_name=gsm8k-grpo-cloud-1hour \
    trial_name=trial0
```

## What Changed in A40 Config

| Parameter | Default | A40 Optimized | Impact |
|-----------|---------|--------------|--------|
| `gradient_checkpointing` | `false` | `true` | Reduces training memory by ~30-40% |
| `sglang.mem_fraction_static` | `0.8` | `0.5` | Leaves more memory for trainer |
| `train_dataset.batch_size` | `8` | `4` | Reduces memory per batch |
| `gconfig.max_new_tokens` | `512` | `256` | Shorter sequences = less memory |
| `actor.mb_spec.max_tokens_per_mb` | `5120` | `4096` | Smaller micro-batches |
| `rollout.max_concurrent_rollouts` | `32` | `16` | Fewer parallel requests |

## Memory Breakdown

### Before (Default Config)
- SGLang server: ~36GB (80% of 44GB)
- Trainer: ~8GB
- **Total: ~44GB** ❌ **OOM!**

### After (A40 Optimized)
- SGLang server: ~22GB (50% of 44GB)
- Trainer: ~6GB (with gradient checkpointing)
- **Total: ~28GB** ✅ **Fits comfortably!**

## Additional Optimizations

If you still get OOM after using the A40 config:

### 1. Set PyTorch Memory Allocator
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 2. Further Reduce SGLang Memory
```yaml
sglang:
  mem_fraction_static: 0.4  # Even lower
```

### 3. Further Reduce Batch Size
```yaml
train_dataset:
  batch_size: 2  # Even smaller
```

### 4. Further Reduce Sequence Length
```yaml
gconfig:
  max_new_tokens: 128  # Even shorter
```

## Why A40 Needs Special Config

- **A40**: 44GB memory (less than A100's 40GB/80GB)
- **A100**: 40GB or 80GB (more headroom)
- **RTX 4090**: 24GB (but cheaper, may need even more optimization)

When both SGLang server and trainer share the same GPU, they compete for memory. A40's 44GB is tight, so we need to optimize both processes.

## Verification

After applying fixes, check memory usage:

```bash
# Inside pod
nvidia-smi

# Should show:
# - SGLang process: ~20-25GB
# - Trainer process: ~5-8GB
# - Total: ~30-35GB (comfortable for A40)
```

## Summary

✅ **Use A40-optimized config**: `gsm8k_grpo_1hour_a40.yaml`  
✅ **Script auto-detects A40**: No manual config needed  
✅ **Memory savings**: ~30-40% reduction with gradient checkpointing  
✅ **Still fits in A40**: ~28GB total usage vs 44GB capacity

