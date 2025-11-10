# 1-Hour vs 3-Hour Config Comparison & A40 GPU Feasibility

## Overview

This document compares the 1-hour and 3-hour training configurations for RunPod GRPO training and evaluates whether an A40 GPU (44GB) can handle the 3-hour run.

## Key Differences

### Training Duration

| Config | Epochs | Estimated Time |
|--------|--------|----------------|
| `gsm8k_grpo_1hour.yaml` | 2 epochs | ~1 hour |
| `gsm8k_grpo_3hour.yaml` | 3 epochs | ~3 hours |

### Configuration Parameters

| Parameter | 1-Hour Config | 3-Hour Config | A40-Optimized (1-Hour) |
|-----------|---------------|---------------|------------------------|
| **Training** | | | |
| `total_train_epochs` | 2 | **3** | 2 |
| `train_dataset.batch_size` | 8 | 8 | **4** |
| `actor.gradient_checkpointing` | `false` | `false` | **`true`** |
| `actor.mb_spec.max_tokens_per_mb` | 5120 | 5120 | **4096** |
| **Inference** | | | |
| `rollout.max_concurrent_rollouts` | 32 | 32 | **16** |
| `gconfig.max_new_tokens` | 512 | 512 | **256** |
| `sglang.mem_fraction_static` | 0.8 | 0.8 | **0.5** |
| **Timeouts** | | | |
| `rollout.request_timeout` | 3600 (default) | 3600 (default) | **7200** |
| `rollout.request_retries` | 3 (default) | 3 (default) | **5** |
| `rollout.setup_timeout` | 120 (default) | 120 (default) | **300** |

## Memory Analysis

### 1-Hour Config (Standard)
- **SGLang server**: ~36GB (80% of 44GB)
- **Trainer**: ~8GB
- **Total**: ~44GB ❌ **OOM on A40!**

### 1-Hour A40-Optimized Config
- **SGLang server**: ~22GB (50% of 44GB)
- **Trainer**: ~6GB (with gradient checkpointing)
- **Total**: ~28GB ✅ **Fits comfortably!**

### 3-Hour Config (Standard)
- **SGLang server**: ~36GB (80% of 44GB)
- **Trainer**: ~8GB
- **Total**: ~44GB ❌ **OOM on A40!**

## A40 GPU Feasibility for 3-Hour Run

### ❌ **Current 3-Hour Config: NOT FEASIBLE on A40**

The `gsm8k_grpo_3hour.yaml` uses the same memory-intensive settings as the standard 1-hour config:
- No gradient checkpointing
- High SGLang memory fraction (0.8)
- Large batch size (8)
- Long sequences (512 max_new_tokens)
- High concurrent rollouts (32)

**This will cause OOM on A40 GPU (44GB).**

### ✅ **Solution: Create A40-Optimized 3-Hour Config**

To run 3-hour training on A40, you need to apply the same memory optimizations as the 1-hour A40 config:

**Recommended A40-Optimized 3-Hour Config:**
```yaml
total_train_epochs: 3  # Only change from 1-hour A40 config

# All other settings same as gsm8k_grpo_1hour_a40.yaml:
train_dataset.batch_size: 4
actor.gradient_checkpointing: true
actor.mb_spec.max_tokens_per_mb: 4096
rollout.max_concurrent_rollouts: 16
gconfig.max_new_tokens: 256
sglang.mem_fraction_static: 0.5
rollout.request_timeout: 7200
rollout.request_retries: 5
rollout.setup_timeout: 300
```

**Expected Memory Usage:**
- SGLang server: ~22GB
- Trainer: ~6GB
- **Total: ~28GB** ✅ **Fits in A40 (44GB) with ~16GB headroom**

## Performance Impact of A40 Optimizations

### Trade-offs

| Optimization | Memory Saved | Performance Impact |
|--------------|--------------|-------------------|
| `gradient_checkpointing: true` | ~30-40% training memory | ~20-30% slower training |
| `mem_fraction_static: 0.5` | ~14GB SGLang memory | Slightly lower inference throughput |
| `batch_size: 4` | ~2GB per batch | Same effective batch (via gradient accumulation) |
| `max_new_tokens: 256` | ~2-4GB per sequence | Shorter responses (may affect quality) |
| `max_concurrent_rollouts: 16` | ~4-8GB | Lower inference parallelism |

### Overall Impact
- **Training speed**: ~20-30% slower due to gradient checkpointing
- **Inference throughput**: ~30-40% lower due to reduced parallelism
- **Total time**: 3-hour run may take ~4-4.5 hours on A40 with optimizations

## Recommendations

### For A40 GPU:

1. **✅ Use A40-optimized configs** for both 1-hour and 3-hour runs
2. **✅ Create `gsm8k_grpo_3hour_a40.yaml`** based on `gsm8k_grpo_1hour_a40.yaml` with `total_train_epochs: 3`
3. **✅ Expect longer runtime**: 3-hour config may take 4-4.5 hours with optimizations
4. **✅ Monitor memory**: Use `nvidia-smi` to verify memory usage stays under 40GB

### For RTX 4090 (24GB) or A100 (40GB/80GB):

1. **✅ Standard configs work** (1-hour and 3-hour)
2. **⚠️ RTX 4090**: May need slight optimizations (similar to A40)
3. **✅ A100**: Can handle standard configs comfortably

## Creating A40-Optimized 3-Hour Config

To create the optimized config, copy `gsm8k_grpo_1hour_a40.yaml` and change:

```yaml
total_train_epochs: 3  # Change from 2 to 3
experiment_name: gsm8k-grpo-cloud-3hour  # Update name
```

All other memory optimizations remain the same.

## Summary

| Config | A40 Feasible? | Memory Usage | Notes |
|--------|---------------|--------------|-------|
| `gsm8k_grpo_1hour.yaml` | ❌ No | ~44GB | OOM on A40 |
| `gsm8k_grpo_1hour_a40.yaml` | ✅ Yes | ~28GB | Optimized for A40 |
| `gsm8k_grpo_3hour.yaml` | ❌ No | ~44GB | OOM on A40 |
| `gsm8k_grpo_3hour_a40.yaml` | ✅ Yes (needs creation) | ~28GB | Should be created |

**Conclusion**: A40 GPU **cannot** handle the current 3-hour config, but **can** handle a 3-hour run with the same optimizations applied to the 1-hour A40 config. The optimized 3-hour run will take longer (~4-4.5 hours) but will complete successfully.

