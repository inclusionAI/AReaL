# GRPO Training Error: NCCL Duplicate GPU

## Status: Partial Success

The training **started successfully** and completed:
- ✅ Config loading fixed (WandB API key)
- ✅ Model loaded (Qwen 0.5B)
- ✅ Dataset loaded (GSM8K)
- ✅ SGLang server started
- ✅ First training step completed (rollout, logp recompute, advantages, PPO update)
- ❌ Failed during weight update broadcast

## Error Details

```
NCCL error: Duplicate GPU detected : rank 0 and rank 1 both on CUDA device 1000
```

This occurs during distributed weight updates. On single-GPU setups, AReaL's weight synchronization mechanism expects multiple GPUs/ranks.

## Solution Options

### Option 1: Use Gloo Backend (Recommended for Single GPU)

The config already uses single GPU setup. The issue is in NCCL initialization for weight updates. We may need to:
1. Set `NCCL_DISABLE=1` and use Gloo backend
2. Or modify the weight update logic to skip broadcasting on single GPU

### Option 2: Check AReaL Single-GPU Documentation

AReaL may have specific single-GPU setup requirements. Check:
- `docs/tutorial/quickstart.md`
- Examples with `n_gpus_per_node: 1`

### Option 3: Disable Async Weight Updates

For single GPU, we might need to disable async weight synchronization.

## Current Progress

Training **did progress** to the first update step:
- Rollouts completed
- Rewards computed  
- PPO loss computed
- Failed during weight broadcast to inference servers

## Next Steps

1. Check if AReaL supports true single-GPU mode
2. Add NCCL workaround environment variables
3. Or modify allocation to avoid distributed weight sync

