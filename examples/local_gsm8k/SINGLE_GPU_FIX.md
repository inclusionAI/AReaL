# Single-GPU Fix: Using Disk-Based Weight Updates

## Problem

AReaL's default weight update mechanism uses NCCL broadcast (`WeightUpdateMeta.from_fsdp_xccl`), which requires separate GPUs for each rank. On single-GPU setups:
- Training process uses GPU 0
- SGLang inference process uses GPU 0  
- NCCL tries to create 2 ranks (trainer rank 0, SGLang rank 1) but both use the same GPU
- **Error**: "Duplicate GPU detected : rank 0 and rank 1 both on CUDA device 1000"

## Solution

Use **disk-based weight updates** (`WeightUpdateMeta.from_disk`) for single-GPU setups:
- Trainer saves weights to disk
- SGLang server loads weights from disk
- No NCCL communication required ✅

## Implementation

The fix has been applied to `examples/math/gsm8k_grpo.py`:

```python
# Detect single-GPU setup
if config.cluster.n_gpus_per_node == 1 and allocation_mode.gen.world_size == 1:
    # Use disk-based updates (no NCCL)
    weight_update_meta = WeightUpdateMeta.from_disk(
        config.experiment_name,
        config.trial_name,
        config.cluster.fileroot,
    )
else:
    # Use NCCL broadcast (multi-GPU)
    weight_update_meta = WeightUpdateMeta.from_fsdp_xccl(allocation_mode)
```

## How It Works

1. **Disk Update Path** (`areal/engine/fsdp_engine.py:403-425`):
   - Trainer saves model to disk at `{fileroot}/{exp}/{trial}/weight_update/`
   - Signals SGLang server via name resolution
   - SGLang server loads weights from disk
   - Cleans up temp files

2. **Advantages**:
   - ✅ Works with single GPU
   - ✅ No NCCL dependency
   - ✅ More reliable for small setups
   - ✅ Easier to debug (can inspect saved weights)

3. **Trade-offs**:
   - ⚠️ Slightly slower than NCCL broadcast (disk I/O)
   - ⚠️ Uses disk space temporarily

## Testing

Run training again with the updated script:

```bash
cd /workspace/AReaL
export WANDB_API_KEY=$(cat wandb/.wandb_api_key)

python3 -m areal.launcher.local examples/math/gsm8k_grpo.py \
    --config examples/local_gsm8k/train_grpo.yaml \
    experiment_name=gsm8k-grpo-local \
    trial_name=trial0
```

This should now work without the NCCL duplicate GPU error!

## Documentation

AReaL supports both weight update methods:
- **NCCL/XCCL**: Fast distributed updates (`from_fsdp_xccl`, `from_megatron_xccl`)
- **Disk**: Compatible updates via file system (`from_disk`)

The codebase already has full support for disk updates - we just need to choose the right method based on GPU count.

