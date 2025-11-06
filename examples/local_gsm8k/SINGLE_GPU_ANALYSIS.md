# Single-GPU Analysis: NCCL Duplicate GPU Error

## Root Cause

The error occurs because:

1. **Allocation Mode**: `sglang.d1p1t1+d1p1t1` creates a **DECOUPLED_TRAIN** allocation
   - SGLang inference: 1 GPU (rank 1 in weight update group)
   - Training: 1 GPU (rank 0 in weight update group)
   - Total: 2 ranks sharing 1 physical GPU

2. **Weight Update Mechanism** (from `areal/engine/fsdp_engine.py:350`):
   ```python
   world_size = meta.alloc_mode.gen.world_size + 1  # 1 + 1 = 2
   ```
   This creates an NCCL process group with 2 ranks, but both try to use GPU 0.

3. **NCCL Limitation**: NCCL doesn't allow multiple ranks in the same process group to use the same physical GPU device.

## Findings from Codebase Search

### Single-GPU Support

✅ **Allocation mode parsing supports single GPU**: `d1p1t1` is valid  
✅ **Config accepts `n_gpus_per_node: 1`**: No validation errors  
❌ **Weight update assumes separate GPUs**: Creates multi-rank NCCL group even for single GPU

### Allocation Types Available

1. **DECOUPLED_TRAIN** (`+` separator): Separate GPUs for training and inference
   - Example: `sglang.d1p1t1+d1p1t1`
   - **Problem**: Requires 2 GPUs but we only have 1

2. **COLOCATE** (`|` separator): Same GPUs shared between training and inference
   - Example: `sglang.d1p1t1|fsdp:d1p1t1` 
   - **Potentially works**: But need to verify if SGLang supports colocated mode

3. **Weight Update Types**:
   - `current_platform.communication_backend` (NCCL): Requires separate GPUs ❌
   - `"disk"`: Saves weights to disk, then loads them ✅ (might work!)

## Possible Solutions

### Solution 1: Use Disk-Based Weight Updates (Recommended)

Instead of distributed NCCL broadcast, use disk-based weight synchronization:

```yaml
# In WeightUpdateMeta configuration
# Change weight update type from NCCL to "disk"
```

However, I need to check if this is configurable or if it's hardcoded.

### Solution 2: Use Colocated Allocation Mode

Try using colocated mode instead of decoupled:

```yaml
allocation_mode: sglang.d1p1t1|fsdp:d1p1t1
```

But need to verify:
- Does SGLang support colocated mode?
- Does colocated mode avoid NCCL weight updates?

### Solution 3: Modify Weight Update Logic

Add a check in `fsdp_engine.py` to skip NCCL broadcast when:
- `world_size == 1` (only one inference server)
- `n_gpus_per_node == 1` (single GPU setup)

Use direct tensor copy or disk-based update instead.

### Solution 4: Use Gloo Backend

Force Gloo backend instead of NCCL for single-GPU:
```python
# In weight update initialization
backend = "gloo" if world_size == 1 and n_gpus_per_node == 1 else "nccl"
```

Gloo works with CPU/GPU and doesn't have the same GPU device restriction.

## Documentation Findings

- ❌ No explicit single-GPU examples in `examples/`
- ❌ No single-GPU troubleshooting guide
- ✅ Quickstart mentions single node (`cluster.n_nodes == 1`) but shows multi-GPU examples
- ✅ LocalLauncher supports single node but assumes multiple GPUs per node

## Next Steps

1. Check if `WeightUpdateMeta.type` can be set to `"disk"` 
2. Try colocated allocation mode: `sglang.d1p1t1|fsdp:d1p1t1`
3. Modify weight update code to detect single-GPU and use alternative method
4. Check if there's an environment variable to disable NCCL for single GPU

## Code Locations

- Weight update initialization: `areal/engine/fsdp_engine.py:332-357`
- Weight update broadcast: `areal/engine/fsdp_engine.py:298-331`
- Allocation mode parsing: `areal/api/alloc_mode.py:286-312`
- WeightUpdateMeta: `areal/api/io_struct.py:105-172`

