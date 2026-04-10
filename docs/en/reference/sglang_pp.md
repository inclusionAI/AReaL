# SGLang Pipeline Parallelism (PP) Support

This document describes AReaL's support for pipeline parallelism (PP) in the SGLang
inference backend. PP allows splitting model layers across multiple GPUs in stages,
enabling inference of models that exceed the memory capacity of a single GPU.

## Overview

Pipeline parallelism partitions a model's layers into sequential stages, each placed on
a separate GPU. During inference, micro-batches flow through the pipeline stages. AReaL
extends its SGLang integration to support PP alongside the existing data parallelism (DP)
and tensor parallelism (TP) dimensions.

The key challenge with PP in an RL training loop is **weight synchronization**: after each
training step, the updated model weights must be transferred from the Megatron training
backend to the SGLang inference servers. When PP is enabled, each pipeline stage holds
only a subset of layers, so the weight update must be coordinated per stage.

AReaL solves this with **per-PP-rank NCCL groups** --- each pipeline stage forms its own
NCCL communication group between the corresponding training rank and all inference
workers at that stage.

## Architecture: Per-PP-Rank NCCL Groups

### Without PP (PP=1)

When PP is not used, a single NCCL group connects the training rank to all inference
workers:

```
Training Rank 0                     SGLang Workers
(all layers)                        (all layers)
      │                                  │
      └──── NCCL Group ─────────────────┘
            (all parameters)
```

### With PP (PP=2 Example)

When PP>1, each training PP rank creates a separate NCCL group. Only the SGLang workers
holding the same pipeline stage participate in that group:

```
Training PP Rank 0                  SGLang PP Rank 0
(layers 0..N/2)                     (layers 0..N/2)
      │                                  │
      └── NCCL Group 0 ─────────────────┘
          (group_name: "update_weight_group_0")

Training PP Rank 1                  SGLang PP Rank 1
(layers N/2..N)                     (layers N/2..N)
      │                                  │
      └── NCCL Group 1 ─────────────────┘
          (group_name: "update_weight_group_1")
```

Each per-PP-rank group has:

- **World size** = `n_servers * tp_size + 1` (all TP workers across all DP instances at
  this PP rank, plus one training rank)
- **Rank offset** based on `tp_size` only (not `tp_size * pp_size`), because only workers
  at one PP rank participate
- A `pp_rank` field in the initialization payload so SGLang knows which workers should
  join

## How It Works

The following diagram shows the end-to-end flow when PP is enabled:

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │                        Training Step                                │
 │                                                                     │
 │  1. Rollout Phase                                                   │
 │     Controller ──> SGLang servers generate trajectories             │
 │     (PP stages process sequentially within each server)             │
 │                                                                     │
 │  2. Training Phase                                                  │
 │     Controller ──> Megatron workers (PP pipeline schedule)          │
 │     Forward: Stage 0 → Stage 1 → ... → Stage K                     │
 │     Backward: Stage K → ... → Stage 1 → Stage 0                    │
 │     Optimizer step on each stage                                    │
 │                                                                     │
 │  3. Weight Sync Phase                                               │
 │     a. Pause SGLang servers                                         │
 │     b. For each PP rank r = 0, 1, ..., K:                           │
 │        Training PP rank r ──NCCL broadcast──> SGLang PP rank r      │
 │        (group: "update_weight_group_{r}")                           │
 │     c. Resume SGLang servers                                        │
 │                                                                     │
 │  4. Version bump                                                    │
 │     Update model version on both training and inference sides       │
 └─────────────────────────────────────────────────────────────────────┘
```

### Weight Update Initialization

During setup, AReaL calls `init_weights_update_group` on each SGLang server for every PP
rank. The request payload includes:

| Field            | Description                                          |
| ---------------- | ---------------------------------------------------- |
| `master_address` | NCCL master address from the training side           |
| `master_port`    | NCCL master port                                     |
| `rank_offset`    | `1 + server_idx * tp_size`                           |
| `world_size`     | `n_servers * tp_size + 1`                            |
| `group_name`     | `"update_weight_group_{pp_rank}"`                    |
| `pp_rank`        | Pipeline stage index (only present when PP>1)        |

### Weight Update Transfer

During each weight sync, the training side broadcasts parameters layer by layer. Each
training PP rank broadcasts only the layers it owns, using its corresponding NCCL group.
This is handled transparently by the existing `update_weights_from_distributed` path in
`RolloutController`.

## Configuration Guide

### Backend String Format

Pipeline parallelism is specified using the `p` dimension in the backend string:

```
sglang:d<DP>p<PP>t<TP>
megatron:d<DP>p<PP>t<TP>
```

The total GPU count for each engine is `DP * PP * TP`.

### Example Configurations

#### 8 GPUs: DP=2, PP=2, TP=1

```yaml
rollout:
  backend: "sglang:d2p2t1"    # 2 × 2 × 1 = 4 GPUs

actor:
  backend: "megatron:d2p2t1"   # 2 × 2 × 1 = 4 GPUs
```

Two SGLang server instances, each spanning 2 GPUs (2 pipeline stages). Four Megatron
workers with 2 data-parallel groups, each having a 2-stage pipeline.

#### 8 GPUs: DP=1, PP=2, TP=2

```yaml
rollout:
  backend: "sglang:d1p2t2"    # 1 × 2 × 2 = 4 GPUs

actor:
  backend: "megatron:d1p2t2"   # 1 × 2 × 2 = 4 GPUs
```

One SGLang server instance spanning 4 GPUs (2 pipeline stages, each with 2 TP shards).
One Megatron group with the same layout.

#### 16 GPUs: DP=2, PP=2, TP=2

```yaml
cluster:
  n_nodes: 1
  n_gpus_per_node: 16

rollout:
  backend: "sglang:d2p2t2"    # 2 × 2 × 2 = 8 GPUs

actor:
  backend: "megatron:d2p2t2"   # 2 × 2 × 2 = 8 GPUs
```

### Supported Configuration Matrix

The following table shows tested DP, PP, and TP combinations. The rollout and actor
backends must use matching PP and TP values so that weight update groups align correctly.

| DP  | PP  | TP  | GPUs per Engine | Notes                     |
| --- | --- | --- | --------------- | ------------------------- |
| 1   | 2   | 1   | 2               | Minimum PP config         |
| 2   | 2   | 1   | 4               | PP with data parallelism  |
| 1   | 2   | 2   | 4               | PP with tensor parallelism|
| 2   | 2   | 2   | 8               | Full 3D parallelism       |
| 1   | 4   | 1   | 4               | Deeper pipeline           |
| 2   | 4   | 2   | 16              | Large-scale config        |

> **Constraint**: The PP and TP sizes in `rollout.backend` must match those in
> `actor.backend`. DP sizes may differ between rollout and actor, but in most cases they
> are kept equal.

## Prerequisites

### SGLang PP Patch

SGLang's pipeline parallelism support for weight update groups requires a patched version
of SGLang. The patch adds the ability for SGLang workers to:

1. Accept a `pp_rank` parameter in the `/init_weights_update_group` endpoint
2. Filter which workers join a given NCCL group based on their PP rank
3. Handle per-PP-rank weight updates during `/update_weights` calls

Ensure you are using a version of SGLang that includes PP support. Refer to the AReaL
installation guide for compatible SGLang versions:

```bash
# Check SGLang version
python -c "import sglang; print(sglang.__version__)"
```

### Megatron-LM

The Megatron training backend must be configured with matching pipeline parallelism.
AReaL's Megatron integration handles PP natively. No additional patches are required for
the training side.

## Running with PP Enabled

### Using the Example Config

```bash
# Local launch (8 GPUs)
python examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_grpo_megatron_pp.yaml \
    scheduler.type=local

# Ray cluster
python examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_grpo_megatron_pp.yaml \
    scheduler.type=ray

# Slurm cluster
python examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_grpo_megatron_pp.yaml \
    scheduler.type=slurm
```

### Overriding PP on the Command Line

You can enable PP on any existing config by overriding the backend strings:

```bash
python examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_grpo_megatron.yaml \
    rollout.backend="sglang:d2p2t1" \
    actor.backend="megatron:d2p2t1" \
    scheduler.type=local
```

### Running the Test Config

For CI or quick validation:

```bash
python examples/math/gsm8k_rl.py \
    --config examples/test/gsm8k_grpo_megatron_pp_test.yaml \
    scheduler.type=local
```

This uses Qwen3-0.6B with small batch sizes and sequence lengths for fast iteration.

## Troubleshooting

### NCCL Group Initialization Fails

**Symptom**: Timeout or error during `init_weights_update_group`.

**Possible causes**:

- SGLang version does not support the `pp_rank` parameter. Ensure you are using a
  PP-compatible build.
- Network connectivity issues between training and inference GPU processes. Verify that
  NCCL can communicate across all GPUs.
- Firewall blocking the NCCL master port. Check that `master_port` is accessible.

**Debug steps**:

```bash
# Enable NCCL debug logging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

### Weight Update Hangs

**Symptom**: Training hangs during the weight sync phase after the first training step.

**Possible causes**:

- Mismatched PP sizes between rollout and actor backends. Both must use the same PP
  value.
- Incorrect world size calculation. Each per-PP-rank group expects exactly
  `n_servers * tp_size + 1` participants.

**Debug steps**:

1. Verify that `rollout.backend` and `actor.backend` have matching `p` values.
2. Check logs for the NCCL group name and world size being used.
3. Ensure all SGLang server processes are healthy (`/health` endpoint).

### Out of Memory

**Symptom**: CUDA OOM during inference or training.

**Possible causes**:

- `mem_fraction_static` too high for the per-GPU memory available with PP.
- Context length too large for the reduced per-GPU model size.

**Solutions**:

- Reduce `sglang.mem_fraction_static` (e.g., from 0.8 to 0.7).
- Reduce `sglang.context_length`.
- Increase PP to spread layers across more GPUs.

### Performance Considerations

Pipeline parallelism introduces bubble overhead --- time when some stages are idle waiting
for micro-batches. For inference this is generally acceptable, but be aware of:

- **Latency**: PP increases per-request latency compared to TP-only configurations due to
  sequential stage processing.
- **Throughput**: With sufficient request concurrency (continuous batching), PP can still
  achieve high throughput.
- **Memory**: PP reduces per-GPU memory usage, enabling larger models or longer context
  lengths.

When choosing between PP and TP, prefer TP for models that fit within the aggregate
memory of TP GPUs. Use PP when the model requires more GPUs than TP alone can provide, or
when cross-node communication bandwidth favors PP's point-to-point pattern over TP's
all-reduce.

## See Also

- [Allocation Mode](alloc_mode.md) - Full reference for backend strings and parallelism
  dimensions
- [Running GRPO on GSM8K](../tutorial/gsm8k_grpo.md) - Tutorial for the base GRPO
  workflow
- [Fine-tuning Large Models with Megatron](../tutorial/megatron.md) - Megatron backend
  tutorial
