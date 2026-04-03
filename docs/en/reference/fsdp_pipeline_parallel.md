# FSDP2 Pipeline Parallelism

This document describes the FSDP2 Pipeline Parallelism (PP) support in AReaL, which
extends the FSDP training backend from 3D parallelism (DP x SP x TP) to 4D parallelism
(PP x DP x SP x TP).

## Overview

Pipeline parallelism splits a model's layers across multiple GPU groups (called
**stages**), where each stage processes a different subset of layers. During training,
microbatches flow through stages in a pipeline, overlapping forward and backward passes
to maximize throughput.

AReaL's FSDP2 Pipeline Parallelism builds on PyTorch's native
`torch.distributed.pipelining` API and adapts it for HuggingFace model architectures.
This enables users who already rely on the FSDP backend to scale to larger models
without switching to Megatron or Archon.

### Why FSDP2 + PP?

Without PP, the FSDP backend supports data parallelism (DP), sequence/context
parallelism (SP), and tensor parallelism (TP). While these are effective for many
workloads, they have limitations:

- **DP** replicates the full model on each rank — memory-bound for very large models.
- **TP** splits individual operations across GPUs, requiring high-bandwidth
  interconnects (NVLink) and incurring communication overhead.
- **SP** splits sequence length, but is bounded by attention head count divisibility.

Pipeline parallelism complements these by splitting **layers** across stages with
minimal communication (only activation tensors are sent between adjacent stages). This
makes PP particularly effective for scaling across nodes where inter-node bandwidth is
lower.

## Architecture

### 4D Device Mesh

When PP is enabled, the device mesh layout becomes:

```
("pp", "dp", "sp", "tp")
```

This mirrors torchtitan's approach where PP is the outermost dimension. The total GPU
count for a training job is:

```
world_size = pp × dp × sp × tp
```

For example, with `fsdp:d4p2t2` on 16 GPUs:

```
pp=2, dp=4, sp=1, tp=2
world_size = 2 × 4 × 1 × 2 = 16
```

### How PP Integrates with FSDP2

The integration follows this sequence during initialization:

1. **Model splitting**: The HuggingFace model is split into pipeline stages by
   distributing transformer layers across stages. Embedding and output layers are
   assigned to the first and last stages, respectively.

2. **Per-stage parallelization**: Each model part (stage) independently receives TP +
   FSDP2 sharding. This means each stage is fully sharded and tensor-parallelized
   within its own DP x SP x TP submesh.

3. **Schedule construction**: A pipeline schedule (e.g., 1F1B, Interleaved1F1B)
   orchestrates how microbatches flow through stages during training.

4. **Runner creation**: An `FSDPPipelinedRunner` wraps the schedule and handles
   forward/backward execution for both training and evaluation.

### Key Optimization: `reshard_after_forward=False`

When PP is enabled, FSDP2 is configured with `reshard_after_forward=False`. In standard
FSDP2 (without PP), parameters are resharded (freed) after each forward pass to save
memory. However, with PP, multiple microbatches pass through the same stage in
succession. If parameters were resharded after each microbatch's forward pass, they
would need to be all-gathered again for the next microbatch — a significant
communication overhead.

By keeping parameters gathered after the forward pass, each stage avoids repeated
all-gathers across microbatches. This is a critical optimization adopted from torchtitan
and verl for FSDP2 + PP workloads.

> **Trade-off**: This increases peak memory usage per stage (parameters remain
> unsharded), but avoids redundant communication. See
> [Handling OOM](../best_practices/handling_oom.md) for memory tuning guidance.

## Configuration

### Backend String

To enable PP with the FSDP backend, add the `p` dimension to the backend string:

```yaml
actor:
  backend: "fsdp:d4p2"       # 4 DP × 2 PP = 8 GPUs
```

You can combine PP with TP and SP:

```yaml
actor:
  backend: "fsdp:d2p2t2"     # 2 DP × 2 PP × 2 TP = 8 GPUs
```

### FSDPEngineConfig PP Fields

All PP-specific fields live under the `fsdp` section of the actor/critic config:

#### `pp_schedule`

Pipeline schedule type. Default: `"Interleaved1F1B"`.

| Schedule                 | Stages/Rank | Description                                             |
| ------------------------ | ----------- | ------------------------------------------------------- |
| `1F1B`                   | 1           | Classic one-forward-one-backward schedule.               |
| `Interleaved1F1B`        | >= 2        | Interleaved schedule with multiple virtual stages/rank. |
| `InterleavedZeroBubble`  | >= 2        | Interleaved schedule with zero-bubble optimization.      |
| `ZBVZeroBubble`          | 2 (V-style) | V-shaped zero-bubble schedule (stages assigned in V).    |

**Single-stage schedules** (`1F1B`) assign exactly 1 virtual stage per PP rank.
**Multi-stage schedules** (`Interleaved1F1B`, `InterleavedZeroBubble`, `ZBVZeroBubble`)
assign 2 or more virtual stages per rank for better pipeline utilization.

**V-style schedules** (`ZBVZeroBubble`) assign exactly 2 stages per rank in a V-shaped
pattern: rank 0 gets stages (0, N-1), rank 1 gets stages (1, N-2), etc. This balances
the forward and backward pass computation.

#### `pp_layers_per_stage`

Number of transformer layers per virtual pipeline stage. Default: `None`.

- If set, the number of virtual stages is calculated as
  `ceil((num_layers + first_less + last_less) / pp_layers_per_stage)`.
- The resulting number of virtual stages must be evenly divisible by the PP degree.
- If `None`, the number of stages per rank is inferred from the schedule type: 1 for
  `1F1B`, 2 for interleaved/ZBV schedules.

#### `pp_first_stage_less_layers`

Number of equivalent layers to subtract from the first stage to account for the
embedding layer overhead. Default: `1`.

Since the first stage also hosts `model.embed_tokens`, it gets fewer transformer layers
to balance compute across stages.

#### `pp_last_stage_less_layers`

Number of equivalent layers to subtract from the last stage to account for the output
head overhead. Default: `1`.

Since the last stage also hosts `model.norm` and `lm_head` (or `score` for critics), it
gets fewer transformer layers.

### Example: Layer Distribution

For a model with 28 transformer layers and `pp_size=4` (default `first_less=1`,
`last_less=1`):

```
Effective layers = 28 + 1 + 1 = 30
Layers per stage = 30 / 4 = 7 (+ 2 remainder stages get 8)

Stage 0: embed_tokens + 7 layers  (layers 0-6)   — 1 less for embed
Stage 1: 8 layers                 (layers 7-14)
Stage 2: 8 layers                 (layers 15-22)
Stage 3: 5 layers + norm + lm_head (layers 23-27) — 1 less for output
```

## Supported PP Schedules

### 1F1B (One Forward One Backward)

The classic pipeline schedule. Each rank holds exactly 1 stage. Microbatches enter the
pipeline, and once the pipeline is full, each rank alternates between one forward and one
backward step.

- **Pros**: Simple, low memory footprint (only 1 stage per rank).
- **Cons**: Pipeline bubble at warmup and cooldown phases.

```yaml
actor:
  fsdp:
    pp_schedule: "1F1B"
```

### Interleaved1F1B

Each rank holds multiple virtual stages (default 2). Microbatches are interleaved across
virtual stages, reducing the pipeline bubble.

- **Pros**: Reduced bubble compared to 1F1B.
- **Cons**: Higher memory (multiple stages per rank), more complex scheduling.

```yaml
actor:
  fsdp:
    pp_schedule: "Interleaved1F1B"
```

### InterleavedZeroBubble

An interleaved schedule with zero-bubble optimization that further reduces idle time by
overlapping backward computation of one microbatch with forward computation of another.

- **Pros**: Near-zero pipeline bubble.
- **Cons**: Higher memory due to retained activations; may use `retain_graph`.

```yaml
actor:
  fsdp:
    pp_schedule: "InterleavedZeroBubble"
```

### ZBVZeroBubble

A V-shaped zero-bubble schedule where each rank holds exactly 2 stages assigned in a
V-pattern (rank *i* gets stages *i* and *N-1-i*). This provides excellent load balancing
and minimal pipeline bubble.

- **Pros**: Best bubble reduction, balanced compute per rank.
- **Cons**: Requires exactly 2 stages per rank; higher memory.

```yaml
actor:
  fsdp:
    pp_schedule: "ZBVZeroBubble"
```

## Full Configuration Example

A 32-GPU setup training a 70B model with 4D parallelism:

```yaml
rollout:
  backend: "sglang:d4t4"       # 4 × 4 = 16 GPUs for inference

actor:
  backend: "fsdp:d2p2t4"       # 2 DP × 2 PP × 4 TP = 16 GPUs for training
  gradient_checkpointing: true
  fsdp:
    pp_schedule: "Interleaved1F1B"
    pp_layers_per_stage: null   # Auto: 2 virtual stages per PP rank
    pp_first_stage_less_layers: 1
    pp_last_stage_less_layers: 1
    memory_efficient_load: true
```

A minimal 8-GPU setup with PP only (no TP):

```yaml
rollout:
  backend: "sglang:d4"          # 4 GPUs for inference

actor:
  backend: "fsdp:d2p2"          # 2 DP × 2 PP = 4 GPUs for training
  fsdp:
    pp_schedule: "1F1B"
```

## Limitations and Known Issues

- **No PP + EP combination**: FSDP Engine does not support combining pipeline
  parallelism with expert parallelism. Use the Archon Engine for PP + EP workloads.

- **HuggingFace model layout required**: The PP implementation assumes the standard
  HuggingFace model structure (`model.embed_tokens`, `model.layers.*`, `model.norm`,
  `lm_head`/`score`). Custom model architectures may require adaptation.

- **Microbatch count**: The number of microbatches should be >= the total number of
  virtual stages (`num_virtual_stages = stages_per_rank × pp_degree`) to avoid
  excessive pipeline bubbles. A warning is logged when this condition is not met.

- **`reshard_after_forward` is forced to `False`**: When PP is enabled, FSDP parameters
  are kept unsharded after forward to avoid repeated all-gathers. This increases memory
  usage compared to non-PP FSDP.

- **Vision-Language Models**: PP support is designed for decoder-only LLMs. VLM support
  with PP has not been validated.

- **Per-layer optim step**: `per_layer_optim_step` and PP have not been jointly
  validated. Use with caution.

## See Also

- [Allocation Mode Reference](alloc_mode.md) — Backend string syntax and GPU allocation
- [Handling OOM Issues](../best_practices/handling_oom.md) — Memory tuning for training
- [Archon: PyTorch-Native Training Engine](../tutorial/archon.md) — Alternative backend
  with PP + EP support
