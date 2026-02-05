# Allocation Mode

This document describes AReaL's allocation mode system, which controls how GPUs are
distributed between inference and training backends during distributed RL training.

## Overview

The `allocation_mode` configuration option is a pattern-based string that specifies:

- Which backends to use for inference (SGLang, vLLM) and training (FSDP, Megatron,
  Archon)
- The parallelization strategy for each backend
- The total number of GPUs required

AReaL parses this string into an `AllocationMode` object that orchestrates resource
allocation across the cluster.

## Syntax

### Basic Format

```
<backend>:<parallelism_dims>
```

### Two-Component Format (Inference + Training)

```
<inference_backend>:<dims> + <training_backend>:<dims>
```

The `+` operator separates components that run on **separate GPU pools**.

### Parallelism Dimensions

| Dimension | Abbreviation | Description                        | Valid For        |
| --------- | ------------ | ---------------------------------- | ---------------- |
| Data      | `d`          | Number of model replicas           | All backends     |
| Tensor    | `t`          | Split operations across GPUs       | All backends     |
| Pipeline  | `p`          | Split layers across GPUs in stages | Megatron, Archon |
| Context   | `c`          | Split sequence length across GPUs  | All backends     |
| Expert    | `e`          | Split MoE experts across GPUs      | Megatron, Archon |

Dimensions are specified as `<abbrev><size>`, e.g., `d4t2` means data parallel size 4
and tensor parallel size 2.

## Calculating GPU Requirements

The total GPUs for a component is computed as:

```
world_size = dp × tp × pp × cp
```

Expert parallelism (`e`) does not increase world size—it redistributes how experts are
placed within the existing GPU mesh.

### Examples

| Allocation Mode                   | Inference GPUs | Training GPUs | Total |
| --------------------------------- | -------------- | ------------- | ----- |
| `d8`                              | -              | 8             | 8     |
| `sglang:d2t4`                     | 8              | -             | 8     |
| `sglang:d2t4 + fsdp:d4t2`         | 8              | 8             | 16    |
| `sglang:d4t4 + megatron:d2p2t4e4` | 16             | 16            | 32    |

## Backend Selection

### Inference Backends

| Backend  | Supported Dimensions |
| -------- | -------------------- |
| `sglang` | `d`, `t`             |
| `vllm`   | `d`, `t`, `p`        |

For inference, `d` represents the number of independent server instances, and each
instance uses `t × p` GPUs.

Note that the internal backed configurations do not affect how AReaL allocate GPUs.
Given allocation mode `sglang:d4t4`, you can also config `sglang.dp_size=4`,
`sglang.ep_size=4`, and `sglang.enable_dp_attention=True`. In this case, we launch 4
model replicas each with 4 GPUs. Within each instance, SGLang will still use DP
attention and expert parallelism to distribute computations in attention and expert
layers.

### Training Backends

| Backend    | Supported Dimensions    | Use Case                                 |
| ---------- | ----------------------- | ---------------------------------------- |
| `fsdp`     | `d`, `t`, `c`           | Default for simple parallelism           |
| `megatron` | `d`, `t`, `p`, `c`, `e` | Required for pipeline or expert parallel |
| `archon`   | `d`, `t`, `p`, `c`, `e` | Alternative to Megatron (experimental)   |

When the backend is omitted, AReaL auto-selects based on the parallelism configuration:

- **FSDP**: Used when only `d`, `t`, `c` are specified
- **Megatron**: Used when `p > 1` or `e > 1`

```
# Equivalent forms
d4t2           # Auto-selects FSDP
fsdp:d4t2      # Explicit FSDP

d2p2t4         # Auto-selects Megatron (pp > 1)
megatron:d2p2t4  # Explicit Megatron
```

## MoE Hybrid Parallelism

For Mixture-of-Experts models, Megatron/Archon supports different parallelism strategies
for attention and FFN (expert) modules using the hybrid syntax:

```
megatron:(attn:<attn_dims>|ffn:<ffn_dims>)
```

This enables
[MoE Parallel Folding](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/transformer/moe#moe-parallel-folding),
which reduces the minimum GPU requirement for combined context and expert parallelism.

### Constraints

- Pipeline parallel size (`p`) must be identical for `attn` and `ffn`
- World size must match (if `d` is omitted in `ffn`, it is derived automatically)
- Expert parallel (`e`) is only valid in the `ffn` section

### Example

```
megatron:(attn:d4p2t2c2|ffn:d2p2t4e2)
```

| Module | dp  | pp  | tp  | cp  | ep  | World Size |
| ------ | --- | --- | --- | --- | --- | ---------- |
| attn   | 4   | 2   | 2   | 2   | -   | 32         |
| ffn    | 2   | 2   | 4   | -   | 2   | 32         |

## See Also

- [Fine-tuning Large MoE Models](../tutorial/megatron.md) - Tutorial for Megatron
  backend
- [Archon: PyTorch-Native Training Engine](../tutorial/archon.md) - Tutorial for Archon
  backend
- [Megatron Performance Best Practice](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/transformer/moe#performance-best-practice)
