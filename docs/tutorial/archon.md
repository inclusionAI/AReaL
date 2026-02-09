# Archon: PyTorch-Native Training Engine

## Overview

Archon is AReaL's PyTorch-native training backend that provides maximum flexibility for
RL researchers without Megatron-Core dependencies. It supports full 5D parallelism (DP,
TP, PP, CP, EP) using PyTorch's native distributed primitives, making it easier to add
RL-specific optimizations and debug distributed training issues.

**Easy to get started**: Simply run `uv sync` to install all dependencies. Unlike
MegatronEngine which requires C++ compiled packages like `transformer_engine`, Archon
uses only pure Python packages with no complex build steps.

The design and core implementation of Archon are inspired by
[torchtitan](https://github.com/pytorch/torchtitan), PyTorch's official reference
implementation for large-scale LLM training. We thank the torchtitan team for their
excellent work in making distributed training accessible through pure PyTorch APIs.

## Engine Comparison

| Feature           | FSDPEngine          | MegatronEngine  | ArchonEngine                |
| ----------------- | ------------------- | --------------- | --------------------------- |
| Backend           | HuggingFace + FSDP2 | Megatron-Core   | PyTorch-native              |
| Model Source      | Any HF model        | Megatron models | Custom Archon models        |
| torch.compile     | Limited             | No              | Yes (default)               |
| Data Parallel     | FSDP2               | Megatron DP     | FSDP2                       |
| Tensor Parallel   | PyTorch DTensor     | Megatron TP     | PyTorch DTensor             |
| Pipeline Parallel | No                  | Yes (VPP)       | Yes (1F1B, Interleaved1F1B) |
| Expert Parallel   | No                  | Full EP/ETP     | Full EP/ETP                 |
| Context Parallel  | Ulysses SP          | Megatron CP     | Ulysses SP                  |
| Supported Models  | Any HF              | Via mbridge     | Built-in + User-defined     |
| Status            | Production          | Production      | Experimental                |

## Key Features

- **PyTorch-native implementation**: No Megatron-Core dependency, using only PyTorch
  distributed primitives (DTensor, DeviceMesh, FSDP2)
- **Full parallelism support**: DP, TP, PP, CP, EP, and ETP with flexible configuration
- **torch.compile by default**: Optimized performance with Inductor compilation
- **Flexible activation checkpointing**: Supports `none`, `full`, `selective`, and
  `memory_budget` modes
- **Native RL training support**: Built-in PPO Actor/Critic implementations
- **Pipeline parallel schedules**: 1F1B and Interleaved1F1B schedules

## Enabling Archon

To use Archon as your training backend, specify it in the `allocation_mode`:

```bash
allocation_mode=sglang:d4+archon:d4
```

### Supported Models

Archon provides built-in support for the following model types:

- `qwen2` - Qwen2 dense models
- `qwen3` - Qwen3 dense models
- `qwen3_moe` - Qwen3 MoE models

For unsupported models without custom implementations, use FSDPEngine or MegatronEngine
instead.

### Adding Custom Models

Users can add custom model implementations by creating a new model spec. The key
components are:

1. **Model class** (`nn.Module`): The model architecture implementation
1. **ModelArgs class**: Dataclass for model configuration, with `from_hf_config()`
   method to convert from HuggingFace config
1. **StateDictAdapter class**: Converts between HuggingFace and Archon weight formats
1. **Parallelize function**: Applies TP, CP, EP, FSDP, and activation checkpointing
1. **ModelSpec**: Registers all components together

Example structure (see `areal/experimental/models/archon/qwen3/` for reference):

```
areal/experimental/models/archon/your_model/
├── __init__.py
├── spec.py                    # ModelSpec registration
├── model/
│   ├── model.py               # Model class
│   ├── args.py                # ModelArgs dataclass
│   └── state_dict_adapter.py  # Weight conversion
└── infra/
    └── parallelize.py         # Parallelization logic
```

Register your model spec in `areal/experimental/models/archon/__init__.py`:

```python
from areal.experimental.models.archon.your_model import spec  # noqa: F401
```

> **Tip**: AI-powered coding tools (e.g., Claude Code, OpenCode) can help accelerate the
> process. Use the `/add-archon-model` skill for a semi-automated guide that analyzes
> HuggingFace source code and generates implementation scaffolding. See the
> [AI-Assisted Development Guide](../reference/ai_assisted_dev.md) for setup and usage.

## Parallelism Configuration

Archon uses the same parallelism syntax as Megatron. See
[Allocation Mode Reference](../reference/alloc_mode.md) for the complete syntax guide.

Basic example:

```bash
# Dense model: 4 DP × 2 PP × 2 TP = 16 GPUs
allocation_mode=sglang:d4t2+archon:d4p2t2
```

### MoE Support

Unlike FSDPEngine, Archon provides full MoE support with Expert Parallelism (EP) and
Expert Tensor Parallelism (ETP). For MoE models, you can use hybrid parallelism with
separate configurations for attention and FFN (expert) modules:

```bash
# MoE model with hybrid parallelism
allocation_mode=sglang:d4t4+archon:(attn:d1p4t2c2|ffn:d1p4t1e4)
```

This enables
[MoE Parallel Folding](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/transformer/moe#moe-parallel-folding),
reducing GPU requirements for combined context and expert parallelism.

## Advanced Configuration

Archon-specific options are configured under `actor.archon.*`:

| Option           | Default           | Description                                    |
| ---------------- | ----------------- | ---------------------------------------------- |
| `pp_schedule`    | `Interleaved1F1B` | Pipeline schedule: `1F1B` or `Interleaved1F1B` |
| `enable_compile` | `True`            | Enable torch.compile for TransformerBlocks     |
| `ac_mode`        | `selective`       | Activation checkpointing mode                  |
| `offload_params` | `False`           | Offload FSDP parameters to CPU                 |

See [Performance Tuning](#performance-tuning) for detailed guidance on these options.

## Performance Tuning

### torch.compile

Archon enables `torch.compile` by default for optimized performance. When compile is
enabled, `pad_to_maximum=True` is automatically set to avoid dynamic shape issues with
Inductor.

To disable compilation (useful for debugging or unsupported operations):

```bash
+actor.archon.enable_compile=False
```

### Activation Checkpointing Selection

Choose the appropriate AC mode based on your memory constraints:

| Mode            | Memory Usage | Recomputation | Use Case                                |
| --------------- | ------------ | ------------- | --------------------------------------- |
| `none`          | Highest      | None          | Small models, sufficient memory         |
| `selective`     | Medium       | Partial       | Default, balanced trade-off             |
| `full`          | Lowest       | All layers    | Large models, memory constrained        |
| `memory_budget` | Configurable | Auto-tuned    | Fine-grained control (requires compile) |

For `memory_budget` mode, adjust `ac_memory_budget` (0.0 = max recompute, 1.0 = no
recompute):

```bash
+actor.archon.ac_mode=memory_budget +actor.archon.ac_memory_budget=0.5
```

## Limitations

Current limitations of Archon Engine:

- **Weight tying not supported with PP**: Models with `tie_word_embeddings=True` cannot
  use Pipeline Parallelism (PP > 1) because embeddings and output layers are on
  different GPUs
- **Tree training**: Not yet supported (`enable_tree_training` will show a warning)
- **Experimental status**: APIs may change in future releases

## Debugging Tips

### Viewing Parallel Configuration

Archon logs parallel dimensions at initialization:

```
Initialized Archon engine with parallel dims: pp=2, dp_shard=4, tp=2, cp=1, ep=1, etp=1
```

### Common Issues

| Issue                              | Possible Cause                    | Solution                                        |
| ---------------------------------- | --------------------------------- | ----------------------------------------------- |
| Shape mismatch across microbatches | Variable sequence lengths with PP | Set `pad_to_maximum=True`                       |
| OOM during compilation             | torch.compile memory overhead     | Try `+actor.archon.enable_compile=False`        |
| "tie_word_embeddings" error        | PP with weight-tied model         | Use PP=1 or different model                     |
| Slow first iteration               | torch.compile warmup              | Expected behavior, subsequent iterations faster |

### Activation Checkpointing Debug

Enable AC debugging to capture detailed information (slower):

```bash
+actor.archon.ac_debug=True
```

## Migration from FSDPEngine

To migrate from FSDPEngine to Archon:

### 1. Update allocation_mode

```bash
# Before (FSDPEngine)
allocation_mode=sglang:d4t2+fsdp:d8t2

# After (Archon)
allocation_mode=sglang:d4t2+archon:d8t2
```

### 2. Configuration Mapping

| FSDPEngine Option        | Archon Equivalent             |
| ------------------------ | ----------------------------- |
| `gradient_checkpointing` | Same (controls AC globally)   |
| N/A                      | `actor.archon.ac_mode`        |
| N/A                      | `actor.archon.enable_compile` |
| N/A                      | `actor.archon.pp_schedule`    |

### 3. Model Compatibility

Ensure your model is supported by Archon (qwen2, qwen3, qwen3_moe) or implement a custom
model spec.

### 4. New Capabilities

With Archon, you gain access to:

- Pipeline Parallelism (`p` dimension)
- Expert Parallelism for MoE (`e` dimension)
- torch.compile optimization
- Flexible activation checkpointing modes

## Examples

### Dense Model (Qwen3-8B)

Create a config file `archon_qwen3_8b.yaml`:

```yaml
# Archon config for Qwen3-8B on 3 nodes (24 GPUs)
# SGLang: 4 replicas × 2 TP = 8 GPUs
# Archon: 4 DP × 2 PP × 2 TP = 16 GPUs

experiment_name: archon-gsm8k-grpo
trial_name: trial-0

cluster:
  n_nodes: 3
  n_gpus_per_node: 8

allocation_mode: sglang:d4t2+archon:d4p2t2

scheduler:
  type: ray

actor:
  path: Qwen/Qwen3-8B
  gradient_checkpointing: true
  archon:
    pp_schedule: Interleaved1F1B
    enable_compile: true
    ac_mode: selective
```

Run the experiment:

```bash
python3 examples/math/gsm8k_rl.py --config archon_qwen3_8b.yaml
```

### MoE Model (Qwen3-30B-A3B)

Create a config file `archon_qwen3_moe.yaml`:

```yaml
# Archon config for Qwen3-30B-A3B MoE on 4 nodes (32 GPUs)
# SGLang: 4 replicas × 4 TP = 16 GPUs
# Archon: 1 DP × 4 PP × (attn: TP2×CP2, ffn: TP1×EP4) = 16 GPUs

experiment_name: archon-moe-gsm8k-grpo
trial_name: trial-0

cluster:
  n_nodes: 4
  n_gpus_per_node: 8

allocation_mode: "sglang:d4t4+archon:(attn:d1p4t2c2|ffn:d1p4t1e4)"

scheduler:
  type: ray

actor:
  path: Qwen/Qwen3-30B-A3B
  gradient_checkpointing: true
  archon:
    pp_schedule: Interleaved1F1B
    enable_compile: true
    ac_mode: selective
```

Run the experiment:

```bash
python3 examples/math/gsm8k_rl.py --config archon_qwen3_moe.yaml
```

## See Also

- [Allocation Mode Reference](../reference/alloc_mode.md) - Complete guide on allocation
  mode syntax
- [Fine-tuning Large MoE Models](megatron.md) - MegatronEngine alternative for MoE
  models
