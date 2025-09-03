# Troubleshooting Out-of-Memory (OOM) Issues

Out-of-Memory (OOM) errors are one of the most common challenges when running
reinforcement learning training at scale. This guide provides practical solutions to
diagnose and resolve OOM issues across all stages of your AReaL workflow: generation,
training, and weight updates.

## Understanding Memory Usage

Before diving into solutions, it's essential to understand which parameters most
significantly impact memory consumption in AReaL:

### Core Parameters

- **`allocation_mode`**: Controls how inference and training workloads are distributed
  across GPUs. Different parallelism strategies have vastly different memory
  requirement. For example, tensor parallelism typically uses less memory per GPU than
  data parallelism for large models.

- **`train_dataset.max_length`**: The maximum prompt length in your training dataset.
  Longer prompts require more memory during both generation and training phases.

- **`gconfig.max_new_tokens`**: Maximum tokens to generate per prompt. Combined with
  `max_length`, this determines the total sequence length and significantly impacts
  memory usage.

- **`actor.mb_spec.max_tokens_per_mb`**: The maximum tokens processed per micro-batch
  during forward/backward passes. This is your primary lever for controlling training
  memory usage. The minimum safe value is
  `train_dataset.max_length + gconfig.max_new_tokens`.

- **`max_concurrent_rollouts`**: Number of parallel generation requests sent to
  inference servers. Higher values improve throughput but consume more memory.

### Engine-Specific Parameters

- **Inference Engine**: Parameters like `sglang.mem_fraction_static` control how SGLang
  allocates GPU memory. See the [SGLang Documentation](https://docs.sglang.ai/) for
  detailed tuning options.

- **Training Engine**: FSDP sharding strategies and other PyTorch settings affect
  training memory usage. Refer to the
  [FSDP Documentation](https://docs.pytorch.org/docs/stable/fsdp.html) for advanced
  configuration.

::::{note} The `train_dataset.batch_size` parameter doesn't directly impact peak memory
usage. Focus on the parameters listed above when troubleshooting OOM issues. ::::

## Resolving Generation OOM Errors

If you encounter OOM errors during the inference/generation phase (typically visible in
`llm_server.log`), try these solutions in order of effectiveness:

### 1. Reduce Concurrent Rollouts (Most Effective)

Lower the `max_concurrent_rollouts` parameter to reduce the number of parallel
generation requests:

```yaml
max_concurrent_rollouts: 200  # Try reducing from default values like 256
```

This is usually the most effective solution as it directly reduces the memory pressure
on inference servers.

### 2. Adjust Parallelism Strategy

Modify your `allocation_mode` to increase tensor parallelism, which distributes model
weights across more GPUs:

```yaml
# Before: sglang.d4+d4 (4 data parallel processes)
# After: sglang.d2t2+d4 (2 data parallel, 2 tensor parallel)
allocation_mode: sglang.d2t2+d4
```

However, a larger tensor parallelism degree will harm generation throughput.

### 3. Tune SGLang Parameters

Fine-tune memory allocation for the SGLang inference engine:

```yaml
sglang:
  mem_fraction_static: 0.8  # Reduce from 0.9 to leave more memory headroom
```

For more advanced SGLang tuning options, consult the
[SGLang Documentation](https://docs.sglang.ai/).

## Resolving Training OOM Errors

OOM errors during the training phase require different strategies focused on reducing
the memory footprint of gradient computation and model updates.

### 1. Optimize Micro-batch Size

Set `max_tokens_per_mb` to its minimum safe value to reduce memory usage per training
step:

```yaml
actor:
  mb_spec:
    max_tokens_per_mb: 4096  # train_dataset.max_length + gconfig.max_new_tokens
```

For multi-turn conversations, calculate this as:

```
max_tokens_per_mb = <longest_conversation_length> + gconfig.max_new_tokens
```

This value heavily depends on your `RolloutWorkflow` implementation.

### 2. Enable Ulysses Sequence Parallelism

When context lengths are very long and you cannot reduce `max_tokens_per_mb` further,
enable Ulysses sequence parallelism to distribute long sequences across multiple GPUs:

```yaml
fsdp:
  ulysses_sp_size: 2  # or 4, 8 depending on your setup
```

::::{important} `ulysses_sp_size` must be a common divisor of both your FSDP parallelism
degree and the model's number of attention heads.

For example, with 40 attention heads and allocation mode `d32`:

- Valid values: `1, 2, 4, 8`
- Invalid values: `16, 32` (exceed attention head constraints) ::::

## Resolving Weight Update OOM Errors

Weight updates can consume significant additional memory, especially when using
NCCL-based synchronization (the default method).

### 1. Switch to Disk-Based Updates

Replace memory-intensive NCCL updates with disk-based weight synchronization:

```python
# Instead of NCCL-based updates
weight_update_meta = WeightUpdateMeta.from_disk(config.saver)
```

See the [Weight Updates Guide](../lite/gsm8k_grpo.md) (specifically the "Transferring
Weights to Inference Servers" section) for detailed implementation instructions.

### 2. Reduce Memory Buffer Size

If staying with NCCL updates, reduce the memory buffer used for weight chunking:

```python
# In WeightUpdateMeta.from_fsdp_nccl() calls
WeightUpdateMeta.from_fsdp_nccl(
    ...,
    weight_chunked_mem_mb = 512,  # Reduce from default (typically 1024+)
)
```
