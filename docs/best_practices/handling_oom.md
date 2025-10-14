# Handling OOM Issues

OOM errors are pretty common when you're doing large-scale RL training. Here's how to
tackle them across generation, training, and weight updates in your AReaL workflows.

## Understanding Memory Usage

Before jumping into fixes, let's understand which parameters actually matter for memory
usage:

### Core Parameters

- **`allocation_mode`**: How you split inference and training across GPUs. For large
  models, tensor parallelism typically uses less memory per GPU than data parallelism.

- **`train_dataset.max_length`**: Your maximum prompt length. Longer prompts = more
  memory.

- **`gconfig.max_new_tokens`**: How many tokens to generate per prompt. This plus
  `max_length` gives you your total sequence length.

- **`actor.mb_spec.max_tokens_per_mb`**: Tokens per micro-batch during forward/backward
  passes. This is your main knob for controlling training memory. Can't go below
  `max_length + max_new_tokens`.

- **`max_concurrent_rollouts`**: How many generation requests you run in parallel. More
  requests = better throughput but higher memory usage.

### Engine-Specific Parameters

- **Inference Engine**: `sglang.mem_fraction_static` controls how much GPU memory SGLang
  uses. Check the [SGLang docs](https://docs.sglang.ai/) for more tuning options.

- **Training Engine**: FSDP sharding and other PyTorch settings also impact memory
  usage. The [FSDP docs](https://docs.pytorch.org/docs/stable/fsdp.html) have more
  details.

> Don't worry about `train_dataset.batch_size` - it doesn't actually affect peak memory
> usage. Stick to the parameters above when troubleshooting OOM issues.

## Resolving Generation OOM Errors

When you hit generation OOM errors (you'll see them in `llm_server.log`), here's what to
try:

### 1. Reduce Concurrent Rollouts (Most Effective)

Lower the number of parallel generation requests:

```yaml
max_concurrent_rollouts: 200  # Try reducing from default values like 256
```

This is usually your best bet since it directly reduces memory pressure on the inference
servers.

### 2. Adjust Parallelism Strategy

Try increasing tensor parallelism to spread your model weights across more GPUs:

```yaml
# Before: sglang:d4+fsdp:d4 (4 data parallel processes)
# After: sglang:d2t2+fsdp:d4 (2 data parallel, 2 tensor parallel)
allocation_mode: sglang:d2t2+fsdp:d4
```

Just keep in mind that higher tensor parallelism will slow down your generation
throughput.

### 3. Tune SGLang Parameters

You can also tweak how SGLang allocates memory:

```yaml
sglang:
  mem_fraction_static: 0.8  # Reduce from 0.9 to leave more memory headroom
```

Check out the [SGLang docs](https://docs.sglang.ai/) for more advanced tuning options.

## Resolving Training OOM Errors

Training OOM errors are trickier - you need to reduce the memory footprint of gradient
computation and model updates.

### 1. Optimize Micro-batch Size

Your first move: set `max_tokens_per_mb` as low as safely possible:

```yaml
actor:
  mb_spec:
    max_tokens_per_mb: 4096  # train_dataset.max_length + gconfig.max_new_tokens
```

For multi-turn conversations, calculate it like this:

```
max_tokens_per_mb = <longest_conversation_length> + gconfig.max_new_tokens
```

The exact value will depend on how your `RolloutWorkflow` is implemented.

### 2. Enable Ulysses Sequence Parallelism

If you're dealing with really long contexts and can't reduce `max_tokens_per_mb` any
further, try Ulysses sequence parallelism to spread sequences across multiple GPUs:

```yaml
# Before: sglang:d4+fsdp:d4 (4 data parallel processes)
# After: sglang:d4+fsdp:d2c2 (2 data parallel, 2 ulysses context parallel)
allocation_mode: sglang:d4+fsdp:d2c2
```

> Just remember: Ulysses context parallel size needs to divide evenly into your model's
> attention heads.
>
> For example, with 40 attention heads:
>
> - These work: `1, 2, 4, 8`
> - These don't: `16, 32`

### 3. Switch to a Lightweight Optimizer

Depending on the training engine, AReaL supports different optimizers.

| Optimizer       | FSDP | Megatron | Name      |
| --------------- | ---- | -------- | --------- |
| AdamW (default) | ✅   | ✅       | adam      |
| SGD             | ✅   | ✅       | sgd       |
| AdamW_bf16      | ✅   | ❌       | adam_bf16 |

When encountering an OOM error, you can switch to a more memory-efficient optimizer. `SGD` and `AdamW_bf16` are more lightweight than the default `AdamW`. You can switch by setting `actor.optimizer.type: <name>` in your YAML configuration file (e.g., `actor.optimizer.type: sgd`).

## Resolving Weight Update OOM Errors

Weight updates can eat up a lot of memory, especially when using NCCL synchronization
(which is the default).

### 1. Switch to Disk-Based Updates

The easiest fix is switching from NCCL to disk-based weight synchronization:

```python
# Instead of NCCL-based updates
weight_update_meta = WeightUpdateMeta.from_disk(config.saver)
```

Check the "Transferring Weights to Inference Servers" section in the
[Weight Updates Guide](../lite/gsm8k_grpo.md) for the full implementation details.

### 2. Reduce Memory Buffer Size

If you want to stick with NCCL, try reducing the memory buffer size for weight chunking:

```python
# In WeightUpdateMeta.from_fsdp_xccl() calls
WeightUpdateMeta.from_fsdp_xccl(
    ...,
    weight_chunked_mem_mb = 512,  # Reduce from default (typically 1024+)
)
```
