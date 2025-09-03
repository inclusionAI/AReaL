# Handling Out-of-Memory (OOM) Issues

This guide provides practical strategies to resolve Out-of-Memory (OOM) issues across different stages of your workflow, including generation, training, and weight updates.

## Key Factors in Memory Consumption

Understanding the key parameters that influence memory consumption is crucial for mitigating OOM errors. Here are the primary factors:

- `allocation_mode`: GPU parallel strategy allocation mode
- `train_dataset.max_length`: Maximum sequence length for training data
- `gconfig.max_new_tokens`: Maximum number of tokens to generate
- `actor.mb_spec.max_tokens_per_mb`: Maximum tokens per micro-batch of actor for each forward pass
- `max_concurrent_rollouts`: Maximum number of concurrent rollouts to the inference engine
- Inference Engine Parameters: Refer to [SGLang Documentation](https://docs.sglang.ai/)
- Training Engine Parameters: Refer to [FSDP Documentation](https://docs.pytorch.org/docs/stable/fsdp.html)

::::{note}
The `train_dataset.batch_size` parameter does not directly affect memory usage.
::::

## Generation

To resolve OOM errors during inference, consider the following strategies:

- Reduce `max_concurrent_rollouts`: This is often the most effective method. Lowering this value decreases the number of parallel generation tasks, thereby reducing total memory consumption.
- Adjust `allocation_mode`: Increase the `tp_size` (tensor parallelism size) to distribute memory load across multiple GPUs.
- Tune Inference Engine Parameters: Tune the parameters specific to SGLang engine like `actor.sglang.mem_fraction_static`. For guidance, refer to the [SGLang Documentation](https://docs.sglang.ai/).

## Training

OOM errors during the training phase can often be addressed with the following strategies:

- Set `actor.mb_spec.max_tokens_per_mb`: Set this to its minimum possible value, which should be `train_dataset.max_length + gconfig.max_new_tokens`
    - For multi-turn tasks, the minimum `gconfig.max_tokens_per_mb` should be `<length of the longest prompt> + gconfig.max_new_tokens`
- Increase FSDP Ulysses Sequence Parallelism: increase `sp_size` to reduce memory usage

## Weight Updates

NCCL-based weight updates are enabled by default, which requires additional memory to transferring weights. You can reduce the OOM errors by the following options:

- Switch to Disk-Based Weight Updates: See [Weight Updates Guide](../lite/gsm8k_grpo.md#transferring-weights-to-inference-servers)

- Reduce `weight_chunked_mem_mb` parameter: Adjust this parameter in the `WeightUpdateMeta.from_fsdp_nccl()` function