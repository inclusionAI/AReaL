# Fine-tuning Large MoE Models

Compared to PyTorch FSDP, Megatron-LM supports full 5D parallelism, delivering better
scaling and efficiency. AReaL fully supports customized RL training with Megatron-LM as
the backend. This guide explains how to harness the Megatron training backend and train
large MoE models for your application.

## Enabling Megatron in `allocation_mode`

Shifting from FSDP to Megatron requires only a single line of change: the
`allocation_mode` field from `sglang:d4+fsdp:d4` to `sglang:d4+megatron:d4`.

We already have some internal logic for determining the backend to use if the backend
name is omitted. If neither pipline parallelism nor expert parallelism is enabled, FSDP
will be used as the backend. Otherwise, Megatron will be used. However, we encourage
specifying the backend name explicitly like above.

## Understanding `allocation_mode`

The allocation mode is defined in
[areal/api/alloc_mode.py](https://github.com/inclusionAI/AReaL/blob/main/areal/api/alloc_mode.py).
The allocation mode is a pattern-based string option that tells AReaL how to parallelize
models across GPUs in training and inference backends. When running the experiment,
AReaL converts the string option into an `AllocationMode` object that stores the backend
choice and parallel strategy for each model. For a simple example,
`sglang:d4+megatron:t4` configures AReaL to use the SGLang backend with **data
parallel** size 4 and the Megatron training backend with **tensor parallel** size 4.

### Training Parallel Strategy

For a dense model, there are only 4 available parallel dimensions: data parallel (DP,
d), tensor parallel (TP, t), pipeline parallel (PP, p), and context parallel (CP, c).
The numbers that follow the single-character abbreviation of parallel dimensions
describe the parallel size. For example, `megatron:d2t4p2c2` describes a 32-GPU parallel
strategy that has DP size 2, TP size 4, PP size 2, and CP size 2.

For MoE models, the AReaL allocation mode supports separate parallel strategies for
expert modules and attention modules, which is related to the
[MoE Parallel Folding](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/transformer/moe#moe-parallel-folding)
feature in Megatron. It reduces the minimal number of GPUs required to enable both
context and expert parallelism (EP, e), and enables different TP sizes for attention and
expert modules for better efficiency. The parallel strategies for attention and expert
modules are denoted by `attn:` and `ffn:`, and separated by `|`. For example,
`megatron:(attn:d1p4t2c2|ffn:d1p4t1e4)` describes a 16-GPU parallel strategy with PP
size 4, that has DP size 1, TP size 2, and CP size 2 for attention modules and DP size
1, TP size 1, and EP size 4 for expert modules.

**5D parallel strategy Tuning Guides:**

- [Megatron Performance Best Practice](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/transformer/moe#performance-best-practice)
- [verl with Megatron Practice](https://github.com/ISEEKYAN/verl_megatron_practice)

### Inference Parallel Strategy

The optimal parallel strategy is ususally different for training and inference.
Inference parallel strategies only accept DP, TP, and PP, e.g., `vllm:d2t4`. Note that
DP degree is the number of independent instances to deploy. Other parallelism
configurations are passed through the `sglang` and `vllm` field in configurations, e.g.,

```yaml
sglang:
  ep_size: 2
  dp_size: 4
  enable_dp_attention: true
  ...
```

Note that the above configurations controls the internal hybrid parallelism strategy
within each inference instance, e.g., DP attention. These techniques are ususally not an
orthogonal dimension to DP, TP, and PP that determine GPU allocation. We refer to the
large-scale EP delopment guide of
[SGLang](https://lmsys.org/blog/2025-05-05-large-scale-ep/) and
[vLLM](https://docs.vllm.ai/projects/ascend/en/v0.9.1-dev/developer_guide/performance/distributed_dp_server_with_large_ep.html)
for detailed information.

## Aligning Inference and Training Precision

Due to the sparse nature of MoE models, the logits calculated by forward passes during
inference and training could be severely misaligned, leading to unstable training
results. To mitigate this instability, it is highly recommended to set
`actor.megatron.use_deterministic_algorithms=True` to disable nondeterministic
calculations in Megatron, although this may cause a ~10-20% slowdown in training steps.

As an example, you can run GRPO on the Qwen3 30B-A3B MoE model and GSM8K dataset (on a
32-GPU ray cluster) directly with the following command:

```bash
# NOTE: Allocation mode here is only for illustration purposes. It is not optimized.
python3 examples/math/gsm8k_rl.py --config <megatron_config.yaml> \
    scheduler.type=ray \
    experiment_name=megatron-moe-gsm8k-grpo trial_name=trial-0 allocation_mode=sglang:d4t4+megatron:(attn:d1p4t2c2|ffn:d1p4t1e4) \
    cluster.n_nodes=4 cluster.n_gpus_per_node=8 actor.path=Qwen/Qwen3-30B-A3B \
    actor.megatron.use_deterministic_algorithms=True
```
