# Fine-tuning Large MoE Models with Megatron-LM Training Backend

Compared to PyTorch FSDP, Megatron-LM supports full 5D parallelism, delivering better
scaling and efficiency, especially for large MoE models. AReaL fully supports customized
RL training with Megatron-LM backend. This guide shows you how to wire the Megatron
training backend into your own training script.

## Example: Training Qwen2-1.5B-Instruct with GRPO on GSM8K

In [Getting Started with AReaL-lite](../lite/gsm8k_grpo.md) we walk through a GRPO run
that depends on the FSDP backend. The Megatron equivalent lives in
[examples/math/gsm8k_grpo_megatron.py](https://github.com/inclusionAI/AReaL/blob/main/examples/math/gsm8k_grpo_megatron.py).
You can modify your own training script to replace the FSDP backend with the Megatron
one through these simple steps:

- **Actor backend** – Replace `FSDPPPOActor` with
  [`MegatronPPOActor`](https://github.com/inclusionAI/AReaL/blob/main/areal/engine/ppo/actor.py),
  which is the composition of PPO functionals and the base Megatron training engine
  [`MegatronEngine`](https://github.com/inclusionAI/AReaL/blob/main/areal/engine/megatron_engine.py).
- **Train Engine Initialize** - Add `parallel_strategy=parallel_strategy` and
  `seed=config.seed` as arguments to `actor.initialize(...)` and `ref.initialize()`.
- **Weight Update Meta** - Replace `WeightUpdateMeta.from_fsdp_xccl` with
  `WeightUpdateMeta.from_megatron_xccl`.

Besides these differences, the remainder of the control loop, rollout workflows,
evaluation, and checkpoint handling stays the same. This design makes it extremely easy
to port customized FSDP training scripts to the Megatron backend.

To run the example on a single node with 8 GPUs, execute:

```bash
python3 -m areal.launcher.local examples/math/gsm8k_grpo_megatron.py --config examples/math/gsm8k_grpo_megatron.yaml
```

## Training MoE Models with Megatron 5D Parallelism

### Enabling 5D Parallelism

To enable 5D parallel strategy in Megatron, you need to properly configure the
allocation mode in AReaL, defined in
[areal/api/alloc_mode.py](https://github.com/inclusionAI/AReaL/blob/main/areal/api/alloc_mode.py).
The allocation mode is a pattern-based string option that tells AReaL how to parallelize
models across GPUs in training and inference backends. When running the experiment,
AReaL converts the string option into an `AllocationMode` object that stores the backend
choice and parallel strategy for each model. For a simple example,
`sglang.d2+megatron.t2` configures AReaL to use the SGLang backend with data parallel
size 2 and the Megatron training backend with tensor parallel size 2. Here we only
discuss how to configure the parallel strategy for the Megatron training backend, which
is the `megatron.t2` part in the previous example. For full details, please check the
documentation strings in
[areal/api/alloc_mode.py](https://github.com/inclusionAI/AReaL/blob/main/areal/api/alloc_mode.py).

For a dense model, there are only 4 available parallel dimensions: data parallel (DP,
d), tensor parallel (TP, t), pipeline parallel (PP, p), and context parallel (CP, c).
The numbers that follow the single-character abbreviation of parallel dimensions
describe the parallel size. For example, `megatron.d2t4p2c2` describes a 32-GPU parallel
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

### Aligning Inference and Training Precision

Due to the sparse nature of MoE models, the logits calculated by forward passes during
inference and training could be severely misaligned, leading to unstable training
results. To mitigate this instability, it is highly recommended to set
`actor.megatron.use_deterministic_algorithms=True` to disable nondeterministic
calculations in Megatron, although this may cause a ~10-20% slowdown in training steps.

As an example, you can run GRPO on the Qwen3 30B-A3B MoE model and GSM8K dataset (on a
32-GPU ray cluster) directly with the following command:

```bash
python3 -m areal.launcher.ray examples/math/gsm8k_grpo_megatron.py --config examples/math/gsm8k_grpo_megatron.yaml \
    experiment_name=megatron-moe-gsm8k-grpo trial_name=trial-0 allocation_mode=sglang.d4t4+megatron:(attn:d1p4t2c2|ffn:d1p4t1e4) \
    cluster.n_nodes=4 cluster.n_gpus_per_node=8 actor.path=Qwen/Qwen3-30B-A3B \
    actor.megatron.use_deterministic_algorithms=True
```
