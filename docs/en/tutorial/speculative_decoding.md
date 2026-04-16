# Speculative Decoding with EAGLE

## Overview

Speculative decoding is a technique that accelerates autoregressive text generation by
using a lightweight **draft model** to propose multiple candidate tokens in parallel,
which the full **target model** then verifies in a single forward pass. When candidates
are accepted, the effective throughput increases significantly — often 2-3x — without
changing the output distribution.

AReaL integrates **EAGLE** (Extrapolation Algorithm for Greater Language-model
Efficiency) as its speculative decoding backend. EAGLE uses the target model's hidden
states to predict future tokens through a small auxiliary head, making it particularly
well-suited for RL training loops where the policy model evolves continuously.

### Why Speculative Decoding for RL Training?

In RLHF / GRPO training pipelines, rollout generation is often the throughput
bottleneck. Speculative decoding directly addresses this by:

- Reducing per-sample generation latency during rollout
- Increasing GPU utilization during the inference phase
- Maintaining identical output quality (the verification step is exact)

When combined with **MTP (Multi-Token Prediction) online training**, the draft model
stays aligned with the evolving policy, preserving high accept rates throughout training.

## Prerequisites

Before enabling speculative decoding, ensure:

1. **Model with MTP layers**: Your base model must include MTP (Multi-Token Prediction)
   head layers. Models such as `Qwen/Qwen3-0.6B` and other Qwen3 variants ship with
   MTP layers that can serve as EAGLE draft heads.

2. **SGLang backend**: Speculative decoding requires the SGLang inference backend.
   Ensure SGLang is installed and configured:

   ```bash
   pip install "sglang[all]>=0.4.7"
   ```

3. **Megatron-Core >= 0.12.0**: MTP online training requires Megatron-Core version
   0.12.0 or later, which includes the `MultiTokenPrediction` module with built-in
   gradient isolation (embedding detach and functional_call for LM head). This ensures
   MTP loss gradients only update MTP layer parameters without corrupting the main
   policy model.

4. **Sufficient GPU memory**: The draft model adds a small memory overhead on the
   inference GPUs. Reduce `sglang.mem_fraction_static` if needed (e.g., from `0.85` to
   `0.80`).

## Configuration

### SGLang EAGLE Configuration

Speculative decoding is configured under the `sglang` section of your experiment YAML.
The key fields live in `SGLangConfig`:

```yaml
sglang:
  model_path: ${actor.path}
  dtype: bfloat16
  mem_fraction_static: 0.80
  context_length: 32768

  # --- Speculative Decoding ---
  speculative_algorithm: "EAGLE"         # or "EAGLE3"
  speculative_draft_model_path: null     # null = use built-in MTP heads
  speculative_num_steps: 3              # number of draft steps per iteration
  speculative_eagle_topk: 1             # top-k for draft token selection
  speculative_num_draft_tokens: 4       # draft tokens proposed per step
  speculative_attention_mode: null      # null uses default attention
```

| Parameter | Default | Description |
|---|---|---|
| `speculative_algorithm` | `null` | Algorithm name: `"EAGLE"` or `"EAGLE3"`. `null` disables speculative decoding. |
| `speculative_draft_model_path` | `null` | Path to an external draft model. `null` reuses the target model's built-in MTP layers. |
| `speculative_num_steps` | `3` | How many autoregressive draft steps EAGLE performs before verification. |
| `speculative_eagle_topk` | `1` | Number of top-k candidates retained at each draft step. |
| `speculative_num_draft_tokens` | `4` | Total draft tokens fed to the verifier per speculative iteration. |
| `speculative_attention_mode` | `null` | Override attention kernel used during draft. `null` uses the engine default. |

### MTP Online Training Configuration

To keep the draft model aligned with the policy as it trains, enable MTP online
training in the `actor` section:

```yaml
actor:
  backend: "megatron:d4p1t1"
  path: Qwen/Qwen3-0.6B

  # --- MTP Online Training ---
  enable_mtp_training: true
  mtp_num_layers: 1                    # must match model's MTP architecture
  mtp_loss_scaling_factor: 0.1         # weight of MTP loss vs. main RL loss

  # Megatron-specific MTP settings (in actor.megatron)
  megatron:
    mtp_num_layers: 1                  # mirrors actor.mtp_num_layers
    mtp_loss_scaling_factor: 0.1       # mirrors actor.mtp_loss_scaling_factor
```

| Parameter | Default | Description |
|---|---|---|
| `enable_mtp_training` | `false` | Master switch for MTP online training. |
| `mtp_num_layers` | `1` | Number of MTP head layers to train. Must be > 0 when enabled. |
| `mtp_loss_scaling_factor` | `0.1` | Weight of the MTP auxiliary loss. Must be in (0, 1.0]. |

When `enable_mtp_training` is `true`, the trainer computes an auxiliary next-token
prediction loss on the MTP heads and adds it (scaled) to the main RL objective. This
ensures the draft heads continuously improve their prediction accuracy as the policy
changes.

## Full Example

Below is a minimal GRPO + EAGLE configuration for GSM8K with 4 GPUs:

```yaml
experiment_name: gsm8k-grpo-eagle
trial_name: trial0
seed: 42
tokenizer_path: ${actor.path}

cluster:
  n_nodes: 1
  n_gpus_per_node: 4

actor:
  backend: "megatron:d2p1t1"
  path: Qwen/Qwen3-0.6B
  enable_mtp_training: true
  mtp_num_layers: 1
  mtp_loss_scaling_factor: 0.1

sglang:
  model_path: ${actor.path}
  speculative_algorithm: "EAGLE"
  speculative_num_steps: 3
  speculative_num_draft_tokens: 4
  mem_fraction_static: 0.80

train_dataset:
  path: openai/gsm8k
  type: rl
  batch_size: 128
```

For the complete configuration file, see
[`examples/math/gsm8k_grpo_megatron_eagle.yaml`](https://github.com/inclusionAI/AReaL/blob/main/examples/math/gsm8k_grpo_megatron_eagle.yaml).

## Monitoring

### Key Metrics

During training, watch the following metrics in your logs or WandB dashboard:

1. **Speculative Accept Rate**
   - Logged as `spec_accept_rate` (= `spec_accept_token_num / spec_draft_token_num`)
   - A healthy accept rate is **0.6 - 0.9** for well-aligned draft models
   - If accept rate drops below **0.4**, the draft model is falling behind the policy

2. **MTP Loss**
   - Logged as `mtp_loss` in training statistics
   - Should decrease over time; a rising MTP loss indicates training instability
   - Typical range: **0.5 - 2.0** depending on model size and task

3. **Generation Throughput**
   - Compare tokens/second with and without speculative decoding
   - Expected speedup: **1.5x - 3x** depending on accept rate and model architecture

### Interpreting Accept Rate Trends

| Trend | Meaning | Action |
|---|---|---|
| Stable 0.7+ | Draft model is well-aligned | No action needed |
| Gradual decline | Policy is evolving faster than draft | Increase `mtp_loss_scaling_factor` |
| Sudden drop | Possible learning rate spike or data shift | Check training stability |
| Very low (<0.3) | Draft model is ineffective | Verify MTP layers are being trained |

## Troubleshooting

### Accept Rate is Very Low

1. **Verify MTP training is enabled**: Check that `actor.enable_mtp_training: true` is
   set. Without online training, the draft model will quickly become stale.

2. **Check MTP layer count**: Ensure `actor.mtp_num_layers` matches your model's
   architecture. Qwen3 models typically have 1 MTP layer.

3. **Increase MTP loss weight**: If the accept rate degrades over time, try increasing
   `mtp_loss_scaling_factor` from `0.1` to `0.2` or `0.3`.

### Out of Memory (OOM) During Inference

1. **Reduce memory fraction**: Lower `sglang.mem_fraction_static` (e.g., `0.75`).

2. **Reduce draft tokens**: Lower `speculative_num_draft_tokens` from `4` to `2`.

3. **Reduce draft steps**: Lower `speculative_num_steps` from `3` to `2`.

### Training is Slower Than Expected

1. **Check GPU allocation**: Ensure inference and training GPUs are properly separated.
   Use `sglang:d2p1t1` with `megatron:d2p1t1` on 4 GPUs for balanced allocation.

2. **Profile the pipeline**: Enable `perf_tracer.enabled: true` to identify whether
   the bottleneck is in generation, training, or data loading.

3. **Disable speculative decoding temporarily**: Set `speculative_algorithm: null` and
   compare throughput to isolate whether the overhead is from speculation itself.

### MTP Loss is Not Decreasing

1. **Verify model supports MTP**: Not all model architectures include MTP heads. Check
   that the model's config includes MTP layer definitions.

2. **Check learning rate**: The MTP heads share the actor's optimizer. If the base
   learning rate is too low, MTP training may stagnate.

3. **Inspect gradient flow**: Ensure `actor.gradient_checkpointing` is not interfering
   with MTP gradient computation.

## Advanced Configuration

### Using an External Draft Model

Instead of relying on built-in MTP layers, you can provide a separate draft model:

```yaml
sglang:
  speculative_algorithm: "EAGLE"
  speculative_draft_model_path: /path/to/eagle-draft-model
```

Note that when using an external draft model, `enable_mtp_training` should typically be
`false` unless the external model's weights are also updated during training.

### EAGLE3 Algorithm

EAGLE3 is an improved variant that supports more flexible tree-structured speculation:

```yaml
sglang:
  speculative_algorithm: "EAGLE3"
  speculative_num_steps: 5
  speculative_eagle_topk: 2
  speculative_num_draft_tokens: 8
```

EAGLE3 generally achieves higher accept rates but uses more memory for the expanded
draft tree.

### Draft Weight CPU Backup

When using colocated training and inference (i.e., the same GPUs serve both), draft
model weights may be lost during GPU memory reclamation. Enable CPU backup:

```yaml
sglang:
  enable_draft_weights_cpu_backup: true
```

This keeps a CPU copy of draft weights that is restored after each training step.

## References

- [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077)
- [SGLang Documentation](https://sgl-project.github.io/)
- [AReaL Megatron Backend Tutorial](megatron.md)
- [AReaL Allocation Mode Reference](../reference/alloc_mode.md)
