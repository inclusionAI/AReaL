# LoRA Reference

LoRA is a parameter-efficient fine-tuning technique that injects trainable low-rank
matrices into pre-trained weights, typically around linear layers. Compared with
full-parameter fine-tuning, this reduces memory usage and compute cost substantially,
making RL fine-tuning of large models much more practical on limited hardware.

In AReaL, this is especially useful for:

- reinforcement learning with very large models, including 70B+ models, on relatively
  modest hardware such as 8 x 80 GB GPUs,
- enabling larger batch sizes because LoRA reduces training memory pressure,
- simplifying transfer and deployment because only the LoRA adapters need to be saved
  and shipped,
- \[Future\] fine-tune multiple LoRA adapters more efficiently in parallel for better
  hardware utilization (see RFC
  [#609](https://github.com/inclusionAI/AReaL/issues/609)).

This guide explains how to enable LoRA in RL training and configure the related
parameters.

## Backend Support

The current LoRA support matrix in AReaL is:

| Engine   | vLLM | SGLang |
| -------- | ---- | ------ |
| FSDP2    | ✅   | ✅     |
| Megatron | ✅   | ❌     |
| Archon   | ❌   | ❌     |

**Example scripts:**

| Engine       | Example script                                    |
| ------------ | ------------------------------------------------- |
| FSDP2        | `examples/math/gsm8k_grpo_lora.yaml`              |
| FSDP2 Delta  | `examples/math/gsm8k_grpo_lora_delta_sync.yaml`   |
| Megatron     | `examples/math/gsm8k_grpo_megatron_lora.yaml`     |
| Megatron MoE | `examples/math/gsm8k_grpo_megatron_lora_moe.yaml` |

For Megatron + vLLM, AReaL now supports:

- LoRA fine-tuning on MoE architectures such as Qwen3 MoE with XCCL-based LoRA weight.
- Cross-node LoRA training when the Megatron and rollout groups span multiple nodes.

## Core LoRA Parameters

| Parameter         | What it controls                                                                                        | Typical values        |
| ----------------- | ------------------------------------------------------------------------------------------------------- | --------------------- |
| `use_lora`        | Enables LoRA fine-tuning mode.                                                                          | `true` / `false`      |
| `lora_rank` (`r`) | Rank of the low-rank adapters. Higher rank increases capacity and memory/compute cost.                  | `8`, `16`, `32`, `64` |
| `lora_alpha`      | LoRA scaling factor. Effective adapter scale is commonly thought of as proportional to `alpha / r`.     | `16`, `32`, `64`      |
| `target_modules`  | Which model submodules receive LoRA adapters. This is the most important architecture-specific setting. | e.g. \[`all-linear`\] |
| `peft_type`       | PEFT method type. In AReaL configs, this is LoRA.                                                       | `lora`                |

## LoRA Delta Sync (Incremental Weight Update)

### Overview

In standard LoRA weight synchronization, the training engine merges LoRA adapter
weights into the base model and transmits the **full merged weights** to the inference
engine on every training step. For large models this transfer is expensive and can
become a bottleneck.

**LoRA Delta Sync** takes a different approach: the base model weights are transmitted
only once (on the first synchronization), and all subsequent synchronizations transmit
only the LoRA adapter weights (`lora_A` / `lora_B` matrices). Because the adapter
parameters typically account for **less than 1 %** of total model parameters, this
dramatically reduces weight synchronization time and network bandwidth consumption.

### Applicable Scenarios

LoRA Delta Sync is designed for the following combination:

- **Training engine:** FSDP (FSDP2)
- **Inference engine:** SGLang
- **Fine-tuning method:** LoRA (`use_lora: true`)
- **Weight update mode:** Disk

> **Note:** Delta Sync is not available with vLLM or the Megatron training engine.

### How It Works

The synchronization proceeds in two phases:

1. **First synchronization (Phase 1)**
   - **Phase 1a** -- The FSDP engine saves the **base model weights** (excluding
     LoRA parameters) to disk in HuggingFace safetensors format. SGLang loads
     them via the `/update_weights_from_disk` endpoint.
   - **Phase 1b** -- Immediately after, the LoRA adapter weights are saved to
     disk, and SGLang loads them via the `/load_lora_adapter` endpoint. The
     SGLang server loads the adapter on top of the base model.

2. **Subsequent synchronizations (Phase 2)**
   - Only the **updated LoRA adapter weights** are saved to disk and loaded via
     `/load_lora_adapter`. The base model weights already reside in
     the inference engine's GPU memory and are not re-sent.

This two-phase design means that after the initial (full) sync, every subsequent
weight update transfers only the tiny adapter delta, making each iteration
significantly faster.

### Configuration

To enable LoRA Delta Sync, set `lora_delta_sync: true` in the actor (training engine)
section of your YAML configuration, alongside the standard LoRA settings:

```yaml
actor:
  backend: "fsdp:d4"
  path: Qwen/Qwen2.5-1.5B-Instruct

  # Weight update mode: disk when lora_delta_sync is enabled
  weight_update_mode: disk

  # Standard LoRA settings
  use_lora: true
  peft_type: lora
  lora_rank: 16
  lora_alpha: 16
  target_modules: [all-linear]

  # Enable delta sync
  lora_delta_sync: true
```

The SGLang section should enable LoRA support and set the maximum LoRA rank:

```yaml
sglang:
  model_path: ${actor.path}
  dtype: ${actor.dtype}
  enable_lora: ${actor.use_lora}
  max_lora_rank: ${actor.lora_rank}
  mem_fraction_static: 0.8
```

A complete working example is available at
`examples/math/gsm8k_grpo_lora_delta_sync.yaml`.

### Key Parameters

| Parameter            | Required value / notes                                                                  |
| -------------------- | --------------------------------------------------------------------------------------- |
| `use_lora`           | `true`                                                                                  |
| `lora_delta_sync`    | `true` -- enables the incremental sync path.                                            |
| `weight_update_mode` | `disk`  |
| `lora_rank`          | Must match between training and inference configs (e.g. `16`).                          |
| `lora_alpha`         | LoRA scaling factor, same as standard LoRA.                                             |
| `sglang.enable_lora` | `true` -- the SGLang server must be launched with LoRA support enabled.                 |
| `sglang.max_lora_rank` | Must be >= the `lora_rank` used by the training engine.                               |

### Performance Benefits

- **Reduced transfer volume:** Adapter weights are typically < 1 % of total model
  parameters. For a 70B model with `lora_rank=16`, the adapter is roughly a few hundred
  MB versus tens of GB for the full model.
- **Lower synchronization latency:** After the one-time base model sync, each
  subsequent weight update completes in a fraction of the time.
- **Better GPU utilization:** Less time spent on weight transfer means more time
  available for training and inference.

### Important Notes

- **SGLang memory saver:** When `enable_lora` is set in the SGLang config, the launcher
  automatically enables `enable_memory_saver=True`. This keeps the base model weights
  in GPU memory across iterations (only the KV cache is released between rounds),
  avoiding re-transmission of the base weights.
- **SGLang version compatibility:** The SGLang server must support the
  `/load_lora_adapter` and `/update_weights_from_disk` API endpoints. Ensure
  you are using a compatible SGLang version.
- **Adapter versioning:** Each sync produces a versioned adapter name (e.g.
  `lora-gsm8k-v0`, `lora-gsm8k-v1`). The previous adapter is automatically unloaded
  before the new one is loaded.
- **First sync overhead:** The very first synchronization still transmits the full base
  model, so its cost is comparable to a standard full-weight sync. The savings begin
  from the second synchronization onward.
- **Rollout config:** Set `use_lora: true` in the rollout section as well so that the
  inference engine applies the loaded adapter during generation.

## Practical Notes

- Start with `r=16` or `r=32` for most models, then tune upward only if needed.
- Keep `target_modules` consistent with your model architecture naming.
- For Megatron backend, LoRA requires `megatron-bridge` instead of `mbridge`.
