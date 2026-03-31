# LoRA Reference

AReaL supports LoRA RL finetuning.

## Backend Support

LoRA is supported by both training backends:

- FSDP
- Megatron (through `megatron-bridge`)

For Megatron, set:

```yaml
actor:
  megatron:
    bridge_type: megatron-bridge
```

## Core LoRA Parameters

| Parameter         | What it controls                                                                                        | Typical values        |
| ----------------- | ------------------------------------------------------------------------------------------------------- | --------------------- |
| `use_lora`        | Enables LoRA fine-tuning mode.                                                                          | `true` / `false`      |
| `lora_rank` (`r`) | Rank of the low-rank adapters. Higher rank increases capacity and memory/compute cost.                  | `8`, `16`, `32`, `64` |
| `lora_alpha`      | LoRA scaling factor. Effective adapter scale is commonly thought of as proportional to `alpha / r`.     | `16`, `32`, `64`      |
| `target_modules`  | Which model submodules receive LoRA adapters. This is the most important architecture-specific setting. | e.g. \[`all-linear`\] |
| `peft_type`       | PEFT method type. In AReaL configs, this is LoRA.                                                       | `lora`                |

## Practical Notes

- Start with `r=16` or `r=32` for most models, then tune upward only if needed.
- Keep `target_modules` consistent with your model architecture naming.
- If rollout uses vLLM or SGLang, enable matching LoRA-related rollout options too.
- With Megatron + LoRA, prefer `megatron-bridge` rather than deprecated `mbridge`.
- Currently only dense models (non MoE) are supported.
