# Megatron-HF Bridge Backend

AReaL currently supports two bridge backends for `MegatronEngine`:

- `mbridge` (default)
- `megatron-bridge`

Set the backend with:

```yaml
actor:
  megatron:
    bridge_type: mbridge
```

- Use `bridge_type=megatron-bridge` to enable the new path.
- `mbridge` is the default choice if this argument is not present

## Custom pipeline layout

Set `actor.megatron.pipeline_model_parallel_layout` to use a native Megatron layout with
either bridge backend. For example, a model with 14 transformer layers can use this
layout with PP=2 and VPP=1:

```yaml
actor:
  megatron:
    pipeline_model_parallel_layout: "Et*10|t*4L"
```

`E` denotes embedding, `t` a transformer layer, `m` an MTP layer, `L` loss, and `|`
separates stages. A list of stage lists using native names such as `embedding`,
`decoder`, `mtp`, and `loss` is also accepted. The layout must match the model's layer
counts and enabled MTP head. For VPP, set `virtual_pipeline_parallel_size` explicitly;
the layout must contain PP times VPP stages in virtual-rank-first, physical-rank-second
order. This option does not change the allocation's PP size.

The default is `None`, which preserves the existing automatic split or configured
endpoint counts/accounting options. An explicit layout cannot be combined with
`num_layers_in_first_pipeline_stage`, `num_layers_in_last_pipeline_stage`,
`account_for_embedding_in_pipeline_split`, or `account_for_loss_in_pipeline_split`.

## Why this feature exists

- `mbridge` is being deprecated and does not provide PEFT/LoRA support.
- `megatron-bridge` supports more/ newer model architectures.
- `megatron-bridge` provides built-in PEFT/LoRA implementations.

## Recommendation

- For new GPU training workflows, prefer `megatron-bridge`.
- Keep `mbridge` for backward compatibility and environments that still depend on it.
- Prefer `mbridge` when using disk-based weight broadcast as it has optimized HF
  load/save path.
- If you use XCCL for weight broadcast, load/save time is less important.

## Current limitation

- Tree-attention training in `MegatronEngine` currently supports only `mbridge`.
- The `megatron-bridge` backend is not supported in the tree-attention path yet.
- `megatron-bridge` does support faster/optimized HF model load/save implementations.
