# LoRA 参考

AReaL 支持基于 LoRA 的 RL 微调。

## 后端支持

LoRA 同时支持以下训练后端：

- FSDP
- Megatron（通过 `megatron-bridge`）

在 Megatron 中，请设置：

```yaml
actor:
  megatron:
    bridge_type: megatron-bridge
```

## 核心 LoRA 参数

| 参数              | 作用                                                               | 常见取值              |
| ----------------- | ------------------------------------------------------------------ | --------------------- |
| `use_lora`        | 是否启用 LoRA 微调模式。                                           | `true` / `false`      |
| `lora_rank` (`r`) | 低秩适配器的秩。`r` 越大，表达能力越强，但显存与计算开销更高。     | `8`, `16`, `32`, `64` |
| `lora_alpha`      | LoRA 缩放系数。通常可理解为有效缩放与 `alpha / r` 成正比。         | `16`, `32`, `64`      |
| `target_modules`  | 指定注入 LoRA 的目标子模块。这是最关键、且与模型结构强相关的配置。 | 例如 \[`all-linear`\] |
| `peft_type`       | PEFT 方法类型。在 AReaL 配置中为 LoRA。                            | `lora`                |

## 实践建议

- 可先从 `r=16` 或 `r=32` 开始，再按效果和资源逐步调参。
- `target_modules` 需与具体模型的层命名保持一致。
- 若 rollout 使用 vLLM 或 SGLang，也要同步开启对应 LoRA 相关配置。
- 在 Megatron + LoRA 场景下，优先使用 `megatron-bridge`，不建议使用已逐步弃用的 `mbridge`。
- 当前仅支持 dense 模型（非 MoE）。
