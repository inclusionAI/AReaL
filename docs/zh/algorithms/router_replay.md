# MoE 路由回放（R3, Rollout Routing Replay）

最后更新：2026-04-29

## 背景

在 MoE 模型的异步强化学习中，负责 rollout 采样（SGLang）的策略与正在训练的
策略（Megatron-LM）往往相差一个或多个版本。由于 MoE 路由器是"学习到的稀疏
门控"，微小的权重漂移都会让同一个 token 在推理和训练时被送到不同的专家，
造成 **训练/推理路由不一致**，进而破坏 importance sampling 的比值、导致优化
不稳定。

**Rollout Routing Replay（R3）** 通过以下两个步骤消除这种不一致：

1. **记录**：在推理阶段记录每个 token 的专家分配结果。
2. **回放**：在训练前/反向阶段使用完全相同的专家分配替换由当前权重计算得到的
   路由结果。

R3 参考了 [verl](https://github.com/volcengine/verl) 的实现，并在 AReaL
仓库中被适配到 Megatron 训练后端 + SGLang bridge 模式的推理服务。

## 支持矩阵

| 维度 | 是否支持 | 说明 |
|---|---|---|
| 训练后端 | Megatron-LM（`MegatronEngine`） | 不支持 FSDP。 |
| 推理后端 | SGLang 0.5.9（bridge 模式） | 不支持 vLLM。 |
| 张量并行（TP） | ✅ | 通过 `scatter_to_sequence_parallel_region` 把打包后的路由索引分发到 SP 各 rank。 |
| 专家并行（EP） | ✅ | 补丁后的 `MoEAlltoAllTokenDispatcher.preprocess` 改用 `num_out_tokens = routing_map.sum()`，保证回放清零 padding 行后 dropless 路径依然正确。 |
| 流水并行（PP） | ✅ | `RouterReplayHelper.get_micro_batch_router_list` 根据当前 PP rank 的 `(layer_offset, num_layers)` 对 `RouterReplay.router_instances` 切片。 |
| 虚拟流水并行（VPP） | ✅ | 同一个 helper 会遍历 `virtual_pipeline_model_parallel_size` 指定的各 VP stage。 |
| 上下文并行（CP） | ⚠️ 实验性 | 当 `cp_size > 1` 时使用 `seq_align_to = tp_size * cp_size * 2`；本次只覆盖单元测试，端到端尚未验证。 |
| 数据并行（DP） | ✅ | 每个 DP 副本独立运行 R3，不引入跨 DP 通信。 |
| Dense + MoE 混合层 | ✅ | `is_moe_layer()` 使用 `moe_layer_freq` / `first_k_dense_replace` 识别并跳过 dense 层。 |
| 角色 | 仅 Actor | `config.actor.megatron.enable_router_replay` 仅在 actor 上被置为 True，Critic / Ref / Teacher 不受影响。 |
| Capacity factor | 仅 `moe_expert_capacity_factor is None`（dropless） | 和 verl 的 guard 一致，`num_out_tokens` 覆盖仅作用于 dropless 分支。 |
| FP8 / 量化 padding | ❌ | 当 `moe_router_padding_for_fp8` 或 `moe_router_padding_for_quantization` 开启时跳过 R3，以保持 FP8 dispatch 正确性。 |
| 视觉 / 多模态模型 | ❌ | VLM 路径未接入钩子。 |

## 如何开启

R3 由单一 rollout 开关驱动，其余都会在 `areal/trainer/rl_trainer.py` 中自动串起来。

```yaml
rollout:
    return_routed_experts: true        # 让 SGLang 返回每 token 路由索引

actor:
    backend: "megatron:(attn:d1p1t4|ffn:d1p1t1e4)"   # TP=4, EP=4
    # actor.megatron.enable_router_replay 会被自动设为 True

sglang:
    # R3 需要保证 token 序列与路由结果对齐；trainer 强制
    # skip_tokenizer_init=True，这里显式声明以表明意图。
    skip_tokenizer_init: true
    enable_return_routed_experts: true
```

启动后的自动串联逻辑：

1. `rollout.return_routed_experts=True` 令
   `config.actor.megatron.enable_router_replay = True`。
2. `num_moe_layers` / `topk` 由 `resolve_r3_moe_config()` 从 HF config（
   `num_experts_per_tok`、`num_hidden_layers`、`moe_layer_freq`、
   `first_k_dense_replace`）自动解析。
3. `sglang.skip_tokenizer_init` 被强制置为 `True`（若用户设为 False 会打印
   warning），以避免 tokenizer 往返造成的 token shift 破坏对齐。
4. SGLang bridge 入口
   (`areal/experimental/inference_service/sglang/launch_server.py`)
   在启动时调用 `apply_sglang_r3_patch()`，让
   `TokenizerManager._handle_batch_output` 在 FastAPI 序列化前把
   `routed_experts` 张量按 base64 编码。
5. 训练侧 `MegatronEngine.initialize()` 在模型构造 **之前** 调用
   `apply_router_replay_patch()`（monkey-patch `TransformerConfig.__init__`、
   `TopKRouter.__init__`、`TopKRouter.routing` 与
   `MoEAlltoAllTokenDispatcher.preprocess`），然后通过
   `patch_megatron_engine_for_r3()` 包装 engine。

## 关键数据结构

| 对象 | 作用 |
|---|---|
| `RouterReplay`（每 MoE 层一个） | 保存回放目标索引 `target_topk_idx`、记录缓冲 `recorded_topk_idx` 与当前 `RouterReplayAction`。 |
| `RouterReplay.router_instances` | 类级列表，保存当前 rank 上的每一个 MoE 层实例，每次 `apply_router_replay_patch()` 都会 `clear()`。 |
| `RouterReplayAction` | 枚举：`RECORD` / `REPLAY_FORWARD` / `REPLAY_BACKWARD`。 |
| `RouterReplayHelper.get_micro_batch_router_list()` | 返回当前 `(pp_rank, vp_stage)` 对应的 `router_instances` 切片。 |
| `setup_per_microbatch_replay_forward()` | 在每个 micro-batch 前向之前：把 rollout 格式的 `routed_experts` 对齐到训练 token 排布、按 `cu_seqlens` 打包、scatter 到 SP 各 rank、再分发到每一层 `RouterReplay`。 |

## 正确性要点

* **`num_out_tokens` 覆盖**：Megatron-Core 0.16.0 在 dropless 分支下使用静态值
  `routing_map.size(0) * moe_router_topk`；当 R3 清零 padding 行后，静态值会
  高估 token × topk 数量，因此补丁会在 dropless 分支用
  `int(routing_map.sum().item())` 覆盖。每 step 约 3500 次同步，相较 MoE
  计算完全可忽略。
* **按实例 `__class__` 替换**：micro-batch 迭代器通过动态子类替换 `mb_list.__class__`，
  而不是修改共享类，因此并行存在的其他 engine（例如 critic）不会受影响。
* **右填充 → 左对齐**：`_align_routed_experts_to_mask()` 根据 `cu_seqlens` 把
  rollout 的 `(bs, batch_max_seqlen, L, K)` 右填充张量转换到训练使用的左对齐
  布局。
* **显式校验**：micro-batch 无法被 `bs // n_mbs` 整除时直接抛错，而不是静默丢弃。

## 最小示例

参考 `examples/math/moonlight_16b_a3b_gsm8k_grpo_megatron.yaml`（Moonlight-16B-A3B，
PP=2、TP=4、EP=4，8 卡）。启动：

```bash
python3 examples/math/gsm8k_rl.py \
    --config examples/math/moonlight_16b_a3b_gsm8k_grpo_megatron.yaml \
    scheduler.type=local
```

单机 8×H200 场景可使用 `*_h20.yaml` 变体（PP=1、TP=4、EP=4、
`max_tokens_per_mb=10240`）。

## 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `[R3] Number of replay tensors (...) does not match number of router instances (...)` | HF config 中解析的 MoE 层数与 Megatron 的按 rank 切分层数不一致（多由 `first_k_dense_replace`、`moe_layer_freq`、自定义 pipeline layout 不一致引起）。 | 核对模型 `config.json` 中的 `num_hidden_layers`、`first_k_dense_replace`、`moe_layer_freq`，并确保自定义 `pipeline_model_parallel_layout` 与 MoE 层数一致。 |
| SGLang 返回 `routed_experts: {}`（空字典） | 推理服务未安装 R3 补丁。 | 确保使用 bridge 入口 `areal.experimental.inference_service.sglang.launch_server`（会自动调用 `apply_sglang_r3_patch()`）。 |
| 开启 `moe_router_padding_for_fp8=True` 后 R3 行为异常 | R3 在 FP8 padding 路径上被主动禁用。 | 关闭 FP8 router padding，或关闭 `rollout.return_routed_experts`。 |
| Critic 未生效 | 按设计只在 actor 上启用。 | 若后续需要 MoE critic 回放，需要扩展 `rl_trainer` 与 `MegatronEngine._r3_enabled` 的触发条件。 |

## 参考资料

* PR [#1207](https://github.com/inclusionAI/AReaL/pull/1207)
  `[WIP]feat: add router replay for megatron engine`。
* verl R3 源码：
  [`volcengine/verl`](https://github.com/volcengine/verl)。
* Megatron-Core MoE 并行折叠：
  [NVIDIA/Megatron-LM MoE README](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/transformer/moe)。
