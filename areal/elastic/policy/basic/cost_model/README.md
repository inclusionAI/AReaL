# Qwen3-30B-A3B actor 耗时模型

`config + workload_stats -> actor 各阶段秒数`。离线解析模型，纯标准库；无需服务器、 采集、LLM API 或运行训练。目标硬件为用户指定的
H200、每节点 8 卡；32 卡 trial 为 4 节点。节点间默认 50 GB/s/GPU 是参考情景，可覆盖。

```python
from areal.elastic.policy.basic.cost_model import (
    ActorConfig, ActorCostModel, WorkloadStats, hardware_profile, qwen3_30b_a3b,
)

config = ActorConfig.from_areal(resolved_config_dict)
stats = WorkloadStats.from_areal_metrics(one_actor_batch_metrics)
model = ActorCostModel(qwen3_30b_a3b(), hardware_profile("h200_sxm_reference"))
seconds = model.predict(config, stats)
report = model.estimate(config, stats)  # 假设、packing、敏感性与校准来源
```

## 输入对应 AReaL 哪些数据

| 输入                         | 现有来源／口径                                                                                                        |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| 后端与 DP/TP/PP/CP/EP        | 已解析的 `actor.backend`，如 `megatron:(attn:d4p4t2\|ffn:d2p4e4)`；EP 不额外乘进 world size                           |
| microbatch 与 padding        | `actor.mb_spec`、`actor.pad_to_maximum`；`n_mbs` 是下限，Megatron PP4 实际下限为 8                                    |
| PPO、logprob、重计算、优化器 | `ppo_n_minibatches`、`recompute_logprob`、decoupled/prox 配置、checkpoint 与后端 DDP 配置                             |
| 实际全局序列数               | `ppo_actor/n_seqs`，DP 求和；已经过 rollout/turn 展开，不等于 prompt batch                                            |
| 实际序列 token               | `n_seqs × ppo_actor/seq_len/avg`；可加 `seq_len/min,max`                                                              |
| loss token                   | 可选 `ppo_actor/n_valid_tokens`；不用于缩减 transformer FLOPs                                                         |
| 外层 padding token           | 可选 `ppo_actor/n_tokens`；仅记录，它统计原始矩形张量槽位                                                             |
| 更细输入                     | 可选 `sequence_lengths`、同步 `train_microbatches`；分组可来自现有 `TrajBatchMeta.traj_group_sizes`，非默认日志必填项 |
| GPU、拓扑                    | 显式硬件 profile；summary 日志无法据 GPU 型号推导实际网络带宽                                                         |

每次传入同一个 actor batch 的配置和统计，不能把多步窗口累计数混成单步负载。 `update/num_micro_batches`
是标量观察的平均值，`__count` 是观察次数，均不自动解释为总 microbatch 数。 缺少长度列表时，用保总 token、已知 min/max
的确定性替代分布；另一种有界长尾分布用于敏感性分析。 默认日志没有长度平方和、expert route 或通信重叠率，模型不要求这些字段。 DP 原子分配的是
trajectory dict，可包含多行 turn。仅有 sequence lengths 仍不等于精确分发； 可选 `trajectory_group_sizes` /
`trajectory_dispatch_weights` 或 `per_dp_sequence_lengths` 细化。 summary
缺少这些元数据时采用代理分发，`exact_dispatch=false`，不据此断言实际 DP 可行性。

## 输出与组件

外层输出与 `rl_trainer.py` 对齐：`recompute_logp`、`compute_advantage`、`train_step`； 跳过的 logprob
阶段输出 0。训练包含全部 PPO minibatch 更新，每个 minibatch 一次 optimizer step。 内部计算、暴露通信、pipeline
bubble、optimizer、梯度同步、参数 gather 是解析拆分； 不声称 Archon 默认 trace 单独测到了
forward/backward/optimizer。

- `packing.py`：trajectory 原子分发／缺省 sequence 代理、PPO 拆分、FFD、跨 DP 同步数量、后端对齐和 padding。
  `pad_to_maximum` 的 dummy segment 也计入 attention 的长度平方和。
- `components.py`：算子 roofline、MoE、TP/CP/EP/FSDP、Adam、PP 调度； `Serial/Parallel/Scale`
  提供组合，`ActorComponents` 可独立替换模块。
- `actor.py`：组合为三个 actor 阶段。MoE FLOPs 使用 top-k，权重搬运与 FSDP 使用全部本地 expert 参数。
- `observability.py`：解析已有 summary、rank microbatch 日志、Chrome JSON/JSONL。 trace 只取一种
  trainer/worker 层级，同线程嵌套区间求并集、各 rank 取最大；缺 step 的事件保持未归属。
- `calibration.py`：公开总步时间只拟合一个正比例系数，不据此辨识内部各阶段。

Megatron PP4 层划分按当前后端参数桶得到 `11/13/13/11`；8 卡节点下其 DP 通信留在节点内。 Archon 梯度 reduce-scatter 按每
engine microbatch 计，final head 的 gather 缓存与主体分开。

主公式：算子时间为 `max(FLOPs/有效计算吞吐, bytes/有效HBM带宽)`，再组合暴露通信与调度。 Archon CP 使用 Ulysses
all-to-all；FSDP 使用完整 dense/expert 参数及各自分片组。 普通 EP all-to-all 与 HybridEP/DeepEP 分层通信分开；公开
benchmark 的优化不自动套给 trial。 所有带宽用**单向十进制 GB/s**；BF16 peak 不使用 sparse TFLOPS。

## 直接运行

在 AReaL 根目录执行；目录入口可绕开顶层训练依赖，普通 Python 即可使用：

```bash
python areal/elastic/policy/basic/cost_model predict --trial S031 \
  --workload examples/autopilot/actor_cost/workload.json \
  --actor-overrides examples/autopilot/actor_cost/actor_overrides.json
python areal/elastic/policy/basic/cost_model trial-suite \
  --workload examples/autopilot/actor_cost/workload.json \
  --actor-overrides examples/autopilot/actor_cost/actor_overrides.json
python areal/elastic/policy/basic/cost_model public-validation
```

真实使用将 `--trial` 换为 `--config resolved.json`，将 `--workload` 换为 `--metrics row.json`。
`--hardware-json` 接收自定义 `Hardware`；`--group-gbps` 可覆盖 `tp/cp/ep/dense_dp/expert_dp/pp`
组的有效带宽。`--output` 保存 JSON。`trace path.jsonl` 读取已有 trace。 截图未指定的训练选项必须由 resolved config 或
overrides 补足；示例 workload 是合成数据， 9 个 trial 共用同一 actor 负载以验证接口，不代表公司的 tau2 实验或各 prompt
batch 的展开结果。

## 校准证据与限制

[模型结构](data/qwen3_30b_a3b.json) 来自固定 HF revision；
[公开数据](data/public_training.json)、[来源](data/public_sources.json) 与
历史[可重现验证](../../../../../../artifacts/repository-archive-20261021/files/examples/autopilot/actor_cost/public_validation.json)已移入本地归档；公开输入仍随代码保存，可通过
`public-validation` 重新生成报告。 NVIDIA Bridge 的 H100 natural-pretrain 用于拟合，短步 SFT 留出验证；优化
benchmark 单独作协议差异检验。 当前留出误差约 **38%**，尚不能证明“小 batch、H200、Archon 各阶段准确”。 公开 Liger H200
数字与用户的 dense Qwen3-32B 解析表只作参考，不冒充目标后端实测标签。

默认输出解析值。公开校准仅在匹配域使用 `--public-calibration matching`； H200/Archon 使用 `transfer`
必须显式指定，结果标为未验证外推。 敏感性范围是替代负载及效率假设的情景范围，**不是置信区间或误差上界**。 Selective checkpoint、编译、overlap
使用公开配置可支持的粗粒度近似；PP 使用填充／排空服务时间近似。 静态内存只给下界，不保证能运行。当前支持
BF16/FP16、全参数、FFD、granularity=1、ETP=1； 其他精度／offload／tree／KK 需替换相应组件，不能静默当作已支持。
初始化、warmup、checkpoint、refit、rollout 不在稳态 actor 阶段时间内。
