# Milestone 准备：在线联合闭环与 Expert 模拟

整理日期：2026-10-21。下表保留两条开发线的交付快照；Expert 已整合到 Elastic 主线，正式版本名待确定。

整合来源：Elastic `5d119b99` ← Expert `09adf2b5`。在线执行和离线参考保持原职责，统一 R10 入口并保留旧基线复现。 组合验证：634 项
CPU 通过、1 项 CUDA 跳过，R10 新旧模式数值复现；
[整合记录](../../validation/elastic-expert-integration-20261021.json)。整合不新增 GPU 或真实收益验收。

## 固定代码与职责

| 分支                            | 代码快照                            | 职责                                                              |
| ------------------------------- | ----------------------------------- | ----------------------------------------------------------------- |
| `feature/autopilot-elastic`     | `9d773bc3`，在线 M2 代码 `2ae0135c` | Basic policy、联合 config apply、两类 revision 的验收／部署／恢复 |
| `feature/expert-cost-simulator` | `f11e3802`                          | 独立 Expert cost model／simulator、R10 回放与校准、参考档案       |

[Elastic M2 快照](https://github.com/Tournesoly/areal-autopilot-elastic/blob/9d773bc334b3db610ba91d785ddeccdb1cc84a11/examples/autopilot/elastic/milestones/m2/README.md)
·
[三条循环架构图](https://github.com/Tournesoly/areal-autopilot-elastic/blob/9d773bc334b3db610ba91d785ddeccdb1cc84a11/examples/autopilot/elastic/figures/elastic-loops-compact.svg)
· [Expert 进度](../../../../../areal/autopilot/elastic/policy/expert_designed/PROGRESS.md)

- **Policy loop**：真实观测 → cost model＋optimizer → config apply → 训练／发布 → 已提交历史 → 再规划。
- **Model revision**：历史实测 → evaluator 反馈 → agent 修订模型 → 验收 → 安全激活。
- **Optimizer revision**：冻结有效模型 → evaluator 反馈 → agent 修订搜索 → 验收 → 安全激活。
- **Expert**：每组实验固定的离线参考，不含 optimizer；新版 R10 入口已合入 Elastic，原始基线通过显式 legacy 参数保留。

## 已验证

| 范围              | 证据与结果                                                                                                               | 解释边界                                                                                   |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------ |
| 在线联合闭环      | 单机 Qwen3-0.6B／v2 FSDP pure-DP＋SGLang；7 次 PPO、8 次推理；自动 Actor 2→1、Rollout 1→2；模型／Adam 比对和恢复续训通过 | 严格串行、合成 PPO 输入、受控 profiles／proposer；不代表真实 agent 修订或收益实验          |
| R10 条件回放      | Qwen3-30B-A3B／4 H20 Actor＋4 H20 Rollout，16 步／64 groups；实测回放最大 step 边界差 0.029s                             | 固定原到达和 batch；runner 完成时间为可消费时间代理，不反馈新配置对 Rollout 的影响         |
| Expert 阶段与发布 | advantage 前段校准后，同 8 个验证 step WAPE 5.64%；`update_weights` 后 11 步 WAPE 3.45%                                  | 发布误差与旧常数基线相同；整段校准不辨识组件速率；严格 logp／train 的 9／11／16 步仍未覆盖 |
| 原分支回归记录    | 在线 M2：Linux CPU 481 通过／1 跳过；Expert 相关 CPU 339 通过、5 组 CLI 验证通过                                         | 对应整合前的两份代码快照                                                                   |
| 整合 CPU／回放    | macOS CPU 634 通过／1 CUDA 跳过；R10 legacy 与结构化前段校准均通过                                                       | 模型／校准档案及在线核心代码不变；尚无整合后的 GPU、真实 agent 或收益验收                  |

[在线 H20 验证](https://github.com/Tournesoly/areal-autopilot-elastic/blob/9d773bc334b3db610ba91d785ddeccdb1cc84a11/examples/autopilot/elastic/validation/joint-runtime-revision-20261019.json)
· [R10 advantage 校准](../../validation/r10-advantage-prefix-calibration-20261020.json) ·
[权重发布模型验证](../../validation/r10-publication-20261020.json)

## 定版前需明确

1. 验收整合后的在线主线、固定 Rollout 路径、离线模拟及 fixture 依赖；保持在线执行与参考模型独立。
1. 对目标配置补阶段、发布和切换开销测量；Megatron／Archon／H200 已有模拟接入，跨配置精度及真实执行适配待验证。
1. 定义真实 agent 对照实验的工作负载、预算和质量／成本验收；当前结果不构成 cost-efficiency 收益结论。
1. 九条 Tau2 仍属公司端 v1 数据；v1/v2 dispatch／packing／计时边界需核对，最后 200 步继续保留不评分。

历史验收数字对应上表原始快照；整合后的 CPU／CLI 检查另行记录。新一轮 GPU／真实 agent 实验和正式 tag／release 后续推进。
