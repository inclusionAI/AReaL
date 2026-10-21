# M2：Actor＋Rollout 联合配置与双组件 revision 闭环

代码基准：`feature/autopilot-elastic`／`2ae0135c`，2026-10-19。 这是当前实现的固定快照；持续进度见
[PROGRESS.md](../../../../../PROGRESS.md)。

在 [M1 资源唤起／释放与训练恢复](../m1/README.md) 基础上，已贯通联合配置决策、真实应用、 历史采集，以及 cost model 和 optimizer
的独立修订、验收和部署。 验收范围为单机 Qwen3-0.6B＋AReaL v2 FSDP pure-DP＋SGLang，严格串行。
当前结论是执行链路成立；模型精度、训练质量和成本收益仍需独立实验。

## Policy loop

`ready batch → model＋optimizer → config apply → train／checkpoint／publish → committed history → 再规划`

- `OnlineJointPolicy` 使用真实 ready batch 和已提交历史构造有限窗口；默认 horizon=2，未来负载重复最近已知计数。
  冷启动保留初始配置；空闲仍使用 timeout 释放 Actor。
- `SerialJointModel` 是 profile 驱动的简化串行 cost model：预测训练、checkpoint、发布、整段 episode 和启停耗时，
  再按两侧实际驻留卡数累计 GPU seconds。训练允许有界局部 token 缩放；episode 要求匹配 Rollout 配置及负载计数， 不推断 TP
  加速。episode 包括环境／导出／入队，不能解释为纯推理。
- `JointOptimizer` 在显式完整配置白名单内搜索。每个候选在整个预测窗口保持同一配置，以 Actor＋Rollout 总 GPU seconds 为目标，
  可限制作业时间与评估次数；默认切换收益门槛 5%。执行当前配置决策后重新规划，不搜索逐步任意配置序列。
- `ElasticLoop` 掌管执行门禁；`ConfigApplication＋JointConfigExecutor` 记录意图、排空／置换资源、恢复状态、
  加载同版权重并提交回执。支持 Actor pure-DP 卡数及 SGLang replicas／TP／容量参数；结果不确定时关闭 admission。
- checkpoint 和权重发布成功后才记录实际配置、负载与测量；episode 保留生成时的完整源配置，避免把切换后的配置错当测量来源。

## 两条独立 revision loop

启用 revision command 后，至少积累 4 次已提交测量；默认每增加 4 次 update 启动一窗，最多 3 窗， 每窗每组件最多 2 次提案。触发器目前是
update 周期；单个后台 CPU worker 修订，训练可以继续。

| 组件       | evaluator 的依据                                            | 接受条件                                                            |
| ---------- | ----------------------------------------------------------- | ------------------------------------------------------------------- |
| Cost model | 该次历史中的实际完整配置、负载、阶段／整段 episode 实测耗时 | 同样覆盖样本，phase×config 分组 WAPE 均值改善，无分组退化或覆盖损失 |
| Optimizer  | 当前冻结可执行模型、同一组规划窗口和配置白名单              | 配置质量不退化，并改善模型成本，或在同质量下减少查询／求解开销      |

历史按完整 update 的时间顺序划分：前 60%（至少 2 次）供校准，其余用于反馈验证；验证耗时标签由 evaluator 保存。
模型输入使用校准历史，不夹带验证耗时；小样本组抑制数值反馈。这是反复反馈的验证窗口，不是最终独立测试集。

模型阶段接受新版本或保留旧版本后，冻结有效模型，先重验旧 optimizer，再修订 optimizer。
两个组件各有验收记录；至少一个组件被接受，且两者在当前上下文均有效，才导出同一个部署 bundle。 bundle
包含两份可执行源码、profile、设置和验收身份，在安全规划边界原子激活。

当前模型 revision 只改变条件耗时预测；执行顺序、GPU 记账、transition profile、配置白名单、horizon 和 idle timeout 固定。
optimizer 的返回配置、成本与查询开销由 evaluator／runtime 复核。有限预测表用于搜索评测，部署的是可执行模型源码。

## 执行与恢复边界

- 训练权重版本、config generation、policy generation／bundle ID 分开记录。policy 激活保留历史与权重，不要求重启 GPU。
- 配置申请、部署意图／回执、完整有效联合历史和双游标持久化；恢复不重放已提交 update 或配置申请。
- 当前 whole-episode gate、单在途 batch、staleness=0；没有异步 Actor／Rollout 吞吐验收。

## 代码职责

| 位置                                                                                                                                                                                                                            | 职责                                                           |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| [`basic/online.py`](../../../../../areal/autopilot/elastic/policy/basic/online.py)、[`optimizer.py`](../../../../../areal/autopilot/elastic/policy/basic/optimizer.py)                                                          | 在线联合窗口、串行模型、搜索与历史反馈                         |
| [`elastic/loop.py`](../../../../../areal/autopilot/elastic/loop.py)                                                                                                                                                             | 调度动作、执行门禁、训练提交与发布                             |
| [`runtime/config.py`](../../../../../areal/autopilot/elastic/runtime/config.py)、[`history.py`](../../../../../areal/autopilot/elastic/runtime/history.py)                                                                      | 联合配置应用／恢复与真实观测                                   |
| [`revision/online.py`](../../../../../areal/autopilot/revision/online.py)、[`serial.py`](../../../../../areal/autopilot/revision/serial.py)、[`joint_evaluator.py`](../../../../../areal/autopilot/revision/joint_evaluator.py) | 后台触发、模型验收、冻结模型后的 optimizer 验收                |
| [`runtime/policy.py`](../../../../../areal/autopilot/elastic/runtime/policy.py)、[`serial_policy.py`](../../../../../areal/autopilot/elastic/runtime/serial_policy.py)                                                          | bundle 准备、身份验证、安全激活与恢复                          |
| [`automatic_swe.py`](../../automatic_swe.py)、[`fixtures/`](../../fixtures/)                                                                                                                                                    | 组装入口与小模型执行设施；运行参数见 [README](../../README.md) |

[架构图](../../figures/elastic-loops-compact.svg) 分开表示 policy loop、model revision 和
optimizer revision。 固定 Rollout 的 `BasicPolicy` 生命周期路径、离线异步 `JointSimulator`／历史 evaluator
仍是独立模式，不能混作在线串行模型。 Expert 在另一分支／对话开发，仅作固定对照，不是当前 Basic 的执行依赖或修订标签来源。

## 已验证与未验证

详细输入身份、结果、失败修正及产物引用见 [H20 验证记录](../../validation/joint-runtime-revision-20261019.json)。

| 验收项                 | 已验证结果                                                                                     |
| ---------------------- | ---------------------------------------------------------------------------------------------- |
| 双组件 revision 与上线 | 模型／optimizer 均先拒后接受；v5 激活，后台修订与真实训练重叠                                  |
| 联合自动配置           | 随后自动 Actor 2→1、Rollout 1→2；保持同一权重版本并继续训练                                    |
| 训练与推理             | 7 次真实 PPO update；v0–v7 共 8 次真实推理探针                                                 |
| 状态保持与恢复         | 19.44 亿模型／Adam 元素精确比对；v6 重建协调器对象和真实 GPU 服务，继续到 v7，无训练／配置重放 |
| CPU 回归与清理         | Linux 481 通过、1 项 CUDA RNG 跳过；驱动／独立审计退出 0，自有容器／tmpfs 清理、8 卡显存归零   |

这些结果使用合成 PPO 输入、受控 seed profiles 和确定性外部 JSON proposer；不是新一轮真实 SWE／Tau2 实验或真实 LLM 联合修订。
测量验收针对给定实际负载计数的条件耗时，不证明未来整窗预测、新配置精度、reward 或 cost-efficiency 改善。 checkpoint 位于自有 host
tmpfs；协调器对象／服务重建不等于全新 OS 进程、host 重启或掉电恢复。

## 整理与后续顺序

1. 以此快照为边界，先理清在线联合主线、固定 Rollout 兼容路径、离线实验及 fixture 的入口与依赖；重构时保持已验证行为。
1. 再用真实 agent、独立 workload 与固定 policy 对照，分开评估模型、optimizer 和联合修订，计入搜索／修订／切换开销。
1. 最后适配原生 SWE／Tau2＋Archon／Megatron。九条 Tau2 为 v1，需核对与 v2 的
   dispatch／packing／计时差异，原始数据留公司端。
