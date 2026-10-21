# Tau2 计划与公司端交接

范围：S031、S029、T0390、S026、S030、S028、S025、S024、S013。 当前状态统一维护在
[PROGRESS.md](../../PROGRESS.md)，Expert 细项见
[专用进度](../../areal/autopilot/elastic/policy/expert_designed/PROGRESS.md)。旧论文 draft 仅作参考。

[M2 实现快照](elastic/milestones/m2/README.md) 固定当前代码、验收与边界。
[架构图](elastic/README.md#policy-and-revision-loops) 区分 policy loop、cost-model revision 和
optimizer revision；当前联合在线闭环采用严格串行执行。

## 目标与实验约束

- 保留已有等待回收能力，下一阶段研究 rollout／actor 配置的成本优化及 AutoPilot 修订增益。
- 并行开发 Expert 与 Basic。校准并验证 `expert_designed` cost model／simulator，作为离线实验的固定高质量对照；它不包含
  optimizer。 Basic 的模型与 optimizer 是被评估、被修订的对象，专家模型在每组实验中冻结。
- 首轮建议固定各 trial 的训练语义、逻辑 batch、数据复用与模型；配置改变带来的队列、staleness、权重同步和切换开销需一起评估。
- 报告 GPU-hours、完成时间与 held-out 质量；episode、actor sequence uses、attention／loss tokens 分开计数。
  质量容差、时间约束和可调配置集合需结合真实运行确认。
- evaluator 负责评估候选、返回修订反馈和判断接受／拒绝；agent 据此修改 Basic。 模型 evaluator
  输入该次训练历史中的实际配置、负载与必要初态，以真实执行测量为标签。 首先验证历史已覆盖范围的预测准确性；未见配置的泛化另行验证，不作为当前历史修订的先决条件。
  校准、反复反馈的验证集与最终保留集分开；公司端最后 200 步继续保留不评分。
- simulator 实验在相同场景和指标上比较 `basic_v0`、各轮接受的修订版本与固定 expert，观察 Basic 的改善及其与 expert 的差距。
  expert 是高质量对照，不默认用其输出指导 Basic 拟合或生成修订标签；改善趋势由实验验证。
- optimizer 在当前冻结模型上搜索；evaluator 在固定预算下以同一模型复核返回的配置，并报告配置成本、求解时间与查询开销。 专家 simulator
  可提供离线成本参考，评分与修订反馈仍由 evaluator 负责。最终用独立实测验证，计入采样、搜索、修订和切换开销。
- 对照包括现有固定配置、固定 Planner、普通在线校准，再分别评估模型修订、optimizer 修订及联合修订。

## 公司端数据与补测

版本约束：用户确认九条 Tau2 来自 AReaL v1，目标系统是 v2。执行版本、配置与计时边界进入历史输入身份； 公司端先核对两版的
dispatch／packing／阶段定义。v1 历史误差不能直接证明 v2 异步模拟或新配置精度。

真实配置、日志与 checkpoint 在公司端读取；本地原始数据和运行产物放在已忽略的 `outputs/autopilot-tau2/`。 仓库里的
`trials.json`、情景输入和截图摘要不能作为真实运行配置或测量数据。

公司端截图摘要报告已检查 988 个完整训练 step：Qwen3-30B-A3B、H200，Actor32＋Rollout32 分卡， Rollout 均为
`sglang:d8t4`，staleness=2，group 并发上限 128。这里记录的是交接信息，本机未核验原始日志。 已有 Actor 阶段耗时、序列／token
汇总、rank microbatch 记录和 group/session trace；microbatch 尚需关联 step/phase 并去除 rank 重复。

| Trial              | Actor backend | Actor mesh                  | Batch：groups × samples |
| ------------------ | ------------- | --------------------------- | ----------------------- |
| S031 / S030 / S013 | Megatron      | `attn:d4p4t2 \| ffn:d2p4e4` | 16 × 8                  |
| S029 / S028        | Archon        | `attn:d4t4c2 \| ffn:d4e8`   | 16 × 8                  |
| S026               | Archon        | `attn:d32 \| ffn:d32`       | 64 × 8                  |
| T0390 / S025       | Archon        | `attn:d8t4 \| ffn:d8e4`     | 32 × 16                 |
| S024               | Archon        | `attn:d8t4 \| ffn:d8e4`     | 32 × 8                  |

仍需公司端提供或补测：

1. 推理 prefill/decode 服务与环境等待的分解，精确运行配置及各 profile 的负载覆盖范围。
1. 同一 workload 的跨配置测量；九条只有一种 Rollout 布局，不能据此验证任意联合配置。
1. 有向转换的真实耗时与驻留卡数：恢复／重分片、发布、checkpoint、启停；60s/120s 仍为情景假设。
1. 保留原始 trajectory 顺序的逻辑序列、source Actor profile，以及独立校准／验收划分。

`timeperf/rollout` 是获取／准备 batch 的时间，不能当纯推理；session finalize 是 group 收尾， 不能当每条 episode
独立结束。Basic 的批次并发代理也不能直接使用 group 上限 128。 缺失数据不填零；参数与真实数据在公司端生成，本机不要求传原始日志。

历史 trace 可以复核旧配置；改变 rollout 或 actor 配置会影响到达时间与队列，不能固定历史到达序列就宣称新配置收益。

## 后续顺序

1. Expert 与 Basic 独立推进；先用已有历史验证模型与覆盖范围，跨配置测量另行补足。
1. 冻结专家模型版本作为对照；准备 Basic 正常／弱模型起点及共用评测集。
1. 跑通固定 Planner 和普通在线校准基线，验证模型排序与实际执行收益。
1. 接通完整配置的离线 revision，分别及联合评估两类修订；弱模型起点实验使用同一独立验收标准。
1. 只对已验证支持的配置转换接入在线循环，再检查成本、完成时间与质量。

## 应用接入

现有验收基线为 Qwen3-0.6B＋FSDP＋SGLang 的执行、policy 应用、历史采集与两类 revision。 下一步先接原生在线入口的按需训练，顺序见下节；原生
SWE／Tau2 的完整 workflow 适配另行推进。 九条 Tau2 的研究范围和 Expert 独立参考职责不变。

已搭建 `runtime/config.py`：完整配置申请、代次／训练版本预检、持久意图／回执及恢复；`ElasticLoop` 显式入口和自动 Actor
及联合决策共用这条路径。联合适配器支持 v2 FSDP pure-DP Actor 与 SGLang 副本／TP／容量参数，在共享 GPU 池中置换、加载同版权重并联合恢复；
验证状态见 `PROGRESS.md`。固定 Rollout 与严格串行联合 policy bundle 应用已接入。

- config：先验证 source→target 能力，在严格队列／权重提交边界关闭
  admission、执行并核验；失败恢复源配置，状态不确定时保持关闭，不重放已提交训练。 `policy_bundle_id`、`config_generation`
  与训练权重版本分别记录。
- policy：不可变 bundle 绑定模型／optimizer 源码、拟合状态和验收身份；预加载后在规划边界整体切换，不确定提交按持久指针恢复。模型更新后先重评
  optimizer，runtime 历史不清空；切策略不回退权重或重启 GPU。
- 联合自动：`OnlineJointPolicy → JointOptimizer → SerialJointModel` 使用真实 ready batch
  与已提交历史预测，自动应用首个完整配置，随后采集实测再规划。 当前严格串行，阶段 profile 按完整配置隔离；Rollout 使用整段 episode 测量。离线异步
  `.joint.evaluate` 与有限表不直接充当在线模型。
- 联合 revision：历史 timing evaluator → 冻结可执行模型 → 重验／修订 optimizer → 联合 bundle → 安全激活；
  双组件独立验收，完整有效历史和双游标持久化。串行 scope 绑定配置白名单、horizon、idle timeout 与执行身份； transition
  profile／GPU 记账不允许被修订。当前验证只覆盖条件耗时和固定模型搜索。
- 联合执行：当前要求 whole-episode gate、session 移除和单在途 batch，启动时回读真实 SGLang 参数；失败恢复两侧源配置，清理不确定则保持
  admission 关闭。后续适配原生 SWE／Tau2 时保留其后端和任务／算法语义，独立验证恢复及布局转换；当前小模型证据仅用于链路验收。

## 后续实现：原生按需训练

2026-10-21 方案，正在实施，GPU 验收未完成；当前状态见 `PROGRESS.md`。目标来自用户提供的讨论截图：保留 `PPOTrainer` 入口，
Rollout 持续服务，Actor GPU workers 按数据需求启动／恢复／释放，后续支持独立重配置两侧。

首轮以 v2 online 模式、分离部署和固定布局为范围，先用 FSDP 回归，再接 Megatron 同布局恢复。 新增生命周期选项默认维持常驻行为，与现有
`agent.mode=online` 分开；保留原生 PPO 计算、batch／group 和 reward 语义。 CPU 协调器、Rollout、轨迹接收与持久缓冲跨
Actor 代次存活；复用 elastic 机制，不直接套用严格串行的 `ElasticLoop`。 critic／ref／teacher、LoRA
与各后端组合须明确能力范围，不能静默省略状态或改变训练语义。

| 阶段              | 主要改动                                                                                                                                                      | 验收边界                                                                                                                              |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| 1：入口与数据解耦 | `rl_trainer.py` 拆 CPU／Rollout 与 Actor 初始化；从 Rollout 准备 batch；分开数据绑定与权重连接。复用 callback／workflow，补持久轨迹交接、幂等 ID 和有界缓冲。 | 原生入口先提供推理，Actor 无 GPU 进程；完整合格 batch 才唤醒；重复通知与导出响应丢失不丢样本、不重复入队。                            |
| 2：固定布局闭环   | 抽取 `runtime/training` 生命周期与 checkpoint 机制供原生 trainer 调用；补 v2 完整状态恢复及 AWEX 代次重建联。                                                 | 连续完成三轮启动／训练／发布／释放，Rollout PID 不变；模型、optimizer、LR、RNG、step 与消费游标一致；同布局 FSDP／Megatron 分别验收。 |
| 3：异步服务验收   | 将整段 episode 门禁拆为训练、发布与重配置各自的边界；保留原生 staleness 语义，明确过期轨迹、背压与发布规则。                                                  | Actor 等待／恢复／训练时 Rollout 可推进；发布仅在必要安全窗口暂停；多 batch 与故障恢复不重放已提交 update。                           |
| 4：独立拓扑调整   | 先验证 Actor DP 调整、Rollout 副本增删，再扩展 TP／PP／EP；重新发现 dispatch topology，重建权重传输计划。                                                     | 每种 source→target 独立声明能力并验证状态恢复；Rollout 副本缩容先排空，TP 改变允许受控重建；不承诺任意布局转换。                      |

关键约束：

- 数据：当前在线通知在内存中，`export_trajectory` 会移除源记录。新增按需路径需可重试的 export／ack 或持久 outbox； payload
  落盘后才确认交接，ID 包含运行／服务身份，记录生成权重版本、group 和内容摘要。 原生 tensor／RTensor 需转为不依赖旧进程的持久载荷，不能直接塞入现有
  JSON 队列；队列复用其消费锁与 checkpoint 游标契约。
- 恢复：原生 `RecoverHandler` 当前拒绝 v2；新增明确的恢复适配，覆盖模型、optimizer、LR、RNG、训练进度与消费游标。
  恢复前校验模型／配置／checkpoint 身份；不通过关闭恢复检查来接通流程。
- 提交：update → 完整 checkpoint 与消费游标提交 → 发布权重并确认 serving 版本 → 空闲释放。
  发布失败只重试已提交权重；训练结果不确定时停止该代 workers，从最近提交恢复。
- 建联：当前 AWEX `/disconnect` 仅清登记信息，未调用 worker teardown；补双端通信组／缓存清理及确认，之后才销毁 Actor。
  新代次重新发现 topology／元数据、生成 transfer plan、建通信组，首次发布 full sync；connect／update／disconnect
  互斥、幂等并可查询结果。 拒绝旧代次请求，部分初始化失败须清理双方资源；不能把 RPC 超时当作远端已取消。
- 发布：当前 controller 未检查返回的 `status="error"` 就恢复生成，须补成功状态与所有目标副本的版本核验。
  部分更新失败时隔离受影响副本，用已提交完整权重修复，确认一致后再恢复服务。
- 调度与收益：首轮只在已分配 GPU 池内释放／唤起，跨作业租约与 Slurm 资源归还另验。 使用 batch readiness 与 idle
  timeout，记录启停／恢复／发布耗时、两侧 GPU·h、队列等待与服务延迟； 与常驻 Actor 在相同输入和训练语义下比较，收益尚未确认。异步执行接
  policy／revision 前需重新验证模型及执行契约。

用户已授权实现期间使用 8×H20。以下为拟运行任务，尚未启动；启动前核对现有作业、模型／运行环境与布局支持。 按阶段顺序运行，实际 GPU 编号由空闲资源确定。

初始 staleness 与生命周期规则（拟实施，阈值为验收起点，尚非收益最优参数）：

- 正确性冒烟设 `max_head_offpolicyness=0`；正式 online RL／Hermes 设 `2`。 消费前用当前 Actor
  更新前版本与参与训练的生成 token 最旧版本校验，版本差须在 `[0, S]`；多轮保留逐 token 版本，不能用最后一轮版本覆盖整条轨迹。
  超界轨迹保留审计记录并明确拒绝训练；不改版本、不补零、不静默截断。Actor 空闲／重启不推进权重版本。
- 延续原生 `(S+1)×B` 的训练流水线额度目标；首轮 `B=4` 个单样本组，`S=2` 对应最多 12 个在途／待消费训练组的额度。 原生 manager 主要约束
  workflow 提交，外部 Agent Service 请求需额外接入新训练 episode 的 admission；额度满时背压新 episode，已接纳会话继续完成。
  额度与完整 batch 就绪分别计数；消费拒绝、提交和恢复均从持久记录核对，Actor 重启不能重置仍存活 Rollout 的计数。
- 拉起：至少一个完整、持久化、奖励与版本合格的 batch 就绪，Actor 为停止状态，旧代 workers 已确认释放；首轮阈值为 1 batch，不预测性预热。
  释放过程中到达的新 batch 记录唤起需求，待旧代清理完成再启动；重复通知不重复创建 workers。
- 释放：无完整合格 batch、无在途训练／checkpoint／发布／重配置，checkpoint 与消费游标已提交且 serving 版本已确认， 连续空闲达到
  60s，且 Actor 就绪后已驻留至少 120s。空闲计时以最近一次成功训练发布／进入无完整 batch 状态为起点；零散轨迹不重置计时。 partial batch
  留在 CPU，未结束的 Agent 会话本身不阻止释放；真正断联前再次原子检查就绪状态。
- 生命周期冒烟用 idle=5s、最短驻留=0，专门覆盖三轮启停，不作为成本收益配置。 正式同口径对照固定 `S=2`，比较常驻 Actor
  与上述按需规则，报告恢复等待、过期拒绝率、服务延迟与 GPU·h；成本优化策略后续再接。

用户确认在线验收必须启动真实 v2 Agent Service：客户端 → Agent Gateway／DataProxy／Worker → Inference
Service／SGLang。 先以 `AgentRunnable` 适配轻量 GSM8K Agent，复用题目与评分逻辑；现有
`MathAgent.run(data, ...)` 不能直接当作该协议实现。 完成后接真实
`examples.hermes.hermes.HermesAgent`，验收多轮对话、受控工具调用及会话与推理 session 的对应关系。 两阶段均将真实轨迹和评分送入原生
`PPOTrainer`，执行更新、AWEX 发布及 Actor 释放／恢复；Agent Service 和 Rollout 跨 Actor 代次存活。

| 任务                | 模型与 GPU 分配（Actor＋Rollout）                              | 初始规模与目的                                                                                      |
| ------------------- | -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| 生命周期冒烟        | Qwen3-0.6B；FSDP 2＋SGLang 1                                   | 6 次 update、3 轮启停；用固定真实轨迹检查完整状态恢复、权重发布及资源释放。                         |
| 原生 online RL      | Qwen2.5-1.5B-Instruct；FSDP 4＋SGLang 2                        | 12 次 update；沿用 Hermes 的 `PPOTrainer` 在线入口，外部客户端送入 GSM8K train 问题并回传真实评分。 |
| Megatron 同布局恢复 | Qwen2.5-1.5B-Instruct；目标 `megatron:d2p2` 4＋`sglang:d2t2` 4 | 12 次 update；相同在线任务，验证不同训练／推理并行布局下的恢复、拓扑发现与 AWEX 重连。              |

在线任务采用短突发→空闲→再突发以及持续到达，固定请求内容与输入顺序；每 batch 拟用 4 条独立轨迹，保留 online 单样本组语义。 通过原生
session／completion／reward／export 通路，复用 GSM8K scorer。随后单独进行真实 Hermes 多轮／工具交互在线 RL 验收，先跑 6
次 update、3 轮启停；GSM8K 结果不代替 Hermes 验收。 恢复一致性另用同一冻结轨迹序列比较常驻／重启执行；在线生成结果不要求逐 token 相同。
在上述任务中插入重复 callback、导出确认丢失、Actor 故障、AWEX 失败与旧代次 RPC；只操作本轮自有实例。 记录样本身份／消费游标、训练与 serving
版本、状态误差、Rollout 可用性、队列／延迟、GPU·h 和清理结果；小规模测试不作学习质量提升结论。

CPU 测试扩展现有 trainer、queue、recovery、inference 和 weight-update 测试文件；GPU 验收覆盖空队列、真实在线轨迹、
同布局恢复、AWEX 重连、发布失败及重复通知。FSDP 证据不代替 Megatron／其他布局验收。

## 当前接口与文件组织

`feature/autopilot-elastic` 为主线，已整合 Expert 的独立实现、档案和新版 R10 回放；Basic 的在线执行与 revision
接口保持独立。暂不添加 benchmark、Tau2 实验入口、合成场景文件或新测试文件。联合模型测试扩展现有
`test_elastic_lifecycle_optimizer.py`；runtime 与 revision 的既有接口保留。整合验证见
[Milestone 清单](elastic/milestones/preparation-20261021/README.md)。

```text
areal/autopilot/
├── contracts.py                  # 现有公共观测／动作／结果协议
├── elastic/
│   ├── policy/
│   │   ├── basic/                # 被评估、被修订的策略
│   │   │   ├── cost_model/       # actor / rollout / transition
│   │   │   ├── joint.py         # 联合配置、固定工作量与结果
│   │   │   ├── simulator.py      # optimizer 使用的预测模拟器
│   │   │   ├── optimizer.py      # 配置与生命周期搜索
│   │   │   └── policy.py         # 生命周期决策 / 离线 JointPolicy.plan
│   │   └── expert_designed/      # 独立固定参考模型、模拟器与 R10 档案
│   │       ├── cost_model/
│   │       │   ├── actor.py      # 专家 actor 性能模型
│   │       │   ├── rollout.py    # 专家 rollout 性能模型
│   │       │   └── transition.py # 同步、启停与配置转换成本
│   │       └── simulator.py      # 工作流状态、时间与 GPU 成本
│   └── runtime/                 # 执行机制；config.py 管理申请／回执／恢复
└── revision/
    ├── contracts.py              # 修订请求和结果约定
    ├── evaluator.py              # 独立评估、反馈、接受判断
    ├── model_evaluator.py        # 实际历史输入、模型误差／覆盖验收
    ├── model_adapters.py         # Basic Actor／Rollout／Joint 预测适配
    ├── model_execution.py        # 模型工厂、子进程协议与历史验收适配
    ├── joint_evaluator.py        # 冻结联合模型、配置／开销验收
    ├── cycle.py                  # 模型→冻结联合模型→optimizer 的离线顺序编排
    ├── agent.py                  # 根据反馈提出修订
    └── loop.py                   # 组织修订和验收循环
```

仅列相关模块，省略现有辅助文件和 `__init__.py`。 expert 不含 optimizer 或 `policy.py`，也不实现
`SchedulingPolicy`。 两套模型分别持有预测公式与校准参数；专家版本在每组实验中冻结。 evaluator 指导修订，agent 修改 Basic；对比
Basic 各轮版本与固定 expert 的同口径结果。

Basic 已支持固定窗口上的联合配置枚举：共享推理服务通道、外部等待、Actor 消费、批次 staleness、发布／checkpoint
和显式切换／收尾成本。各候选保持训练语义与工作量；按总 GPU seconds 搜索，支持完整作业时间上限与候选预算。 详细输入、公式和近似边界见
[Basic 接口](../../areal/autopilot/elastic/policy/basic/README.md)。 历史模型 evaluator
已接通候选源码工厂、worker 与 revision loop，支持阶段测量先行、固定划分、误差／覆盖及同样本验收； 给定 packing
或环境等待的诊断与完整预测明确区分。联合 optimizer 已接通固定模型／场景／预算上的源码修订，支持逐场景成本与搜索开销验收、恢复指纹检查； 两类修订已通过
`PolicyRevisionCycle` 串联：工厂显式导出 `.joint`，历史预测与联合表双 worker 复核，按新模型重新评估 optimizer；
阶段恢复保留总轮数预算且不重复已完成的 agent 调用。本地固定 Rollout 的历史采集、自动触发与 bundle 激活已接通；联合自动执行独立接入。
联合修订部署已接入同一串行 fixture；高精度和收益仍需独立实验。接口和限制见
[revision](../../areal/autopilot/revision/README.md)。高精度和实际收益需公司端真实数据验证。

## 代码入口与复核

- [Basic policy](../../areal/autopilot/elastic/policy/README.md)：actor 模型与配置／生命周期搜索。
- [revision](../../areal/autopilot/revision/README.md)：历史模型与联合配置 optimizer
  的离线修订、固定上下文验收。
- [runtime 与历史证据](elastic/README.md)：已验证 pure-DP 执行、SWE 基础设施；当前先在同一小模型环境打通全链路。

以下命令从仓库根目录运行，无需公司数据或模型 API：

```bash
python -S areal/autopilot/elastic/policy examples/autopilot/elastic/lifecycle_scenario.json
python tests/run_autopilot_cpu.py -q \
  tests/test_elastic_basic_policy.py tests/test_elastic_lifecycle_optimizer.py \
  tests/test_autopilot_actor_cost.py tests/test_autopilot_revision.py \
  tests/test_autopilot_revision_llm.py tests/test_elastic_loop.py \
  tests/test_elastic_policy.py tests/test_elastic_queued_trainer.py \
  tests/test_elastic_batch_queue.py tests/test_elastic_recovery.py
pre-commit run --all-files
```

CPU 回归需要 Python 3.12+、pytest、NumPy、OpenAI SDK 及其 HTTP 依赖；完整 pre-commit 的 CLI
文档生成还会导入训练依赖。 CPU runner 仅绕过顶层 `areal` 的训练栈导入，不模拟 GPU；真实引擎验证需公司端环境。
