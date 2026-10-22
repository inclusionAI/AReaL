# Tau2 计划与公司端交接

范围：S031、S029、T0390、S026、S030、S028、S025、S024、S013。当前状态统一维护在
[PROGRESS.md](../../PROGRESS.md)，Expert 接口与边界见
[Reference](../../areal/elastic/policy/reference/README.md)。设计长文与历史进度保留在本地归档。

[M2 历史快照](../../../artifacts/repository-archive-20261021/files/examples/autopilot/elastic/milestones/m2/README.md)
保留旧串行执行器的代码身份与证据。 [架构图](elastic/README.md#policy-and-revision-loops) 区分 policy
loop、cost-model revision 和 optimizer revision；当前执行入口为原生 PPOTrainer，联合配置模型保留作离线参考。

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

执行统一为 `PPOTrainer → OnDemandActor`；删除旧串行训练器、Rollout 包装和小模型 fixture 入口。 共享
checkpoint、队列、在线 inbox 与进程所有权移入 `areal/elastic/state/`。

- AReaL 的 `PolicyAdapter` 统一接默认超时／实测 Basic；AutoPilot 的 `RevisionAdapter` 接修订和独立验收后的
  bundle；默认 Monitor 统一提供观测、历史和测量恢复，旧串行 bundle 拒绝部署到原生 scope。
- `ConfigApplication` 保留 CAS、持久意图／回执和恢复；原生 executor 只接受相同固定布局，独立拓扑调整待实现。
- `RuntimeRevisionLoop` 使用显式 host 协议；按完整 update 划分校准／验收并保留预算。原生首轮限定
  horizon=1、固定布局，阶段误差验收不代表异步完成时间或收益。
- Basic／Expert、离线联合模型、历史 evaluator 和证据保留；旧执行器的 GPU 结果不作为原生路径验收。

## 后续实现：原生按需训练

2026-10-21 方案；固定布局主流程已实施和 GPU 验收，故障闭环与完整结果见 `PROGRESS.md`。目标来自用户提供的讨论截图：保留 `PPOTrainer`
入口， Rollout 持续服务，Actor GPU workers 按数据需求启动／恢复／释放，后续支持独立重配置两侧。

首轮以 v2 online 模式、分离部署和固定布局为范围，先用 FSDP 回归，再接 Megatron 同布局恢复。 新增生命周期选项默认维持常驻行为，与现有
`agent.mode=online` 分开；保留原生 PPO 计算、batch／group 和 reward 语义。 CPU 协调器、Rollout、轨迹接收与持久缓冲跨
Actor 代次存活；复用 elastic 机制，不直接套用严格串行的 `ElasticLoop`。 critic／ref／teacher、LoRA
与各后端组合须明确能力范围，不能静默省略状态或改变训练语义。

| 阶段              | 主要改动                                                                                                                                                      | 验收边界                                                                                                                              |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| 1：入口与数据解耦 | `rl_trainer.py` 拆 CPU／Rollout 与 Actor 初始化；从 Rollout 准备 batch；分开数据绑定与权重连接。复用 callback／workflow，补持久轨迹交接、幂等 ID 和有界缓冲。 | 原生入口先提供推理，Actor 无 GPU 进程；完整合格 batch 才唤醒；重复通知与导出响应丢失不丢样本、不重复入队。                            |
| 2：固定布局闭环   | 原生 OnDemandActor 管理生命周期；共享持久状态置于 elastic/state；补 v2 完整状态恢复及 AWEX 代次重建联。                                                       | 连续完成三轮启动／训练／发布／释放，Rollout PID 不变；模型、optimizer、LR、RNG、step 与消费游标一致；同布局 FSDP／Megatron 分别验收。 |
| 3：异步服务验收   | 将整段 episode 门禁拆为训练、发布与重配置各自的边界；保留原生 staleness 语义，明确过期轨迹、背压与发布规则。                                                  | Actor 等待／恢复／训练时 Rollout 可推进；发布仅在必要安全窗口暂停；多 batch 与故障恢复不重放已提交 update。                           |
| 4：独立拓扑调整   | 先验证 Actor DP 调整、Rollout 副本增删，再扩展 TP／PP／EP；重新发现 dispatch topology，重建权重传输计划。                                                     | 每种 source→target 独立声明能力并验证状态恢复；Rollout 副本缩容先排空，TP 改变允许受控重建；不承诺任意布局转换。                      |

关键约束：

- 数据：按需路径需可重试的 export／ack 或持久 outbox； payload 落盘后才确认交接，ID 包含运行／服务身份，记录生成权重版本、group
  和内容摘要。 原生 tensor／RTensor 需转为不依赖旧进程的持久载荷，不能直接塞入现有 JSON 队列；队列复用其消费锁与 checkpoint 游标契约。
- 恢复：按需路径提供 v2 完整状态适配，覆盖模型、optimizer、LR、RNG、训练进度与消费游标。 恢复前校验模型／配置／checkpoint
  身份；不通过关闭恢复检查来接通流程。
- 顺序：update → 暂停 Rollout → 发布并确认 serving 版本 → 按周期保存 → 评估／清理／恢复 Rollout → 判断释放。
  释放前补存本轮结束后的完整状态与消费游标，包括周期保存之后的 CPU／RNG 变化。意外故障从最近 checkpoint 恢复并重放未保存 batch， 拒绝未编入
  batch、来自已放弃权重分支的轨迹；发布失败不再意味着该次 update 已持久化。
- 建联：AWEX `/disconnect` 清理双端通信组／缓存并确认，之后才销毁 Actor；通信不确定时先证明故障训练代次已停止，再清理推理端。 新代次重新发现
  topology／元数据、生成 transfer plan、建通信组，首次发布 full sync；connect／update／disconnect
  互斥、幂等并可查询结果。 拒绝旧代次请求，部分初始化失败须清理双方资源；不能把 RPC 超时当作远端已取消。
- 发布：controller 检查成功状态及所有目标副本的版本，再恢复生成。 部分更新失败时隔离受影响副本，用已提交完整权重修复，确认一致后再恢复服务。
- 调度与收益：首轮只在已分配 GPU 池内释放／唤起，跨作业租约与 Slurm 资源归还另验。 使用 batch readiness 与 idle
  timeout，记录启停／恢复／发布耗时、两侧 GPU·h、队列等待与服务延迟； 与常驻 Actor 在相同输入和训练语义下比较，收益尚未确认。异步执行接
  policy／revision 前需重新验证模型及执行契约。

用户已授权实现期间使用 8×H20。以下为验收任务定义；实际配置、失败记录与结果见 `PROGRESS.md` 链接的 GPU
证据。运行前核对现有作业、模型／环境与布局支持，只操作本轮自有资源。

初始 staleness 与生命周期规则（拟实施，阈值为验收起点，尚非收益最优参数）：

- 正确性冒烟设 `max_head_offpolicyness=0`；正式 online RL／Hermes 设 `2`。 消费前用当前 Actor
  更新前版本与参与训练的生成 token 最旧版本校验，版本差须在 `[0, S]`；多轮保留逐 token 版本，不能用最后一轮版本覆盖整条轨迹。
  超界轨迹保留审计记录并明确拒绝训练；不改版本、不补零、不静默截断。Actor 空闲／重启不推进权重版本。
- 延续原生 `(S+1)×B` 的训练流水线额度目标；首轮 `B=4` 个单样本组，`S=2` 对应最多 12 个在途／待消费训练组的额度。 原生 manager 主要约束
  workflow 提交，外部 Agent Service 请求需额外接入新训练 episode 的 admission；额度满时背压新 episode，已接纳会话继续完成。
  额度与完整 batch 就绪分别计数；消费拒绝、提交和恢复均从持久记录核对，Actor 重启不能重置仍存活 Rollout 的计数。
- 拉起：至少一个完整、持久化、奖励与版本合格的 batch 就绪，Actor 为停止状态，旧代 workers 已确认释放；首轮阈值为 1 batch，不预测性预热。
  释放过程中到达的新 batch 记录唤起需求，待旧代清理完成再启动；重复通知不重复创建 workers。
- 释放：无完整合格 batch、无在途训练／checkpoint／发布／重配置，serving 版本已确认， 连续空闲达到 60s，且 Actor 就绪后已驻留至少
  120s。空闲计时以最近一次成功训练发布／进入无完整 batch 状态为起点；零散轨迹不重置计时。 partial batch 留在 CPU，未结束的 Agent
  会话本身不阻止释放；决定释放后补齐最新 checkpoint 与消费游标，保存后再次检查 batch 就绪状态。
- 生命周期验收用 idle=5s、最短驻留=0，专门覆盖启停，不作为成本收益配置。 后续收益对照固定 `S=2`，比较常驻 Actor
  与上述按需规则，报告恢复等待、过期拒绝率、服务延迟与 GPU·h；成本优化策略后续再接。

用户确认在线验收必须启动真实 v2 Agent Service：客户端 → Agent Gateway／DataProxy／Worker → Inference
Service／SGLang。 先以 `AgentRunnable` 适配轻量 GSM8K Agent，复用题目与评分逻辑；现有
`MathAgent.run(data, ...)` 不能直接当作该协议实现。 完成后接真实
`examples.hermes.hermes.HermesAgent`，验收多轮对话、受控工具调用及会话与推理 session 的对应关系。 两阶段均将真实轨迹和评分送入原生
`PPOTrainer`，执行更新、AWEX 发布及 Actor 释放／恢复；Agent Service 和 Rollout 跨 Actor 代次存活。

| 任务                | 模型与 GPU 分配（Actor＋Rollout）                         | 初始规模与目的                                                                                      |
| ------------------- | --------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| 生命周期冒烟        | Qwen3-0.6B；FSDP 2＋SGLang 1                              | 6 次 update、3 轮启停；用固定真实轨迹检查完整状态恢复、权重发布及资源释放。                         |
| 原生 online RL      | Qwen2.5-1.5B-Instruct；FSDP 4＋SGLang 2                   | 12 次 update；沿用 Hermes 的 `PPOTrainer` 在线入口，外部客户端送入 GSM8K train 问题并回传真实评分。 |
| Megatron 同布局恢复 | Qwen2.5-1.5B-Instruct；`megatron:d2p2` 4＋`sglang:d2t2` 4 | 12 次 update；B8满足PP最小微批要求，验证不同训练／推理并行布局下的恢复、拓扑发现与 AWEX 重连。      |
| Hermes 多轮工具     | Qwen3-4B-Instruct-2507；FSDP4＋SGLang1                    | 原生64K服务、24个双轮episode、6更新，至少三代Actor；实际任务失败保持0分。                           |

在线任务采用短突发→空闲→再突发以及持续到达，固定请求内容与输入顺序；每 batch 拟用 4 条独立轨迹，保留 online 单样本组语义。 通过原生
session／completion／reward／export 通路，复用 GSM8K scorer。随后单独进行真实 Hermes 多轮／工具交互在线 RL 验收，先跑 6
次 update、3 轮启停；GSM8K 结果不代替 Hermes 验收。 恢复一致性另用同一冻结轨迹序列比较常驻／重启执行；在线生成结果不要求逐 token 相同。
在上述任务中插入重复 callback、导出确认丢失、Actor 故障、AWEX 失败与旧代次 RPC；只操作本轮自有实例。 记录样本身份／消费游标、训练与 serving
版本、状态误差、Rollout 可用性、队列／延迟、GPU·h 和清理结果；小规模测试不作学习质量提升结论。

CPU 测试扩展现有 trainer、queue、recovery、inference 和 weight-update 测试文件；GPU 验收覆盖空队列、真实在线轨迹、
同布局恢复、AWEX 重连、发布失败及重复通知。FSDP 证据不代替 Megatron／其他布局验收。

## 当前接口与文件组织

```text
areal/trainer/rl_trainer.py             # 唯一 PPO 执行与每轮工作顺序
areal/elastic/runtime.py               # Actor 生命周期、发布、保存与恢复
areal/elastic/{state,control,monitor}/   # 持久状态、控制事务与运行观测／实测
areal/v2/training_service/{checkpoint,runtime_state}.py # 原生状态读写与各 rank LR／RNG
areal/elastic/policy/basic/             # Basic 模型／simulator／optimizer
areal/elastic/policy/reference/         # 独立 Expert 对照
areal/autopilot/revision/               # 通用 proposal／评测编排／接受／store
areal/autopilot/domains/elastic/        # Elastic evaluator、worker、bundle 与 native bridge
```

暂不添加 benchmark、Tau2 实验入口、合成场景或新测试文件；扩展既有测试。 Expert 不含 optimizer、不实现调度策略；实验中冻结，Basic
不依赖其实现。 详细接口与限制见
[执行文档](elastic/README.md)、[Basic](../../areal/elastic/policy/basic/README.md) 和
[revision](../../areal/autopilot/revision/README.md)。

## 代码入口与复核

模型与规划测试入口：

```bash
python tests/run_autopilot_cpu.py -q \
  tests/test_elastic_basic_policy.py tests/test_elastic_lifecycle_optimizer.py \
  tests/test_autopilot_actor_cost.py tests/test_autopilot_revision.py
```

原生协议测试使用完整训练依赖的 Python 环境；fake worker 不能代替 GPU 验收：

```bash
python -m pytest -q tests/test_elastic_loop.py tests/test_elastic_checkpoint.py \
  tests/test_elastic_batch_queue.py tests/test_elastic_recovery.py
pre-commit run --all-files
```

历史结果、文献材料和 R10
数据已移入[本地归档](../../../artifacts/repository-archive-20261021/README.md)，运行代码、基础 profile
和必要测试输入保留。 R10 档案测试需显式设置 `AREAL_TEST_EXPERT_CATALOG`；默认测试不依赖此机器的归档目录。后续逐项评估开发期间需回放入库的材料。
离线 CLI、R10 回放辅助代码及专用验证入口另见
[开发材料归档](../../../artifacts/elastic-policy-development-20261021/README.md)。
