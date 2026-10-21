# Related work：两条主线与三个相关性层级

截至 2026-10-21。引用键对应 [references.bib](references.bib)；版本、来源及分层记录见
[sources.json](sources.json)，检索范围见
[SEARCH.md](SEARCH.md)。以下机制和评测范围来自原始论文；对本项目的判断属于分析。本轮未复现实验。

## 分层与术语

- **直接相关：** Agentic online RL post-training，核验多轮工具／环境交互中生成新轨迹并用于训练的工作；同时记录它实际解决的是 GPU
  配置、请求调度还是环境资源问题。
- **Level 1：** 其他 RL post-training，如单轮推理、同步 RLHF；比较相关的配置调整、校准及修订机制。
- **Level 2：** 通用训练、serving、数据库或算法发现；比较可迁移的方法与新增约束。
- **online 两种证据：** 训练过程中生成新轨迹，与部署后持续接收真实交互分别记录。下表 agentic benchmark 证明前者；不能据此推断后者。
- **policy：** 本项目指系统资源管理策略，包含性能模型与规划算法。RL 模型权重、agent harness、训练超参数与系统策略代码分别标注。

## R1：资源与配置调整

### 直接相关：联合资源与配置的主要参照

| 工作／引用键与定位                                                                                                  | 调整什么；反馈与触发                                                                                                       | 如何执行；评测与比较边界                                                                                                                                                             |
| ------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [Libra](https://arxiv.org/html/2606.03077v3#S4) · libra；联合布局与分配                                             | §4 联合训练 TP／EP／PP／DP、rollout TP buckets 和两侧 GPU 数；周期读取轨迹统计，以成本估计搜索，收益超过转换成本才重分配。 | §4.4 用 core＋hybrid pool 吸收短时变化；持续变化才大规模重划分。§7 包含 Search-R1、R2E-Gym；GPU 总预算固定，还含请求路由优化。最接近联合配置问题。                                   |
| [DynaRL，OSDI 2026](https://www.usenix.org/system/files/osdi26-wang-yuanqing.pdf#page=9) · dynarl；动态多组件分配   | §5.1 以持续低利用率找 donor，在线回归更新吞吐估计，选预测全局吞吐严格改善的重分配；另有请求级优先调度。                    | §4.2 在可中断点迁移状态；§7.1–7.2／Fig. 12 明确评测 rstar2-agent 多轮工具任务，因此列入直接层。§7.6 检查训练语义／效果，不能简化成“固定估计器”。                                     |
| [RLinf，OSDI 2026](https://www.usenix.org/system/files/osdi26-yu-chao.pdf#page=9) · rlinf；执行规划与重新 profiling | §3.4 profiling＋DAG 规划；持续预测偏差触发 checkpoint、重新 profiling／规划和恢复。                                        | §5.1.2（印刷页 839–840）有 Search-R1 多轮检索评测，纳入直接层；该实验展示 agentic 执行效率，未单独证明重新 profiling 在该任务上的增量收益。                                          |
| [QwenGyre](https://arxiv.org/html/2609.33848v1#S4) · qwengyre；资源角色切换                                         | §4 根据数据供给、水位和训练边界调整 core／satellite cells 的训练或 rollout 角色；core 持有优化器。                         | 保留后端进程／通信组，offload 非活跃状态；§6 用 NL2RepoBench、DeepSWE、TerminalBench。训练布局受 cell 结构约束；§5 轨迹处理优化需与资源收益分开。                                    |
| [PEARL](https://arxiv.org/html/2609.35158v1#S4) · pearl；rollout PD 与弹性                                          | §4 根据负载、环境等待和可用 GPU，选择 prefill／decode 分离或共置及 worker 数；使用预测收益阈值与 hysteresis。              | 尽量减少角色变更，复用进程、保留部分轨迹；§6 在 SWE-bench 上回放资源供给 trace。训练 worker 数和绑定固定，可借用其空闲 GPU；未联合搜索任意 actor 布局。                              |
| [TideRL](https://arxiv.org/html/2608.10402v1#S5) · tiderl；就绪感知调度                                             | §5 根据 ready samples、处理速率和等待队列，选择 Ref–Actor 执行方式并迁移 rollout／trainer rank。                           | 在同步边界执行调整；§6 含 WebShop、OSWorld。batch 可在边界内弹性变化，不能将其全部收益归入固定 batch 的资源优化。                                                                    |
| [SeamlessFlow](https://arxiv.org/html/2508.11553v1#S4) · seamlessflow；训练节点复用                                 | §4 capability tags 按数据供给切换节点角色，空闲训练节点用于 rollout；roofline 信息指导硬件优先级。                         | 数据面支持 partial rollout 暂停／恢复；§5 含 SWE 训练与评测。主要是作业内角色复用，归还外部 allocation 的效果需另证。                                                                |
| [ROSE](https://arxiv.org/html/2605.06534v2#S4) · rose；外部共享容量                                                 | §4 依据 serving SLO、并发和 cache affinity，将 rollout 路由到专用与可借用 serving GPU。                                    | 共享计算／显存并传输权重；§6 用 FrozenLake、ALFWorld，serving 侧有生产 trace 回放。训练 workload 仍是 benchmark；没有任意 actor 布局联合优化。                                       |
| [ReFlow](https://arxiv.org/html/2610.21640v1#S5) · reflow；rollout／trainer 重平衡                                  | §6／Alg. 1 用生产／消费速率和 idle 比例估算容量，在完整副本组约束下搜索分配；冷却限制切换。另用 AIMD 调整 rollout 并发。   | §5 预热双角色 worker，保持分片并通过局部广播加入训练副本；通常在权重同步时重平衡。§7 包含 APPS 与 ReTool。引入 experience replay，必须单列数据复用语义；全系统吞吐不是纯重配置增益。 |

### 直接场景中的局部问题

| 工作／引用键                                                    | 已有机制与证据                                                                                                       | 与联合 rollout／actor 配置的距离                                                                             |
| --------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| [ARL-Tangram](https://arxiv.org/html/2603.13019v1#S4) · tangram | action 级管理 CPU、工具／reward 等外部异构资源；§4–6 讨论调度、伸缩和 agent 任务。                                   | 同一类 agentic 负载，主要优化外部资源，作为瓶颈分析和局部资源管理参照。                                      |
| [VenusRL](https://arxiv.org/html/2610.03286v1#S4) · venusrl     | group 完成优先的 action 调度、sandbox 内存准入及模板共享；§6 用 OpenSWE。                                            | 评测固定 16 GPU 生成＋16 GPU 训练；其效果不证明动态 GPU 分配有效。                                           |
| [WAR](https://arxiv.org/html/2607.17299v1) · war                | “WAR System Design” 根据 rollout 负载使用 suffix speculative decoding 与 cache-aware 请求调度；多轮 agent 在线采样。 | 同步执行仍属于在线采样的 agentic 场景。调整的是 rollout 执行／路由策略，未展示 actor／rollout GPU 联合伸缩。 |

以上 12 篇工作有目标场景或其直接子问题的证据，但不能一并当作完整联合重配置方案。已核验的评测没有单独建立“持续真实用户交互到达时，同时联合优化 actor／rollout
配置”的端到端结果；对部署在线场景的覆盖仍需逐项补证。

### Level 1：其他 RL post-training

| 工作／引用键                                                                             | 方法与原文位置                                                                                                 | 为什么保留／迁移限制                                                                                                 |
| ---------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| [Nereus](https://arxiv.org/html/2609.34645v1#S4) · nereus                                | §4 在线校准操作效率、搜索 TP／PP／DP，并判断转换峰值内存与收益回收时间；§7.1 使用 OpenRLHF prompt collection。 | 强普通自适应基线。§7.3 模型误差验证限训练阶段；动态实验在固定 32-GPU allocation 内释放／复用，不等同云资源实际退租。 |
| [RLHFless](https://arxiv.org/html/2602.22718v1#S4) · rlhfless                            | §4 同步 serverless RLHF，按成本调整生成副本，配合 prefill 去重和 cut-and-migrate。                             | sampling actor 指生成副本；训练优化器组缩零与完整 agentic 队列不是其已验证对象。真机与模拟结果分开。                 |
| [RLBoost，NSDI 2026](https://www.usenix.org/system/files/nsdi26-wu-yongji.pdf) · rlboost | rollout 利用可抢占实例，支持权重拉取及部分请求恢复；§6 用 OpenR1-Math，附录 C 回放资源供给 trace。             | 给出外部供给弹性机制；价格／供给假设与独占固定 GPU 池不同。                                                          |
| [ReSpec](https://arxiv.org/html/2510.26475v1#S5) · respec                                | §5.2 离线 profiling 拟合 SD 性能模型，随 active batch 切换 SD 模式／参数；§5.3 在线训练 drafter。              | 证明配置控制和模型权重适应可以并存；drafter 训练不等于性能模型或 planner 代码修订。                                  |
| [ReaL](https://arxiv.org/html/2406.14088v2#S4) · realhf                                  | §4–5 成本估计／execution-plan search，按调用重分配参数与并行布局。                                             | 可比较规划与布局转换，不能把执行预定计划误写成持续修订算法。另有 MLSys 2025 正式版。                                 |
| [HybridFlow](https://arxiv.org/html/2409.19256v2) · hybridflow                           | 混合控制流和训练／生成间高效重分片；设计部分。                                                                 | 执行机制参照；不据此主张已有 LLM 在线改写调度逻辑。                                                                  |
| [StreamRL](https://arxiv.org/html/2504.15930v1#S4) · streamrl                            | §4–5 流式重叠、阶段平衡、长度预测与长尾调度。                                                                  | 提供分离／异步执行基线；多轮工具返回后的负载变化需要另核。                                                           |

### Level 2：通用训练与 serving

| 工作／引用键                                                                                                | 可迁移机制                                                            | 到本项目新增的约束                                                           |
| ----------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| [Pollux](https://arxiv.org/html/2008.12260v2) · pollux                                                      | 在线拟合 goodput，联合资源与训练配置。                                | 学习效率应进入目标；其 batch／学习率适应与首轮固定训练语义需分开。           |
| [DynaTrain](https://arxiv.org/html/2605.18815v1#S4) · dynatrain                                             | Virtual Parameter Space、状态路由和新通信世界构建支持并行布局热切换。 | 是转换执行层；还需 RL 的数据版本、队列和权重发布约束。                       |
| [OOCO](https://arxiv.org/html/2511.21862v1#S3) · ooco                                                       | roofline 性能模型指导在线／离线 serving 混部、抢占与任务迁移。        | online 在这里指低延迟请求；其 SLO 约束需要替换／扩展为 RL 的时效和学习约束。 |
| [Vidur](https://arxiv.org/html/2405.05465v2)、[vTrain](https://arxiv.org/html/2312.12391v2) · vidur、vtrain | profiling 驱动的推理／训练模拟和配置搜索。                            | 组件精度不能替代完整 RL 异步队列、状态转换和跨配置验证。                     |

## R2：LLM-driven online revision of system policy

### 直接相关：本轮未确认完整的同场景实证

核验标准是：**agentic 在线采样训练中，LLM 根据系统运行反馈修改性能模型结构或规划算法，候选经评估后投入运行，并报告包含修订开销的收益。**
这比“参数更新”“重新选择配置”或“RL policy 更新”更具体。

| 已检查的直接场景工作                                              | 实际发生的变化                                                                             | 对修订问题的证据边界                                                                                               |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| Libra、DynaRL、RLinf                                              | 新统计驱动重新规划、参数回归更新或重新 profiling；位置见 R1。                              | 支持强自适应基线；已核验设计段落未报告上述 LLM 系统策略代码修订闭环。                                              |
| QwenGyre、PEARL、TideRL、ReFlow                                   | 角色、布局／执行模式、batch 或并发状态随反馈变化。                                         | 调度程序可以持续做动态决策；这本身不证明程序在被 LLM 改写。                                                        |
| [AReaL 2.0](https://arxiv.org/html/2607.01120v2#S5) · areal2      | §5–6 讨论 agent 权重／harness 演化控制面，§6 描述动态分配／释放资源。                      | 最贴近真实部署在线学习的上游定位；演化对象需与资源 planner／性能模型区分。报告不提供本项目所述联合修订的完整实证。 |
| [OpenTinker](https://arxiv.org/html/2601.07376v2#S3) · opentinker | §3 管理多 LoRA policy 的训练、版本快照、sampler refresh 和 adapter-local optimizer state。 | policy lifecycle 指被训练的 agent policy；不是系统调度算法的代码演化。                                             |

“未确认”限于 [检索范围](SEARCH.md) 和所核验设计／评测，不能作为全球不存在相关工作的证明。

### Level 1：其他 RL post-training 中的相近适应方法

| 工作／引用键                                                | 修订／更新对象和验证                                                                                                            | 关键区别                                                                                                       |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| [LLMZero](https://arxiv.org/html/2606.18388v2#S3) · llmzero | §3 LLM 诊断训练曲线，在 checkpoint 搜索树上提出学习率、KL、batch 等协调变更并 early-stop；§4 在四个 GRPO 数据集按训练预算比较。 | 这是 LLM 驱动训练策略搜索；§3.3 仅恢复模型权重并重置优化器／数据位置，不能作为保持训练状态的资源规划修订证据。 |
| [ReSpec](https://arxiv.org/html/2510.26475v1#S5) · respec   | 在线更新 drafter 权重，SD 参数由已有性能模型选择。                                                                              | 被更新的预测模型是 token drafter，不是系统耗时模型；需要避免名称混淆。                                         |
| [Nereus](https://arxiv.org/html/2609.34645v1#S4) · nereus   | 运行反馈校准效率参数，固定规划算法重新决策。                                                                                    | 非 LLM 的必要参照，检验常规校准是否已经足够；不归为 LLM 代码修订。                                             |

### Level 2：serving、数据库与算法发现

| 工作／引用键                                                                                                       | 修订对象、反馈与部署方式                                                                                                   | 对本项目的影响                                                                                                         |
| ------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| [Autopoiesis](https://arxiv.org/html/2604.07144v1#S5) · autopoiesis                                                | §5–6 LLM 演化 serving 的 schedule／should_reschedule 代码；近期 trace 模拟评估，计入调度耗时，持续 warm-start 并替换策略。 | 最接近在线算法修订的方法先例。§7／附录 H 的构造 workload 和资源供给 trace 有明确范围；不是 RL 学习／staleness 的验证。 |
| [ADRD](https://arxiv.org/html/2604.06566v1#S4) · adrsdb                                                            | §4 共同演化 buffer simulator 与 eviction policy；§5 以真实数据库表现校验并改进代理 evaluator。                             | 模型／评估方法与策略共同改进已有先例；研究式迭代的时间尺度、真实数据库校验与在线 RL 闭环需逐项比较。                   |
| [AlphaEvolve](https://arxiv.org/html/2506.13131v1#S3.SS3.SSS1) · alphaevolve                                       | §3.3.1 在历史数据中心快照模拟器中演化调度启发式，留出验证后部署。                                                          | 提供生成、评估、部署链路；生产部署不自动等于运行时持续演化。                                                           |
| [ADRS](https://arxiv.org/html/2510.06189v3#S4) · adrs                                                              | §4–5 generate–evaluate–refine，依赖可执行 evaluator／模拟器发现系统算法。                                                  | 关注评估可信度与预算；不将离线发现算法等同在线修订。                                                                   |
| [AdaEvolve](https://arxiv.org/html/2602.20133v1)、[EvoX](https://arxiv.org/html/2602.23413v2#S4) · adaevolve、evox | 前者自适应分配发现过程的搜索预算；后者外层演化搜索策略、内层演化候选解。                                                   | 可作为 revision 方法参照；发现过程的搜索策略与运行时资源 planner 分别定义。                                            |

## 仅用于背景与问题证据

[AReaL](https://arxiv.org/html/2505.24298v5)（areal）说明异步执行与 staleness
约束；[Agent Lightning](https://arxiv.org/html/2508.03680v1)（agentlightning）说明
trainer–agent 解耦。二者不单独构成一类 broad related work。

[TVCACHE](https://arxiv.org/html/2602.10986v1#S3)（tvcache）针对工具等待做状态化结果缓存，支持分析外部等待这一瓶颈；其缓存优化不作为动态
GPU 配置或 LLM 系统策略修订的解法。OpenTinker 与 AReaL 2.0 的角色见 R2 直接层。

## 代码与证据范围

代码核查沿用指定版本的静态阅读，未运行作者代码。论文能力、代码接口与实测结果分别解释：

- Libra，7aeb7c0：[planner](https://github.com/NetX-lab/Libra/blob/7aeb7c0424984b1f8e19ad7e7b231bb636e530e4/infra/cost_model/global_resource_planner.py#L677)减去配置的转换成本；[calibrator](https://github.com/NetX-lab/Libra/blob/7aeb7c0424984b1f8e19ad7e7b231bb636e530e4/infra/cost_model/calibrator.py#L70)含在线参数更新。未确认这些代码路径全部用于论文评测；附录
  B 的 CE profiling 与 trie 更新不能混写。
- ADRD，609594ca：[buffer README](https://github.com/audreyccheng/adrd/blob/609594ca4f8d2c9c9c49113183d28c7836e3acc5/buffer_cache/README.md)有双层流程，但
  PostgreSQL
  子模块标为待链接；[index outer loop](https://github.com/audreyccheng/adrd/blob/609594ca4f8d2c9c9c49113183d28c7836e3acc5/index_selection/outer_loop/outer_loop.py#L69)对照真实评分分析代理排名偏差。未做端到端复现。

论文报告的吞吐提升不跨硬件／任务直接排名；重配置、轨迹处理、batch 改变与样本复用的收益分别归因。本项目的跨配置净收益与 LLM 修订增益仍待独立实验。
