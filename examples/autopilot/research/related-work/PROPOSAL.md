# AReaL Elastic Online RL AutoPilot

## Proposal

**Area:** Asynchronous, disaggregated agentic online RL post-training; elastic training;
LLM-driven online revision of policy.

这里的 policy 指系统资源管理策略，包含性能模型与规划算法。online 证据区分训练内在线采样与部署后的持续真实交互；文献的具体覆盖见
[证据矩阵](EVIDENCE.md)。

### Motivation & Related Work

Agentic online RL post-training 的多轮交互、工具等待和轨迹长度变化，会影响 rollout
与训练阶段的负载及瓶颈，但不同任务和资源条件下的可获收益并不相同。因此，我们研究在保持学习质量、数据时效和训练语义的约束下，动态调整 rollout／actor
资源与配置，能否获得覆盖测量、规划和重配置开销的净收益。[Libra](https://arxiv.org/html/2606.03077v3#S2)、[TideRL](https://arxiv.org/html/2608.10402v1#S2)

进一步，我们研究固定规划算法结合普通在线校准的适用边界，模型结构或搜索方法的局限可能持续产生决策损失。我们探索利用 LLM／agent
根据运行反馈修订性能模型与规划算法，并通过独立测量，检验修订能否在满足学习质量与数据时效约束、计入修订和验证开销后，相对强自适应基线带来额外净收益。

#### 资源与配置调整

**直接相关：Agentic online RL post-training。** Libra 联合选择训练布局、rollout 配置与资源划分；DynaRL
通过在线回归和动态调度在组件间重分配资源；RLinf 支持执行规划及偏差触发的重新 profiling，并有多轮检索任务评测。QwenGyre 切换 elastic cells
的角色，PEARL 调整 rollout 的 PD 执行方式，ReFlow 用运行速率反馈重平衡
rollout／trainer。已有机制覆盖了联合规划、反馈适应和执行转换；仍需在相同
workload、动作集合和训练约束下比较净收益。[Libra](https://arxiv.org/html/2606.03077v3#S4)、[DynaRL](https://www.usenix.org/conference/osdi26/presentation/wang-yuanqing)、[RLinf](https://www.usenix.org/conference/osdi26/presentation/yu-chao)、[QwenGyre](https://arxiv.org/html/2609.33848v1#S4)、[PEARL](https://arxiv.org/html/2609.35158v1#S4)、[ReFlow](https://arxiv.org/html/2610.21640v1#S6)

同场景的调整也有不同范围：TideRL 结合资源伸缩和 batch 弹性，SeamlessFlow 将空闲训练资源用于 rollout，ROSE 借用 serving
空闲容量。这些机制的训练语义、可用资源和转换代价需要分别匹配；ARL-Tangram、VenusRL、WAR
所处理的外部资源／请求调度子问题另见矩阵。[TideRL](https://arxiv.org/html/2608.10402v1#S5)、[SeamlessFlow](https://arxiv.org/html/2508.11553v1#S4)、[ROSE](https://arxiv.org/html/2605.06534v2#S4)

**Level 1：其他 RL post-training。** Nereus 将在线校准、布局搜索与转换收益判断结合，构成普通自适应方法的重要参照；RLHFless 和
RLBoost 分别研究 serverless 与可抢占资源下的 rollout 扩展。它们的单轮／同步设定或资源来源，与多轮 agentic
联合配置的假设需逐项比较。[Nereus](https://arxiv.org/html/2609.34645v1#S4)、[RLHFless](https://arxiv.org/html/2602.22718v1#S4)、[RLBoost](https://www.usenix.org/conference/nsdi26/presentation/wu-yongji)

**Level 2：其他训练与 serving 系统。** Pollux 提供 goodput 驱动的资源与训练配置适应，DynaTrain
提供布局切换和状态迁移机制，OOCO 根据性能模型调度在线／离线 serving 混部。这些方法为目标、规划和执行提供参照，迁移时需纳入 RL
的队列、版本同步和学习质量约束。[Pollux](https://arxiv.org/html/2008.12260v2)、[DynaTrain](https://arxiv.org/html/2605.18815v1#S4)、[OOCO](https://arxiv.org/html/2511.21862v1#S3)

#### LLM-driven online revision of system policy

**直接相关：Agentic online RL post-training。** 在本轮核验的公开文献中，尚未确认由 LLM
根据系统运行反馈修订性能模型或资源规划算法、部署后验证净收益的完整工作。直接系统已有动态重规划、在线校准和角色切换；AReaL 2.0 的演化控制面主要讨论 agent 权重与
harness。该检索结果的范围和纳入标准见
[SEARCH.md](SEARCH.md)，不预先构成新颖性结论。[AReaL 2.0](https://arxiv.org/html/2607.01120v2#S5)

**Level 1：其他 RL post-training。** LLMZero 用 LLM 诊断训练动态，在 checkpoint 搜索树上调整学习率、KL、batch
等训练超参数；其训练策略搜索与资源性能模型／规划算法修订的对象不同。ReSpec 的在线 drafter 训练与 SD 配置选择、Nereus
的效率校准，则提供其他反馈适应机制。[LLMZero](https://arxiv.org/html/2606.18388v2#S3)、[ReSpec](https://arxiv.org/html/2510.26475v1#S5)、[Nereus](https://arxiv.org/html/2609.34645v1#S4)

**Level 2：serving、数据库与算法发现。** Autopoiesis 根据近期 trace 的模拟评估持续演化 serving 调度代码；ADRD
共同改进数据库模拟／评估方法与策略；AlphaEvolve 展示了历史数据评估、留出验证与部署的链路。它们分别构成在线代码修订、联合改进和独立验证的先例。我们需要进一步检验，在
RL
的时效、学习质量及修订成本约束下，这些机制能否相对强校准／重规划基线提供额外收益。[Autopoiesis](https://arxiv.org/html/2604.07144v1#S5)、[ADRD](https://arxiv.org/html/2604.06566v1#S4)、[AlphaEvolve](https://arxiv.org/html/2506.13131v1#S3.SS3.SSS1)
