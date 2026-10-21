# AutoPilot related work：调研结论

截至 2026-10-21。按讨论后的范围，采用**两条主线，每条分直接相关、Level 1、Level 2**。保留并重组原 29 篇来源，新增核验 7
篇；重点结论依据原始设计与评测，未复现实验。

| 主线                     | 直接相关：Agentic online RL post-training                                               | Level 1：其他 RL post-training                                | Level 2：其他系统                                     |
| ------------------------ | --------------------------------------------------------------------------------------- | ------------------------------------------------------------- | ----------------------------------------------------- |
| 资源与配置调整           | Libra、DynaRL、RLinf、QwenGyre、PEARL、TideRL、SeamlessFlow、ROSE、ReFlow；另列局部问题 | Nereus、RLHFless、RLBoost、ReSpec、ReaL、HybridFlow、StreamRL | Pollux、DynaTrain、OOCO、Vidur／vTrain                |
| LLM 驱动系统策略在线修订 | 本轮未确认符合完整机制与实证标准的工作；列明直接系统已有的校准／重规划能力              | LLMZero；ReSpec、Nereus 作为不同适应机制的比较                | Autopoiesis、ADRD、AlphaEvolve、ADRS、AdaEvolve、EvoX |

## 主要发现

- **重配置已有强直接工作。** Libra 研究联合布局与分配；DynaRL、RLinf 也有多轮工具任务评测，不能仅列为泛 RL
  背景。QwenGyre、PEARL、ReFlow 分别覆盖角色切换、rollout PD 和资源重平衡。
- **修订问题存在可研究空间，但尚不能断言新颖性。** 本轮未确认在目标场景中由 LLM 修订性能模型／规划算法、验收后投入运行并证明净收益的完整工作；这是有范围的检索结论。
- **新增的 LLMZero 是关键相邻工作。** 它根据训练反馈搜索多阶段超参数；Autopoiesis、ADRD
  则分别提供在线调度代码演化与模型／策略共同改进的跨领域先例。
- **比较需注明语义和资源来源。** TideRL 可改 batch，ReFlow 引入样本复用，ROSE 借用 serving
  容量；它们的总收益不能直接解释为同一训练语义下的联合配置收益。

online 分“训练内生成新轨迹”和“部署后持续接收真实交互”两种证据；直接层的 agentic benchmark 主要验证前者。policy
在本项目指资源管理策略，包含性能模型与规划算法；agent 权重、harness 和训练超参数另行标注。

## 阅读入口

- [Proposal](PROPOSAL.md)：采用已讨论的两段 motivation，并按两主线／三层呈现结论。
- [证据矩阵](EVIDENCE.md)：每篇的调整对象、反馈、执行方式、评测和适用边界。
- [检索范围与纳入判断](SEARCH.md) · [24 条检索记录](search-log.json)。
- [基线与待验证问题](EVALUATION.md) · [BibTeX](references.bib) · [来源及逐篇分层](sources.json)。

后续优先比较同 workload 下的固定配置、普通校准／重规划和修订方法；动态配置净收益与 LLM 修订增益仍待独立实验。
