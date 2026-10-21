# Basic cost model 与配置搜索

`BasicPolicy.decide()` 处理 Actor 生命周期；`JointPolicy.plan()` 离线比较异步联合配置。 `online.py` 的
`OnlineJointPolicy` 保留严格串行的联合模型与决策，供离线参考；均独立于 Expert。

## 联合配置

依赖方向：`JointPolicy → JointOptimizer → JointSimulator → ActorCatalog / RolloutCostModel / JointTransitionModel`。
联合数据协议在 `joint.py`，新类型均可从 `policy.basic` 导入；`ActorCatalog` 复用下文的 Actor 模型。

```python
from areal.elastic.policy.basic import JointConfig, JointOptimizer, JointPolicy, JointSimulator

# catalog、rollout_model、transition_model 和 scenario 由调用者提供。
candidates = tuple(JointConfig(actor, rollout) for actor in catalog.configurations for rollout in rollout_configs)
policy = JointPolicy(
    JointSimulator(catalog, rollout_model, transition_model),
    JointOptimizer(candidates, max_evaluations=128, max_seconds=None, switch_margin=0.05),
)
result = policy.plan(scenario)
```

| 输入                | 约束                                                                                                                                |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `JointConfig`       | Actor 完整 profile；Rollout 的模型、硬件、运行版本、采样语义、replicas/TP、batch/KV 容量与批次并发。额外引擎选项纳入 `runtime_id`。 |
| `JointBatch`        | 固定逻辑 groups、显式 samples、Actor 统计和独立推理负载；不把 actor 的历史 batch 等待当成纯推理时间。                               |
| `RolloutWorkload`   | 各阶段的 prefill/decode token、外部等待秒数与可并行请求数；总请求数、最大请求 KV token。未知阶段并行度默认为 1。                    |
| `RolloutProfile`    | 精确配置下每个 replica 的聚合 prefill/decode token/s、负载覆盖范围、来源及 assumed/measured 标记。                                  |
| `TransitionProfile` | 有向 start/reconfigure/checkpoint/publish/shutdown；每阶段秒数与各资源池的**总驻留卡数**，包括暂时重叠的分配。零成本也需显式给出。  |
| `JointScenario`     | 同一有限窗口、当前配置、资源上限、staleness、checkpoint 周期；初态须已提交且无进行中的 rollout/训练/发布。                          |

Actor 同一 catalog 保持训练语义；有 source profile 的 rank/microbatch 观测只复用于该完整配置，
其他候选从原始顺序的逻辑序列重新分发。缺少精确长度时，先固定一份共享替代分布，并在结果标注。 初始 ready 批次不重复计算已经发生的
Rollout；累计计数使用同一批次起点，
`accepted_batches = provider_version + ready_batches`。`checkpoint_every=0` 显式关闭周期
checkpoint。

模型用一个聚合推理服务通道，阶段耗时为
`(prefill_tokens / prefill_rate + decode_tokens / decode_rate) / min(replicas, parallel_requests)`。
假设阶段内 token 负载均衡；单个请求不会因增加副本而加速。 外部等待释放通道，Actor 消费就绪批次并与 Rollout
重叠；发布等待当前推理阶段结束、暂停新推理，外部等待仍继续。 checkpoint 在匹配的发布版本后执行。staleness 与 `max_inflight_batches`
是**整批次代理**，不能直接填公司配置中的 group 并发上限 128。 TP／batch／运行版本变化需要对应 profile，不按卡数外推速度。GPU seconds
按各阶段驻留卡数积分，只计一次。

搜索对每个候选固定整个窗口的配置，最小化 Actor＋Rollout 总 GPU seconds，包含切换与末尾释放； `max_seconds`
限制完整作业时间。先评估当前配置，预算只减少候选数，不缩短工作窗口；`unevaluated/exhaustive`
报告覆盖程度。支持约束内的当前配置享有切换收益门槛；无合格候选时返回 `config=None`，不会退回忽略 Rollout 的决策。

结果保留耗时、成本、完成工作量、事件、输入来源、假设和逐候选拒绝原因。 这是固定轨迹结果的离线近似，未实现 request/group 级调度、进行中任务迁移或 reward
预测。静态 Actor 内存下界和单请求 KV 检查不保证整体可执行。离线联合 optimizer 已接 revision evaluator。原生执行仅接固定布局 Basic
生命周期策略；
[接口和限制](../../../../examples/autopilot/elastic/README.md#behavior-and-boundaries)。九个
Tau2 trial 的跨配置精度、真实转换成本与收益仍需公司端验证。

## 已有 Actor 生命周期策略

依赖方向：`LifecycleOptimizer → WorkflowSimulator → ActorCostModel`。

- `cost_model/`：`ActorConfig + WorkloadStats → recompute_logp / compute_advantage / train_step`。
  [输入、组件与精度限制](cost_model/README.md)；import 与目录 CLI 均使用本目录下的实现。
- `actor_cost.py`：将三个阶段接入 simulator 的 train 组件。`ActorCatalog` 保留完整配置身份， 同布局 trial
  作为别名，不能按卡数合并。重新校准或替换组件后应新建 catalog，或调用 `clear_cache()`。
- `simulator.py`：组合 actor、rollout、启动／恢复、checkpoint、发布和释放耗时；统一计入驻留 GPU 成本。
  生命周期耗时由调用者提供组件；没有数据时只使用明确声明的情景假设。
- `LifecycleOptimizer`：在显式候选配置中做有限视野动态规划，联合选择配置和等待期间是否释放。 目标为固定样本量下最小 GPU seconds，可选
  training／total 范围和墙钟上限。 保留 cost/time Pareto 前沿、切换收益门槛及评估预算；每次只执行第一个动作。

已知 queue head 使用实际统计；未来使用最近一份含 actor 统计的已观测 batch 及历史 rollout 时长。 各候选处理相同样本和有效
token，不通过改变 PPO minibatch 数、loss/logprob 语义减少工作。 `samples` 必须显式提供，不能用多轮展开后的 `n_seqs`
代替。缺少 actor 统计时拒绝预测。 checkpoint 来源与目标必须相同或出现在显式的有向兼容表中；不会默认跨后端恢复。
所有完整候选方案都包含末尾释放；`max_seconds` 约束的是整个视野。 `restore` 时长需包含重建训练组和恢复／重分片；声明兼容性不能代替相应执行实现。

在 AReaL 根目录离线运行，无需训练依赖：

```bash
python -S areal/elastic/policy/basic \
  examples/autopilot/elastic/lifecycle_scenario.json --output /tmp/decision.json
python -S examples/autopilot/elastic/validation/lifecycle_policy.py
```

JSON 可用 `trials + actor_overrides`，也可用 `candidates: {label: resolved_actor_config}`。
`workload.stats` 接受 `WorkloadStats` 字段；生命周期秒数、常驻推理卡数和兼容表均为显式输入。 程序接口为
`build_actor_policy(catalog, LifecycleCosts(...), ...)`，返回离线生命周期 `BasicPolicy`。原生 host
通过 `NativePolicyRuntime` 校验执行身份，当前仅允许固定布局。

[离线验证](../../../../examples/autopilot/elastic/validation/lifecycle-policy-20261018.json)
覆盖冷启动、长／短空闲、同卡数切换、兼容性拒绝与时间上限。该证据属于离线模型。 原生执行不支持 named profile 或布局变化；GPU checkpoint
跨布局转换、H200/Archon 精度、reward 保持与实际收益待验证。
