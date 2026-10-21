# Expert cost model 与异步 simulator 设计

状态：解析模型与异步模拟已实现；公司端已做同配置阶段校准，跨配置精度尚未验证。公司端报告及数据缺口见
[Tau2 计划](../../../../../examples/autopilot/PLAN.md)。expert 在每组实验中冻结，无 optimizer；
revision evaluator 提供反馈与验收判断，expert 不自动为 Basic 生成修订标签。

版本边界：目标是 AReaL v2，九条 Tau2 数据由用户确认来自 v1。当前 runtime 源码测试不验证 v1 兼容； 先由公司端按实际 commit
核对分配、packing 和计时范围，再决定可复用的阶段模型。v1 拟合不等于 v2 模拟已验证。

## 组件与 trial 档案

已将 r10 的 GPU／系统、模型与两侧 recipe 分开保存，由 `catalog.py` 校验并装配；`trial.py` 从归档输入重建条件模拟。
实测、校准和验证各自保存，完整数据入口见 [README](README.md#%E6%A1%A3%E6%A1%88%E4%B8%8E%E7%BB%84%E4%BB%B6)。

| 档案                  | 保存内容                                                                                 |
| --------------------- | ---------------------------------------------------------------------------------------- |
| GPU 硬件              | 型号／变体、容量、各精度峰值及 HBM 规格，逐字段记录来源和单位                            |
| 系统                  | GPU 数量与拓扑、CPU／内存、PCIe／NVLink／网络；实测有效速率及未测先验分开                |
| 模型                  | HF revision／配置 hash、架构、层数／维度、attention／MoE 结构；结构转换器生成工作量      |
| Actor／Rollout recipe | 后端及软件版本、精度、并行布局、packing／重计算／optimizer，或服务 batch／KV／cache 设置 |
| Trial                 | 上述档案引用、训练语义、输入／观测、日志来源 hash、校准／验证划分、结果与冻结版本        |

在 `profiles/{hardware,systems,models,actor,rollout}/` 保存具名、带版本的数据；在 `trials/<trial_id>/`
保存 manifest、数值观测、校准与验证摘要，较大证据通过相对引用关联。 装配器生成现有
`Hardware`／`ModelShape`／`ActorConfig`／`RolloutConfig`，保留公开 API 和既有数值基线。

校准单独绑定完整硬件／系统、模型、后端版本、配置、公式版本、计时边界和 workload 范围；不能把 r10 的阶段倍率、RPC tail 或清理先验写成 H20／Qwen3
的通用属性。原始观测、拟合参数和验证结果分开，跨 trial 拟合须显式声明数据集合与划分。 r10 的未使用跨节点带宽占位值也保留其原作用域，不能升级为系统实测速率。

| 计算组件             | 拆分边界                                                                                                                                  |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Actor                | 复用算子／通信成本；后端组件负责 dispatch／packing、sharding、collective、optimizer／offload 与执行依赖，组装 logp／advantage／train 阶段 |
| Rollout              | 服务组件估计一次 prefill／decode 迭代；后端策略决定 batch、KV／cache 和服务顺序，逐步提取当前模拟器中的相关规则                           |
| 客户端／环境         | RPC 响应交付、工具／评分／清理独立组合；SWE 环境开销不属于推理后端                                                                        |
| Transition／workflow | 权重发布／checkpoint 描述跨两侧依赖；共享事件模拟器负责 admission、排队、staleness、重叠与资源记账                                        |

`_actor/execution.py` 持有 mesh／层分配／pipeline 执行规则，原算子和通信组件继续复用；
`_rollout/{service,scheduling,types}.py` 分离迭代服务成本、SGLang 调度先验及请求状态。公开类型与入口保留。
新增硬件／模型先添加档案；出现新架构或执行规则时再增加对应代码组件，参数可装配与实测精度分别验收。

所有 trial 引用均固定内容 SHA256；一次装配只消费同一 manifest 下已校验的快照。模型维度从 HF 配置转换并核对。 校准加载只迁移已审核的
`(源码 hash, NumPy 版本)` 绑定，保留原系数／范围／划分，不在加载时拟合；未知组合仅允许未校准装配。

## 输入输出

以下为当前 API；模型实例持有校准与目标 runtime 版本，不扩展训练配置或 runtime。

```text
ExpertActorModel.estimate(config, workload, *, source_profile, observed_packing, auxiliary_stages) -> ActorEstimate
ExpertRolloutModel.estimate(config, service_state, ready_requests) -> ServiceEstimate
ExpertTransitionModel.estimate(action, source_key, target_key) -> TransitionEstimate
ExpertSimulator(frozen_models).simulate(scenario, supplied_plan) -> ExpertResult
replay(initial, records, *, stop_at, source_id) -> ExpertResult
```

| 输入             | 必要内容与口径                                                                                                                                                  |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 配置             | 模型结构／精度、硬件与拓扑、代码／backend 版本；actor 完整 mesh、microbatch、packing、重计算、PPO 更新次数；rollout 副本／TP、batch／token 限额、KV／cache 设置 |
| 训练语义         | groups/step、samples/group、trajectory 到 actor sequence 的展开、复用／过滤规则、staleness、group 并发上限、checkpoint／发布策略；首轮配置比较固定这些语义      |
| Actor workload   | 每个逻辑 batch 的序列数、attention tokens、长度分布、loss tokens；可选 trajectory 分组及按 step／phase／调用／mesh 位置关联的 microbatch token/padding 列表     |
| Rollout workload | group／episode／request 的对应关系、请求依赖、输入／输出长度、外部调用持续时间、接受／失败结果；只有 group 时间时保留该粒度，不伪造 episode 完成时间            |
| 初始状态与计划   | 时间原点、在途任务、队列、训练与服务版本、累计计数、已有 allocation／checkpoint；外部指定配置及切换触发点、结束与清理策略                                       |
| 校准             | 参数、拟合数据标识、适用配置／负载范围、测量口径与模型版本；原始数据和参数可保留公司端                                                                          |

阶段估计返回服务时间／依赖和资源需求；GPU 驻留费用由 simulator 统一计算。 每项估计附
`measured / calibrated / assumed / unsupported` 来源及缺失字段；资源可行性单独记录为
`feasible / infeasible / unknown`。缺测不等于不可行，缺失耗时不填 0；只有确认关闭的阶段才为 0。 ExpertResult
返回总时间、actor／rollout GPU-hours、队列等待、已完成工作量、版本记录及覆盖状态。

## Actor：阶段解析模型加有限校准

输出对齐实际计时边界：`recompute_logp`、`compute_advantage`、`train_step`。 ref／critic
等额外阶段若启用，必须另列；不能因摘要未提到就视为关闭。 借鉴已有 [Basic actor 模型](../basic/cost_model/README.md)
的字段口径，expert 独立持有公式和校准参数。

`cost_model/_actor/` 的初始原语从 `cfdbb716` 冻结后独立维护；不动态引用 Basic。现已加入阶段依赖图、Archon 层分配与 TP／CP
约束、PP 有向边界通信。仅支持非交错 1F1B，resolved config 中的虚拟／交错 PP 明确拒绝。训练 Adam 更新属于服务成本，不是配置搜索。

```text
t_op = max(F_op / P_effective, memory_bytes_op / HBM_effective) + launch_overhead
T_phase_prior = critical_path(算子、通信、同步与 pipeline 调度图)
T_phase = scale[full_profile, phase] × T_phase_prior
T_actor_service = Σ T_enabled_phase
```

FLOPs、bytes、FLOP/s、byte/s、秒分别使用一致单位。有效吞吐来自明确先验或校准，不能把峰值当实测。 通信节点按
collective、消息量、拓扑和可重叠依赖建模；critical path 计入 PP bubble、DP 不均衡， 不把所有 rank 时间相加。MoE
激活计算与完整权重驻留／通信分开，训练 optimizer step 纳入 actor 耗时。

- 先按阶段拟合少量正比例参数；仅有阶段总时间时，不同时辨识计算效率、网络带宽和重叠率。
- `PhaseCalibration.fit` 用正比例最小二乘；profile 包含目标 runtime 版本、完整模型／硬件／配置与公式摘要。特征边界包括
  token／序列数、长度、packing 工作量与证据类型；超界返回 unsupported，不自动外推。ObservedPacking 同时绑定 workload
  fingerprint 和来源。
- 同配置验证可使用实际 microbatch；更换配置必须从原始逻辑 workload 重新分发／packing。 原配置的 rank 列表不能直接搬到新
  mesh。缺少长度／分组时标注代理分布与敏感性范围。
- 按 DP／TP／CP／EP／PP 语义处理日志：合并重复记录，但保留真实分片工作及负载不均衡。 groups × samples、actor sequence 数和 PPO
  sequence uses 分别计数，loss tokens 不代替 attention 工作量。

### 纯 DP FSDP2／BF16 Adam 的主机阶段

FSDP profile 独立于 Archon：默认层级 wrapping、body/root reshard、BF16 梯度和 Kahan optimizer； 支持
CPUOffloadPolicy，未覆盖顶层模型 swapping 或自定义 wrapping。设全局参数数为 P、本地均匀 shard 为 p=P/DP、同步 train
microbatch 总数为 J、PPO optimizer 调用数为 U：

```text
logp H2D = 2p × logp_microbatches
train H2D = 4pJ; D2H = 2pJ
CPU gradient accumulation logical bytes = 6p(J−U)
CPU optimizer logical bytes = (64 + 4 × nonzero_weight_decay)pU
CPU norm/clip logical upper prior = 10pU
```

Checkpoint 重算增加 GPU compute，不额外增加第三次 all-gather。每个层／embedding／root group 分别计 collective
startup；训练 physical parameter tensor、AWEX staging view 和 wrap group 的数量不能互换。 resident
BF16 shard 每次 gather 另计一次本地 packing 读写（4p logical HBM bytes）。主机项以显式每 rank 速率串行相加，GPU
collective/copy 仍使用明确的 overlap 先验；总阶段拟合不能辨识单独 PCIe、DDR 或 overlap 效率。 `fsdp_components`
已纳入 `stages`；CPU/GPU working-set 下界与实测 peak 分开。

### Advantage 的计时修正

公司首轮反馈显示 Expert 首版的 advantage 预测约 `1e-4～1e-3s`，对秒级标签拟合出 `8106～14958` 倍率。
这是计时结构不完整的信号，不能解释为 GPU 效率。源码核查边界如下：

- [trainer](../../../../trainer/rl_trainer.py) 的 `compute_advantage` 包住 compute 调用及
  `get_device_stats().log()`； [计时器](../../../../utils/stats_tracker.py) 是 host
  `perf_counter`，不是 CUDA event。多 controller 的 rank-mean 汇总不能当最慢 worker 墙钟。
- [GAE](../../../../trainer/ppo/gae.py) 的 token 路径循环实际 padded width 减一；turn 路径在 CUDA 上循环
  padded width，CPU 上用实际 turn slots。 [batched_call](../../../../utils/data.py) 还做逐字段
  padding／concat／split；工作量是局部矩阵，不是训练 microbatch。
- 设备须实测确认：直接路径可在 rollout prepare 阶段已搬到 GPU；RPC 路径可在计时内做 localize、搬移／广播、返回 `.cpu()`
  和结果收集。 normalization 还有条件 group 循环、all-reduce、`.item()` 同步；这些选择必须进入 execution_id。

v2 显式采用以下可校准先验；各组件是互不重叠的开销，不能再加入已经包含这些工作的总阶段标签：

```text
worker[r] = batching[r] + recurrence[r] + normalization[r]
T_advantage_prior = transport + max_r(worker[r]) + diagnostics
component = Σ seconds_per_unit[feature] × feature
```

| 组件          | 可用输入／待测参数                                                                      |
| ------------- | --------------------------------------------------------------------------------------- |
| batching      | 每 DP 的 n_seqs × 实际 padded width、trajectory 数；拼接／拆分系数                      |
| recurrence    | GAE loop iterations、n_seqs × iterations；CPU 调度或小 kernel／host dispatch 的有效成本 |
| normalization | group 数、collective 数、padded slots；归一化／同步成本                                 |
| transport     | 边界内实际输入／输出传输字节及固定开销；需确认 controller、对象复制与缓存               |
| diagnostics   | device-stats RPC／日志等边界内固定开销；不能默认为零                                    |

组件参数未知时保留 None；默认不再用固定 40 次 launch 代替整个阶段。缺少实测形状时的 attended-length 代理明确标记， 它不能证明实际
padding／dispatch。校准绑定 profile、阶段、计时边界及上述特征范围，参数拟合不得使用验证或保留集。 仅有总阶段标签不能同时辨识多个组件系数；先采少量组件
spans，或把不辨识的部分保留为明确先验。

### 只有总阶段标签时的经验模型

公司端对 `69ebc25f` 复测仍缺五组件参数，因此显式提供另一路径 `AggregateAdvantageModel`：
`T = a + b × global_attention_tokens`，以非负最小二乘拟合整个 trainer 阶段。 采用可行的无约束解、`a=0` 和 `b=0`
边界最优解中误差最小者，不能简单截断负系数。 至少两个不同 token 值才可拟合 affine；固定 token 数据仅在显式选择常数模型后拟合均值。

模型只承诺相同执行 profile 和校准特征范围内的经验预测；n_seqs、平均长度及可用最大长度同时限制覆盖。 配置／runtime／execution
identity、计时边界和唯一 sample_id 必须确认；scope 不匹配、未知字段或超界继续拒绝。 总阶段拟合不拆分组件，也不叠加另一阶段倍率。执行 profile
在 fit 前后稳定，拟合产物完整进入冻结快照。 输入与用法见
[README](README.md#%E5%8F%AA%E6%9C%89%E9%98%B6%E6%AE%B5%E6%80%BB%E8%80%97%E6%97%B6%E7%9A%84%E6%97%A5%E5%BF%97)；真实误差及优于
token 基线的结论仍待公司端验证。

### 实测 packing 的归属和转换

公司端按 `run_id / source_revision / global_step / phase / call_index` 关联日志。phase 只接受已确认的
`train_step`／`recompute_logp`；train call_index 对应 PPO minibatch，logprob 是单次 forward 的 0。
每条 rank 记录附真实 global-rank→DP/TP/CP/PP 映射、按顺序的 microbatch 字段和 source_id。
模型／backend／版本由每条记录的完整 profile 绑定，source_revision 与目标 runtime_revision 必须相同，不能靠日志行号猜
step。

- 当前
  [Megatron](../../../../engine/megatron_engine.py)／[Archon](../../../../experimental/engine/archon_engine.py)
  日志是切分前 payload。保留各 DP 独立工作；同 DP 的 TP/CP/PP 复制必须一致才去重，EP 不额外乘除。
- `padded_to_tokens` 可能只是目标值；Megatron 逐序列 alignment 后实际长度可增加。 优先使用
  `aligned_tokens + padding_tokens` 或最终 tensor／cu_seqlens；padding 指尾部追加段，不包含逐序列
  alignment。
- 精确 attention 输入需要最终 cu_seqlens 差分（含尾部段）或同口径的长度平方和。 只有 real-token／padding 总数不足以推导
  attention 工作量；缺字段返回 coverage 问题，不能补零后称 measured。 提供平方和时检查尾部段下界、上界与整数
  parity；这些必要条件不替代最终 segments 的逐项核对。
- 精确转换要求完整调用覆盖、同步 microbatch 数和两阶段各自 token 守恒；部分观测允许缺 DP／phase，但不能生成完整阶段计数。混 step、重复物理
  rank 和已知冲突仍拒绝。 PPO 多次调用划分同一批工作，不重复乘 PPO 次数。当前转换器只支持明确的 pre-shard THD 口径。

部分日志与精确 `PackingPlan` 分开保存。`PackingTimingDiagnostic` 仅用校准标签拟合 `T=a+b*nmb`，独立比较实测／预测
count 输入同一冻结经验模型的误差；它不能替代原解析模型的条件验证。 接口、范围及公司端复测口径见
[Packing 诊断](README.md#packing-%E8%AF%8A%E6%96%AD)。

## Rollout：服务模型与外部等待分开

```text
Δt_service = f_theta(layout, prefill_chunks, decode_contexts, KV_state, cache_mode)
request_ready = max(前置请求完成, 前置外部调用完成, admission_time)
request_delivery = last_token_time + response_tail
sample_done = agent_return + sample_cleanup
group_internal_done = max(该 group 必需成员的 sample_done) + group_postprocess
group_ready = delivery_after_pause(group_internal_done)
```

`f_theta` 组合推理算子／显存／通信开销与匹配 profile 的尺度校准；以副本的一次服务迭代为单位， 支持 prefill／decode 混合，完成后更新各请求
token 进度及 KV 状态。排队时间由 simulator 产生。 `sglang:d8t4` 对应的副本服务与 group admission 分开；group 并发上限
128 不代表 128 个推理请求同时执行。

可选 `RPCTimingModel` 将 RPC 标签的有效 service 倍率与固定响应 tail 分开；前者每 iteration 使用一次，后者在释放 GPU
请求／KV 槽后异步等待一次，依赖仍等到交付。取消清除 pending response，恢复只等待剩余时间； publication 不等待 response
tail，交付保留最后 token 的版本。初始快照缺该版本时明确记为未知。 完整底层模型（含 GPU calibration）、配置、来源及请求长度边界进入绑定；RPC
标签无法辨识物理组件，两项始终为 `assumed`。原始 RPC 总时长不能再叠加到已有 GPU service 上。未提供模型保留零 tail 先验。

外部工具／用户模型调用以依赖节点表达，明确其并发限制；重用历史持续时间属于外生延迟假设。 固定请求长度和依赖只能评估给定 workload
下的系统成本，不预测策略变化后的轨迹内容或 reward。 失败、过滤和重试须按原运行语义处理，不能丢掉已消耗的时间／资源。

每条 sample 链在 agent／评分返回后接独立的 cleanup `ExternalCall`；四个成员可并行清理，等待中不占 GPU 请求／KV
槽。`Group.postprocess` 只表达全部成员结束后的导出等整组开销，不再重复加一次清理。r10 的 30s 仅作显式先验；物理清理完成、内部 accepted
与暂停恢复后的结果可读分别计时。

`Scenario.request_routing="group_affinity"` 在组启动时按最少 active groups 分配副本，平局轮换；各
sample／轮次固定该副本，槽位／KV 不足时在原副本等待，工具与清理期间保留归属，内部完成后释放。该策略是明确的模拟假设，不声称复原 runtime 路由；默认
`request_load` 保留逐请求负载分配。恢复中的组必须提供副本，或从其一致的在途请求映射推导；缺失／冲突拒绝。

当前只有一种 rollout 布局且缺少纯推理／外部等待分解：

- **原配置回放**：用可确认的 group 接收事件验证队列消费和成本记账；截断区间保留未知在途状态。 回放不验证 rollout 服务模型，也不能通过改变 actor
  速度就宣称闭环收益。
- **配置预测**：需要请求依赖及匹配的服务测量／明确假设，重新产生完成与接收事件。 缺少分解时不输出伪装为已校准的联合配置结果；`timeperf/rollout`
  不用于拟合纯推理耗时。

## Transition：按源与目标配置计费

```text
T_transfer_prior = setup_latency + transferred_bytes / effective_bandwidth
T_transition = critical_path(checkpoint, release, allocate, restore/reshard, publish)
```

上述节点按实际 action 选择，并非每次都执行全部节点。输入包括源／目标 layout、checkpoint 状态、 数据量、存储／网络路径、临时 GPU
占用和阻塞范围。区分持久 checkpoint 与权重发布； 异步 staging 与传输依赖分别计时，等待服务不重复收费。只有 schedule 要求时才保存
checkpoint。 现有 60s checkpoint／120s restore 只可作为显式假设，不是默认校准值；换布局还需 reshard 测量。

CPU offload 的 AWEX 发布由 `fsdp_publish_tasks` 显式串联 fresh shard staging、同一版本内去重的 contiguous
copies 和实际 peer-plan 传输；跨版本不缓存旧 CPU 权重副本。全量 `update_weights` 标签仅可校准整个发布边界，替换整个 DAG，
不能同时用作其 network 子组件。缺少 cleanup／有效速率／启动开销保持 unknown。

`ExpertPublicationModel` 将完整发布分成控制、逐 rank 准备、网络轮次和接收回填。 准备保留所有 rank 的工作；共享 host 带宽按活跃
staging rank 固定均分。 每轮网络按 sender／receiver 及跨节点共享 NIC 的最大负载估计，各轮串行；发送／接收投递开销与控制成本另计。这是明确的
bulk-synchronous 先验，不复原 NCCL stream 顺序或 collective tree。FSDP／Megatron 使用 dense AWEX
peer-plan 流量；Archon 使用旧式 XCCL bucket 的显式逻辑广播流量与 ACK。后者不代表存在 Archon v2 adapter。 输入与
Actor／Rollout／模型／硬件／runtime 绑定；几何覆盖在导出端校验，模型再查 peer、完整性标记与接收字节量。
模型源码、工作量、速率进入校准身份；整段校准仅改变总耗时，原组件速率不可据此解释为已辨识。

## Simulator：离散事件推进

维护 group admission、各推理副本的服务队列、外部等待、已接受 group 队列、actor 阶段、 训练／服务版本、checkpoint 和 GPU
allocation。配置由 supplied_plan 给定，不在内部搜索。

`Scenario.workflow="v2_inline"` 另区分主线程 batch 入队／1 秒等待、后台按 staleness 提交和 pause 后的结果交付。
resume 使用零线程交接延迟，末步允许已有 pending 一次提交再进入 terminal；这是协议级先验，不预测 OS race 或固定尾部预取数量。

1. 从显式初始状态开始；按时间、事件优先级和稳定 ID 处理事件，同一时间先完成／释放再派发。支持边界见
   [README](README.md#%E8%8C%83%E5%9B%B4)，不接受任意截断状态。
1. admission 后逐次推进请求服务与外部调用；仅当 group 达到既定完成／过滤条件时入队。
1. 按 dispatcher 顺序收集完成结果，再过滤拒绝项；固定 batch 补足 accepted，dynamic batch 按 attempts 结束。全拒绝
   batch 明确 unsupported。actor 可用时展开／packing 并推进阶段。
1. 按实际完成事件更新训练版本、发布状态及服务版本，重新计算 admission 容量与队列状态。
1. 切换按安全边界执行；在途请求是 drain、保留还是取消必须明确。完成相同工作目标后按声明策略清理。

若公司运行版本与本地 [StalenessManager](../../../../infra/staleness_manager.py) 一致，admission 容量为：

```text
capacity = max(0, min(K - running, (s + v + 1) × B - accepted_total - running))
```

`K` 是并发上限，`s` 是 staleness，`B` 是 consumer batch，`v` 严格对应实现中的 version provider； accepted
是累计接受量。公司端需确认计数单位、恢复偏移和拒绝处理，不能直接令 B = groups × samples， 也不能把 admission 约束替换成简单的完成时丢弃规则。

```text
GPU_hours = Σ_interval |unique_allocated_GPUs(interval)| × Δt_interval / 3600
```

驻留期间的训练、推理、等待和切换均计费；已释放资源停止计费。源／目标临时共存按真实 allocation 计， 同一张卡不会因重叠阶段重复计费。首轮为
actor／rollout 分卡；总时间来自事件推进，不是各阶段耗时之和。 结束时在途任务和资源不能免费消失，回放缺尾部数据则单独报告观测窗口。

## 实现与公司端验收顺序

1. 使用已实现的 actor 分阶段预测；公司端关联 microbatch 与真实阶段标签，按 run／连续时段分组校准与留出验证。 同一 step 的
   rank、phase、trace 派生记录必须属于同一划分。
1. 使用原配置 replay 检查可确认的消费归属、版本与 GPU-hours；聚合记录不验证精确归属或 staleness 合规。与预测精度分开验收。
1. 补推理／外部等待分解、同 workload 的跨配置测量及 transition 测量，再校准配置预测路径。
1. 按 backend／布局报告各阶段 WAPE（Σ绝对误差／Σ实测时间）、尾部误差和覆盖率；全零阶段单独检查。 配置排序只在相同
   workload／训练语义的配对测量上验证，不按九个 trial 的总耗时排序。
1. 保留最终独立实测集，达到声明精度后冻结 expert。Basic 各轮版本与其同口径比较，revision 由 evaluator 指导。

继续使用公司端既定划分：584 校准、186 验证、9 间隔、200 最终保留，另 9 个启动 step 仅做接口检查。 对原验证集同时报告 Expert 与 token
线性基线的阶段／actor-total WAPE、P90、各自覆盖率和共同 covered 子集结果； 不丢弃拒绝样本来提高指标，不放宽 bounds 隐藏缺测。T0390
验证仅 2 步、S030 原覆盖 4/8，须同时展示样本数。 本机只改接口和模型结构；新误差与排序由公司端复测返回。

Rollout 最小补测：请求 ID／group／依赖、prefill token／prefix hit、decode 长度、服务迭代时间、排队和外部调用分界、
副本／TP、batch/KV 状态与实际版本。Transition 最小补测：有向 source/target profile、action、开始结束及依赖、
传输字节／路径、阻塞范围、allocation 增减及临时重叠。缺字段继续 unsupported，60s/120s 不升级为测量值。

本轮不增加 benchmark、Tau2 入口、合成场景或测试文件；仅在已有测试文件中覆盖模型边界与事件语义。CPU 测试验证实现，不能替代上述公司端测量验收。
