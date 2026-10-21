# Expert-designed reference model

用于比较 Basic simulator 各轮 revision 的固定参考。已实现独立解析模型、限定范围的阶段校准、异步模拟和观测回放；目标精度尚待公司端验证。

[模型设计与公司端校准](DESIGN.md) 列出公式、输入口径和支持边界。原配置回放与新配置预测分别验收。 实现／验证进度、AReaL 2.0 workflow
时间线及[绘图标准](PROGRESS.md#%E6%97%B6%E9%97%B4%E7%BA%BF%E7%BB%98%E5%9B%BE%E6%A0%87%E5%87%86)统一维护在
[Expert 进度](PROGRESS.md)。

| 文件                                 | 职责                                                    |
| ------------------------------------ | ------------------------------------------------------- |
| `cost_model/actor.py`                | actor 各阶段耗时与配置／workload 的关系                 |
| `cost_model/advantage.py`            | 显式执行设备／计时边界下的 advantage 工作量与组件参数   |
| `cost_model/aggregate_advantage.py`  | 只有阶段总耗时标签时的有界经验模型                      |
| `cost_model/fsdp.py`                 | FSDP CPU offload 工作量、主机速率与 AWEX 发布依赖       |
| `cost_model/packing_observations.py` | 部分 rank 观测、packing 误差及精确转换                  |
| `cost_model/packing_diagnostics.py`  | 给定实测 MB 数的独立经验耗时诊断                        |
| `cost_model/rollout.py`              | rollout 推理性能；区分外部环境／工具耗时                |
| `cost_model/rpc.py`                  | RPC 标签下的有效 service 倍率与异步响应延迟             |
| `cost_model/transition.py`           | 启停、checkpoint／恢复、权重同步及配置切换              |
| `simulator.py`                       | 组合阶段模型，推进工作流／队列状态并计算时间和 GPU 成本 |
| `scenario.py`                        | 请求依赖、初始状态、外部指定配置计划和结果              |
| `replay.py`                          | 原配置观测窗口的计数、版本与 allocation 记账            |
| `core.py`                            | 来源／可行性、校准范围、关键路径和版本摘要              |
| `cost_model/_actor/`                 | 独立持有的算子、通信、dispatch／packing 原语            |

本目录不含 optimizer 或 `policy.py`，不实现 `SchedulingPolicy`。
预测公式与校准参数独立持有，每组对比实验冻结版本；高精度需由目标数据验证。

## 档案与组件

历史 R10
数值档案与图片已移出仓库，见[本地归档](../../../../../artifacts/repository-archive-20261021/README.md)。基础
profiles 和所有模型源码仍随包分发；`Catalog()` 可直接读取基础 profile，历史 trial 需显式传外部 `root`。下表的 `trials/`
指外部 catalog 中的路径。

| 目录／入口                                       | 内容                                                                          |
| ------------------------------------------------ | ----------------------------------------------------------------------------- |
| `profiles/hardware/`、`systems/`                 | GPU 规格与机器拓扑／主机速率分开；来源、单位、实测／先验属性保留              |
| `profiles/models/`                               | HF revision、配置、结构参数；当前转换器覆盖 Qwen3 MoE，核对配置与维度一致     |
| `profiles/actor/`、`rollout/`                    | 软件／执行版本、并行与精度、算子效率和后端策略                                |
| `trials/r10/manifest.json`                       | 固定上述 profile 和各份数据的 SHA256，关联训练语义、源码／数值环境            |
| `trials/r10/workload.json`、`workload/`          | 固定请求依赖、长度／结果及外部耗时；依赖图以 JSONL.gz 保存                    |
| `trials/r10/observations/`                       | group、sample、RPC、工具、Actor、实际 batch 和控制器时间线的数值观测          |
| `trials/r10/calibration.json`、`validation.json` | 原校准系数／范围／数据划分与已有验证结果分开，原始来源 hash 在 `sources.json` |
| `catalog.py`、`trial.py`                         | 校验、装配、重建模拟；不启动训练或下载模型                                    |

```python
from areal.elastic.policy.reference import Catalog

catalog = Catalog(root="/path/to/external/expert_designed")
trial = catalog.trial("r10")
trial.verify()
assembled = trial.models()  # 原校准系数，严格覆盖边界
models, scenario, plan = trial.simulation(
    cleanup_seconds=30, diagnostic_fallback=True
)  # 显式复现原条件对照；默认不开启越界 fallback
```

新硬件／机器、同类模型或后端 recipe 通过新增 profile 文件接入；已被 trial 引用的文件保留原版，新版本使用新名称。 新 trial 的 manifest
引用选定 profile 内容，另存自己的观测、workload 和校准／验证；同名硬件不自动共享校准。 `Catalog(root=...)` 可使用外部目录；只有
GPU／模型时可分别调用 `hardware(name, system)`／`model(name)`。 修改公式或 NumPy 版本后，重新审核校准兼容组合；未审核时
`trial.models(calibrated=False)` 仍可生成解析先验。

Actor 的阶段装配复用 `_actor/` 的算子、通信、packing 和执行规则；Rollout 的服务成本与 batch／KV 策略在 `_rollout/` 分开。
RPC 交付、外部环境与 transition 保留独立边界，排队／staleness／重叠由共享 simulator 推进。档案化不会扩展后端覆盖或提高既有结果的验证等级。

## API

模型使用中性 `WorkloadStats`，不调用 Basic 或训练引擎。FFD 使用项目现有 NumPy，匹配 runtime 的
`argsort`（重复长度并非稳定排序）；本地 NumPy 版本进入 profile。公司端应核对实际运行的 NumPy／源码版本，跨版本或平台不保证 ties 次序相同。
`runtime_revision` 必须标识目标 AReaL／backend 版本；模型结构、硬件峰值与拓扑由调用方提供；有效速率是显式先验或校准值。

目标系统是 AReaL v2；用户确认九条 Tau2 历史来自 v1。当前源码对齐测试只覆盖本仓库 v2 路径，未完成 v1 兼容核对。 `runtime_revision`
是身份绑定，不会自动选择 v1/v2 实现；`expert-actor-v2` 是模型自身版本。v1 阶段拟合或给定实测 packing 的诊断不能直接证明 v2
的分配、异步时间线或新配置精度。公司端须先核对实际 v1 commit、dispatch／packing 与计时边界；同一底层规则确认一致后才可复用，参数按运行版本分开。

```python
actor = ExpertActorModel(shape, hardware, runtime_revision=runtime_revision,
                         advantage_model=advantage_model)
prediction = actor.estimate(actor_config, workload, advantage_workload=advantage_workload)
rollout = ExpertRolloutModel(shape, hardware, rates, runtime_revision)
transitions = ExpertTransitionModel(profiles)
models = ExpertModels(actor, rollout, transitions)
result = ExpertSimulator(models).simulate(scenario, supplied_plan)
```

上述类从本包导入；输入字段见对应 dataclass。`ActorConfig.from_areal(resolved_config)`
读取支持的配置，未支持的执行选项明确拒绝。组件模式的 `advantage_model` 由公司端提供
`AdvantageModel(execution_id, device, controller_mode, gae_mode, terms)`： execution_id
应标识归一化／GAE lambda／KL 配置、张量字段与 dtype、传输／缓存协议。 五类 `AdvantageTerm` 为
batching、recurrence、normalization、transport、diagnostics； coefficients 是命名工作单位的秒数，未知保留
None，零值也需显式来源。仓库不提供目标机器的默认系数。

`AdvantageWorkload.for_workload(...)` 绑定实际 DP 张量形状、trajectory／循环／归一化计数、传输字节及来源。
缺观测时只可使用明确标注的长度代理；缺模型、系数或非零项所需输入则阶段和总耗时返回 `unsupported`， Actor 的其他可预测阶段仍保留。此行为修正 v1 的
HBM-only advantage 数字。 `PlanStep.advantage_workload` 可向异步模拟传递同一工作量的观测；换配置不能复用。

实测 packing 优先用 `actor.observe_packing(config, workload, records)`，自动检查目标 profile 和
runtime。 每条 `PackingRecord` 带 run／source_revision／profile／step／phase／调用序号、global rank
和明确 DP/TP/CP/PP 映射。 公司端确认原运行配置后绑定 `actor.profile_key(config)`；`source_revision` 必须与模型的
`runtime_revision` 相同。 `PackingMicrobatch` 保留原始缺失字段；只有实际 compute、tail padding、attention
平方和可确认时才生成精确 plan。 转换器检查完整 DP／PPO 调用覆盖、并行副本一致性、padding／attention 一致性与 token 守恒。手工 plan 的
`ObservedPacking.for_workload(...)` 保留；两者都仅限同一 profile 和 workload。

解析阶段用 `CalibrationPoint` 保存原始阶段预测、实测秒数、`prediction.features_for(phase)` 和
`timing_boundary=prediction.timing_boundary`，以 `PhaseCalibration.fit(version, points)`
拟合正比例尺度；通过 `dataclasses.replace(model, calibration=(...))`
生成新冻结模型。校准和留出验证数据分开；超出拟合配置／特征边界返回 `unsupported`，并列出具体越界特征；不同 profile 没有校准时返回 `assumed`。
Actor 标签边界是 `trainer_host_wall_with_stats`：worker-only、CUDA event 或多 controller 的
rank-mean 标签不能混入。 `advantage_components` 仅展示未校准的组件先验；最终阶段时长读取 `stages`，总阶段倍率不等于独立组件测量。
代码／profile 变更后旧校准不可直接复用；按原训练划分重新生成预测和拟合，最后 200 步仍保留。

### FSDP2 与 CPU offload

支持 `fsdp:dN` 纯 DP、BF16 参数／梯度与 `adam_bf16` Kahan optimizer；默认层级 wrapping、body/root
`reshard_after_forward=True`。自定义 wrapping、混合并行、per-layer optimizer 和顶层模型 swapping 继续拒绝。
`ActorConfig.from_areal(...)` 读取 `actor.fsdp.offload_params`，不将它混同于 `enable_offload`。

`ExpertActorModel(..., fsdp_host_rates=FSDPHostRates(...))` 显式接收每个并发训练 rank 的 H2D／D2H
GB/s、CPU 逻辑流量 GB/s、每 tensor 传输启动秒数和每次 optimizer 固定秒数。缺活动项保持 unknown；零启动成本也须显式给出。
非零传输启动成本还需 `ActorConfig.fsdp_parameter_tensors` 实际数量，不能拿 wrap group 数替代。 主机速率进入
profile，旧 H200／其他 backend 校准不可复用。

`ActorEstimate.fsdp_work` 保留 gather／reduce、主机传输、CPU 梯度累积和 BF16 Adam/Kahan 流量；
`fsdp_components` 是未校准主机先验，已计入最终 `stages`，不可再相加。CPU offload 下 GPU persistent 参数下界为零， 不代表
GPU footprint 为零；`gpu_working_set_lower_bounds_gb` 另列最大参数组／BF16 logits 下界， 仍不预测
activation、prefetch 或 temporary peak；CPU state 下界在 `host_memory_gb`。逻辑 eager 流量不是实测 DDR
带宽。

`fsdp_publish_tasks(...)` 生成 cleanup → fresh CPU shard staging → contiguous copies →
AWEX network 串行依赖。调用方提供实际 staging tensor 数、去重后 copy 数／字节、明确的 copy 启动成本及完整 peer-plan
网络估计。 整段 `update_weights` 校准替换整个 DAG，不能加在 staging 后面重复收费。

`ExpertPublicationModel` 进一步按实际 rank 工作量和 peer 流量预测整个 `update_weights`：pause／cleanup → 各
rank staging／gather／转换／打包的关键路径 → 网络轮次 → 接收回填 → barrier／resume／版本确认。 共享 host 带宽和跨节点 NIC
容量显式计入；速率／控制开销仍为先验。 网络按资源瓶颈近似，非精确 NCCL 调度。 `PublicationWork` 绑定
Actor／Rollout／模型／硬件／runtime；完整流量由调用方校验并导出，包含副本复制，不能只给参数总量。 缺 counter／速率／覆盖返回 unknown。支持
FSDP／Megatron dense AWEX；Archon 仅旧式 XCCL 的显式 bucket 广播近似，需每 bucket ACK，当前 v2 无 Archon
AWEX adapter。delta／disk／colocate 未支持。 `PublicationEstimate.transition(source, target)`
接现有 simulator；整段校准替换全部组件，保留 raw 组件用于诊断。
[R10 配置与入口](../../../../examples/autopilot/elastic/README.md#structural-weight-publication)。

### RPC 与 GPU service 的计时边界

`ExpertRolloutModel` 预测一次 GPU service iteration。只有 client send→response 标签时，可另给
`ExpertModels(..., rpc_timing=RPCTimingModel(...))`：`service_scale` 每个共享 iteration 乘一次，
`response_seconds` 每个完整 RPC 加一次。最后 token 生成即释放 KV／请求槽；响应交付后才完成 Request
依赖。异步延迟不阻塞其他请求或权重发布，GPU 驻留费用仍按 allocation 计。

RPC 模型以 `RPCTimingModel.profile_for(rollout_model, rollout_config)` 绑定完整底层模型和 GPU
calibration；保留版本、唯一 `sample_ids` 与 `input_tokens`／`output_tokens` 上下界。配置、模型或校准变化需重新拟合。
默认拒绝请求长度超界；显式 `allow_request_extrapolation=True` 仍标注越界。 `rpc_timing=None`
保留零响应延迟先验。`RunningGroup.response_remaining` 可恢复已结束 GPU 服务的在途响应。

两项参数应在原校准段联合拟合，只用冻结参数评价留出段；本包不搜索参数。RPC 标签不能辨识真实 GPU 效率或开销发生位置，因此结果始终 `assumed`，不能作为 GPU
`PhaseCalibration`。长度覆盖不代表 cache／batch／KV 覆盖，实测工具间隙不能再包含该 RPC
的响应区间。已有留出集用于开发后，最终精度需独立数据验收。

### DP 分配与 packing

在模型中显式指定 `dispatch=DispatchConfig("controller", group_size=1)` 或
`DispatchConfig("redistribute")`，并核对目标运行源码。默认 `proxy` 保留近似预测，`exact_dispatch=False`。
选择进入 profile；改变选择后按原训练划分重新校准。

- `controller` 对原子 trajectory／连续 group 的 dispatch weights 做等条目数 balanced-greedy。
  精确输入需要原始 trajectory 顺序、`trajectory_group_sizes`、逐行 `sequence_lengths` 和真实
  `trajectory_dispatch_weights`。Tensor 通常用 attention sum；RTensor 用 attention-mask 矩形
  numel，不能用全局平均 token 替代。 `group_size` 是 actor controller 调用的分组参数，不能从 rollout 的
  samples/group 推断。
- `redistribute` 对 gather 后的 trajectory attention sums 做 FFD，保留原始 bin／插入顺序。输入顺序为源 DP
  rank，再各 rank 本地顺序； 当前 runtime 要求 gather 前各 list 等长，因此总 trajectory 数须整除 DP，分配后各 rank
  的数量可不同。此模式不接受不同口径的 controller weights。
- 没有 trajectory 分组、路径或权重时继续明确标注代理；缺失逐序列信息不补成实测值。给定 `per_dp_sequence_lengths` 是条件输入，仍绑定
  source profile，不能跨配置搬用。

`prediction.packing.rank_work` 按 phase／PPO call／DP 输出 sequence 数、real tokens、
`capacity_microbatch_lower_bound`、同步前 `local_microbatches` 和同步后
`synchronized_microbatches`。 local 数已包含配置的 minimum／divisor，未包含跨 DP 对齐；同步按 runtime 重复
packing 至一致。给定实测 count 时 local 数保持未知。 组件 advantage 和 logprob／train 使用同一 dispatch 选择；聚合
advantage 仍是独立的阶段经验拟合。

### Packing 诊断

缺少序列信息时用 `actor.observe_packing_work(config, workload, records)`；它保留 `None`，可仅覆盖一个阶段或部分
DP。 记录可提供逐 MB 的 `PackingMicrobatch`；若只有每 call／rank 汇总，使用
`microbatches=()`、`summary=PackingSummary(microbatch_count, real_tokens=..., ...)`。 MB
count 必须是同步后单次调用的整数；PPO calls 相加，DP／TP／CP／PP 不相加。汇总不能逆向生成逐 MB 数组。 预测对比的
`workload.train_microbatches` 必须为 `None`，实测次数只放在 records 中；带实测 count 的计划和来源未声明的手工 plan
会被拒绝，避免虚假的零预测误差。

`PackingSummary` 可选字段为 real／aligned／compute／tail padding tokens、attention 平方和及请求的
padded_to 总量；另可提供同步前 `local_microbatch_count`，未知保持 `None`。 已知 rank real tokens 时核对
local／同步 count 的容量下界。可确认的 `aligned + tail = compute` 会推导；未知 attention 保持未知。Archon 普通 THD
无序列 alignment 时，核对源码／路径后可由适配器填入
`compute=padded_to`、`aligned=real`、`tail=compute-real`；此规则不能套到 Megatron。

```python
work = actor.observe_packing_work(config, workload, records)
predicted = actor.estimate(config, workload).packing
packing_error = work.compare(predicted)  # 同 config、同 workload
wape, covered, expected = packing_error.metric("microbatch_count")
# 也支持 real_tokens / compute_tokens / aligned_tokens /
# padding_tokens / attention_length_squared；未知字段不计入分母，仍报告覆盖率。
# 新增 capacity_microbatch_lower_bound / local_microbatch_count，逐 rank 定位分配和局部 packing 误差。

# 每个 phase 独立拟合，calibration_points 只包含原校准集。
point = PackingTimingObservation(work, "train_step", measured_seconds,
                                "trainer_host_wall_with_stats")
diagnostic = PackingTimingDiagnostic("train_step").fit(version, calibration_points)
report = diagnostic.evaluate(point, predicted)  # point 必须是留出 run/step
# report.packing_error：packing 的逐项误差
# report.given_observed_packing：实测同步 MB 数输入 g 的耗时
# report.given_predicted_packing：预测 MB 数输入同一冻结 g 的耗时
# report.conditional_timing_error：前者的绝对相对误差
```

`g(n)=a+b*n` 为非负拟合的**经验诊断**，不接入 Actor 默认预测或配置搜索；不是 Transformer／通信公式已经验证。 `n` 为各 PPO call
同步 MB 数之和；缺 DP／call、跨 profile、token／序列数／已知最大长度／MB 数超界继续 unsupported。 常数模型须显式选
`form="constant"`。保留拟合版本、样本来源与范围，拒绝用拟合样本调用 `evaluate`。

校准集中 `[10,24]` 之类的 MB 范围不是 runtime 的合法上限。默认严格模式保留原拒绝；显式传 `extrapolate_count=True`
可只对经验诊断的 MB count 外推，返回 `evidence="assumed"` 和越界说明。 配置、版本、计时边界、token／序列／长度范围及 DP／call
覆盖仍须通过检查，原系数与边界不变。 外推结果与严格 covered 结果分别报告；不得用验证 step 63 的 25 回填校准边界，也不能据此声称默认 packing
已准确。

在同一 covered 留出子集汇总 `Σ|预测秒数−实测秒数| / Σ实测秒数` 得 WAPE，并列 P90 和覆盖率；分别报告两种输入，不把逐样本相对误差均值称 WAPE。
`packing_timing_delta=g(n_pred)-g(n_obs)` 只分解有符号残差，两项 WAPE 不能相加。 字段完整时仍可走
`work.to_exact_plan()`／`actor.observe_packing(...)` 验证原解析耗时；只有总量时该路径保持 unsupported。

预测 packing 已对齐 runtime NumPy FFD 及 Archon 两次 rounding：先 256，再
`lcm(page_size, 2*TP*CP)`；Megatron 当前限定 wrapper THD。 缺完整序列／dispatch 信息时继续标注
surrogate，无法据此声称 S026 或新配置已准确预测。profile 改变后按原划分重新拟合；S013 校准边界保持，最后 200 步不评分。

### 只有阶段总耗时的日志

显式选用 `AggregateAdvantageModel`，拟合 `T = a + b × sequence_tokens`，`a,b >= 0`。 这里的 tokens
是全局 attention tokens，不含并行复制、padding 或 PPO 重复使用；无需组件参数和精确 packing。 仅有相同 token 数时 affine
不可辨识，可在校准集内预先选择 `form="constant"`，不会自动回退。

```python
from dataclasses import replace

model = AggregateAdvantageModel(execution_id=execution_id)
actor = ExpertActorModel(shape, hardware, runtime_revision, advantage_model=model)
profile = actor.profile_key(actor_config)
points = tuple(
    AdvantageObservation(row.sample_id, profile, row.workload, row.seconds,
                         row.timing_boundary)
    for row in calibration_rows
)
actor = replace(actor, advantage_model=model.fit(fit_version, points))
prediction = actor.estimate(actor_config, validation_workload)
```

上述 `row` 字段由公司端适配。每个 trial 独立确认 runtime／config 和
execution_id；后者标识设备、controller、训练选项和标签聚合， 信息不足时限定到 run，不能跨 trial 复用。`timing_boundary`
必须经核查为 `trainer_host_wall_with_stats`； rank-mean／worker-only 标签拒绝进入这一路径，不允许仅重命名标签。

范围检查使用校准集的 tokens、n_seqs、平均长度及可用的真实最大长度；验证缺字段、超界或换配置返回 unsupported。 拟合结果标记
calibrated，组件列表为空，不能再加 `compute_advantage` 的 `PhaseCalibration` 或 `AdvantageWorkload`。
logprob／train 仍分别校准。fit 不改变执行 profile，完整系数、版本和来源进入 simulator snapshot。
形式选择与系数只使用校准集；公司端报告原验证集误差／覆盖率，保持最后 200 步未评分。

原配置使用
`replay(ReplayInitial(...), records, stop_at=..., source_id=...)`。它只检查已提供事件的一致性并记账，不验证
admission/staleness 策略；缺少 group ID 时不声称验证消费归属。模拟结果的 `snapshot_id` 包含模型代码、中性 workload
helper、目标 runtime 版本、校准、场景和计划；修改代码后应重启进程生成新快照。

## 时间线绘图标准

规范与参考图统一维护在
[Expert 进度文档](PROGRESS.md#%E6%97%B6%E9%97%B4%E7%BA%BF%E7%BB%98%E5%9B%BE%E6%A0%87%E5%87%86)；workflow
最新认知与验证结论也在该文档更新。

## 范围

- Actor：Megatron／Archon 的 FFD、PPO minibatch、阶段计算／通信和非交错 1F1B；另支持上述纯 DP FSDP2
  profile。静态显存只是下界；`overlap_grad_reduce` 暂按串行通信上界先验，收益需匹配 profile 的实测校准。
- Rollout：固定轨迹长度与结果，decode 优先、完整长度 KV 预留、逐服务迭代推进；这些是明确假设，不等同于完整 SGLang 调度器。
- `Scenario.request_routing="group_affinity"` 可将整组 samples／后续请求固定于同一副本；按 active group
  数均衡、平局轮换是模拟假设，默认 `request_load` 保留原逐请求分配。恢复中的组使用 `RunningGroup.replica`
  或一致的在途请求副本，未知映射拒绝。
- 每个 sample 的清理用前置依赖明确的 `ExternalCall` 表达，可并行且不占 GPU 请求槽；`Group.postprocess`
  另计导出等整组收尾。未知耗时不视为测量，零值或 30s 常数均须显式标为先验；示例与图见 [Expert 进度](PROGRESS.md)。
- 初始预测状态限于已提交版本、服务迭代边界，且无未开始的 submitted/pending backlog、未消费拒绝结果或已部分组好的
  batch；`complete=False` 返回 unsupported。恢复中的请求、外部调用与 group 收尾可给剩余进度。
- 发布暂停新任务 activation；已运行请求在服务边界更新权重并保留进度。显式 drain 包括已提交 backlog。`task_id` 排序用显式整数
  ID，未给时用输入序号；非确定性模式使用固定 seed。
- `Scenario.workflow="v2_inline"` 区分主线程按 batch 入队／1 秒等待和后台提交；pause 时 coroutine
  完成可先更新累计计数，结果到 resume 才交付。末步 resume 给已有 pending
  一次提交机会再结束，使用零线程交接延迟先验，不能视为真实线程调度或固定预取数量。默认 `eager` 保留原协议。
- 切换必须提供 source／target profile、依赖、阻塞和资源事件；缺失耗时不填零。终止显式 drain/cancel 并释放资源，失败保留已付成本。
- 不预测训练质量、轨迹变化或外部服务费用；目标硬件精度、跨配置排序和实际收益均未验证。

[revision evaluator](../../../autopilot/revision/README.md) 负责评估、反馈与接受判断，agent 据此修改
Basic。 expert 与 Basic 各版本在相同场景和指标下比较；expert 输出不默认作为 Basic 的修订标签。
