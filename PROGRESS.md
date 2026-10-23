# AutoPilot 进度

更新：2026-10-24。[架构](areal/elastic/README.md)／[计划](examples/autopilot/PLAN.md)／[GPU 验收证据](../artifacts/gpu-native-acceptance-20261022/README.md)／[实测阶段时间线 SVG](../artifacts/elastic-timeline-20261022/README.md)。

**Hermes 1.7B 常驻 PPO：等待GPU资源，OOM修正待实测**

香港时间14:52检查：八卡均被另一项root服务`efair-h20-headroom-ext1-actual-lookahead-normal-1`占用约81GiB／卡，磁盘可用约139GB；用户要求等待，未启动重跑。[重跑提案](../artifacts/hermes-qwen3-17b-20261023/train-resident-5ep-retry-proposal.json)复核确认：`n_mbs=10`仅为最少微批数，不能保证长序列显存安全；64K单份BF16
logits约18.55GiB，分配器配置不构成已验证修复。资源空闲后须先验证原失败规模及64K完整前后向，再决定正式重跑；5轮／batch40／125更新保持。原`train-1000x5-run1`第4次更新反传OOM，仅3／125更新、998题完成和2题服务错误，无checkpoint，Trainer／客户端均退出1，见[显存分析](../artifacts/hermes-qwen3-17b-20261023/train-resident-5ep-oom-analysis.json)和[故障证据](../artifacts/hermes-qwen3-17b-20261023/train-resident-5ep-check-20261023T035234.json)；[本轮残留已清理](../artifacts/hermes-qwen3-17b-20261023/train-resident-5ep-failure-cleanup.json)，998条正常结果逐题／逐调用计时完整。30分钟巡检`hermes`仍仅跟踪原运行。

[前三步时间图](../artifacts/hermes-qwen3-17b-20261023/hermes-resident-sample-timeline.png)的Rollout仍标“未采集”，历史数据不能补造；[Agent局部](../artifacts/hermes-qwen3-17b-20261023/hermes-agent-service-detail.png)灰区保持未归因，客户端逐轮构造是候选来源，尚非历史逐段归因。

**当前实现**

- Rollout batch采集默认关闭，分别记录prefill／decode请求数、上下文token与CUDA
  event时间，mixed拆分计数；采集代码已推送`cc430cfd`，5项CPU检查通过，GPU兼容性和开销待公司端实测，见[采集说明](docs/zh/best_practices/perf_profiling.md#sglang-batch-%E8%AF%B7%E6%B1%82%E6%95%B0%E4%B8%8E-token-%E6%97%B6%E5%BA%8F)。
- 主目录 `AReaL`，分支 `feature/autopilot-elastic`，基线
  `v2.1.0`（`ecc8b0e4`）；共享DAPO／SWE工作流、CC适配和2×2操作文档已纳入本次提交，供公司端拉取实测。
- 唯一执行：`PPOTrainer → elastic/runtime.py`。PPO → 发布确认 → 周期保存／评估／清理 → 恢复 Rollout →
  判断释放；释放前补齐完整状态与消费游标。
- `on_demand` 默认 Basic 固定超时；`PolicyAdapter` 接策略，AutoPilot `RevisionAdapter`
  接修订／激活。Reference 在线入口启动前报未实现。
- `state/control/monitor/policy` 分工；v2 保存完整状态与 rank LR／RNG；区分
  applied／published／checkpoint。Monitor 直接采集执行与轨迹，不读取 W&B。
- 发布不确定时封闭 admission，停止故障训练代次；重建 AWEX 后，同版本也完整发布恢复权重。专用通信组截止／abort 与独占线程已完成真实故障验证。
- PPO、FP32 master／moments、Basic／Reference
  数学未改。非核心材料见[开发归档](../artifacts/elastic-policy-development-20261021/README.md)与[原始归档](../artifacts/repository-archive-20261021/README.md)。

**已验证：固定布局原生按需训练阶段 1–3**

| 验收                                | 结果                                                                                                                                                     |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| FSDP → 真实 v2 Agent Service／GSM8K | 冒烟24轨迹／6更新／3代；1.5B正式48轨迹／12更新，S2与4次429通过。正式组含空间保护后从checkpoint3重放4–6，共5代                                            |
| Megatron d2p2＋SGLang d2t2／GSM8K   | 正式96轨迹／12更新／3代（639.8s），S2、3次429与持续到达通过；最终同步实现另以48轨迹／6更新／3代回归通过                                                  |
| 完整恢复冻结对照                    | FSDP 1257项、Megatron 142项DCP元数据全部读取；分别1244／146张量逐位一致，LR／RNG／step／消费游标／CPU逻辑进度一致，原阈值未放宽                          |
| Hermes 0.19.0／原生长上下文4B       | 24 episode／48轮外部交互／26次真工具／6更新／4代（922.3s）通过，含跨权重版本episode；任务失误保持0分                                                     |
| 交接／Actor故障                     | 丢失导出与ACK响应、重复callback、发布确认丢失、旧pair拒绝、真实worker被杀后恢复通过；24唯一轨迹／6逻辑更新／8次训练／5代                                 |
| AWEX真实sender故障                  | 16唯一轨迹／4逻辑更新／5次训练／3代；完整重发checkpoint2后仅重放3。两个SG推理PID不变，真实端点拒绝旧pair                                                 |
| CPU／静态                           | 扩大回归1009通过，遗漏admission字段夹具随后定向通过；最终通信／恢复回归Linux113通过，Ruff与diff检查通过。macOS缺Megatron及CUDA／Triton的收集限制不计通过 |

健康代次的 Rollout PID 均保持；各GPU任务结束无残留。失败现场、逐token
staleness、来源ACK与重放审计见证据链接。测试服务及本轮两个专用容器已停止；完整状态的文件摘要与小证据已保存到本地。

**边界与后续**

- [Slurm＋Singularity执行方式](examples/task_workflows/RUN_SLURM_SINGULARITY.md)与默认Docker并列保留：宿主私有socket服务启动逐题SIF／独立overlay，Hermes／CC共用工具与SWE评分；宿主入口已拆为Python3.6.8标准库模块，训练SIF仍用3.12，Docker默认保留。33项正常路径及真实3.6.8宿主／3.12客户端跨版本7项通过，见[兼容性证据](../artifacts/task-workflows-singularity-py36-20261024/offline-validation.json)。[公司已有CPU探针](../artifacts/company-apptainer-probe-20261023/evidence.json)仅验证Apptainer1.3.2经`singularity`／`srun --mpi=none`运行Python；新overlay／隔离／GPU／NCCL／训练待公司验收，原长序列OOM未解决。
- [Expert观测补采](../artifacts/expert-trace-collection-20261023/offline-validation.json)：Hermes／CC逐调用usage长度、工具fan-out依赖、锁等待／服务时间、清单就绪时间及CPU轨迹导出工作量已记录；26项离线正常路径通过，GPU待验收。缓存未知保留null，实际消费／控制边界仍须对照；Expert源码／系数未改，1.7B
  dense适配仍待做。
- [2×2八卡操作文档](examples/task_workflows/RUN_2X2_8GPU.md)已覆盖Hermes／CC ×
  DAPO／SWE在H20／H200上的评测、4＋4常驻PPO、训练后评测与trace缺口。shell／内嵌Python语法与8种训练配置解析通过，见[文档检查](../artifacts/task-workflows-2x2-docs-20261023/checks.json)；未启动GPU任务，原OOM未解决，引擎阶段／batch关联仍待实现。
- [任务工作流已统一](examples/task_workflows/README.md)：旧目录与兼容入口已删除，八卡评测／训练文档及
  `configs/dapo_17b_resident.yaml`
  均迁入同一目录。1.7B正式配置解析结果与迁移前完整一致，删除旧目录后21项离线回归通过、无跳过，见[整合证据](../artifacts/task-workflows-consolidation-20261023/offline-validation.json)。模型／Docker传输仍为替身，未启动服务器训练；原
  OOM 未解决。
- [CC 已接入共享工作流](examples/task_workflows/README.md#%E4%BD%BF%E7%94%A8-claude-code)：`CCTaskAgent`
  通过每题 relay 绑定 session／episode，通过 MCP 复用 Python／仓库环境及 DAPO／SWE 评分。真实
  SDK0.1.71／CLI2.1.123
  的多轮工具与奖励正常路径、并发隔离共3项通过，合并回归21项无跳过，见[证据](../artifacts/cc-task-workflows-20261023/offline-validation.json)。模型／Docker
  传输为替身；真实 GPU token／logprob／训练及压缩／子调用仍待实测，SDK可见 transcript 与逐调用计时已保存。
- [Hermes＋DAPO 数学工作流](examples/task_workflows/README.md)：四个适配文件已实现，8 项本地正常路径检查及
  resident／on_demand 结构化配置校验通过。 英文 14,116 条原始题目／标签保留，已冻结 200／1,000／200
  题清单，见[数据身份与检查记录](../artifacts/hermes-math-implementation-20261022/checks.json)。
  八卡纯推理首轮200题评测及16题resident训练闭环已完成，正式学习与到达率对照待开展。
- H20 环境：SSH正常，镜像与依赖未改动，`/mnt/data`可用约139GB。本轮失败进程与训练Agent已停止，八卡现由另一项系统任务占用；原日志与结果保留。
- [八卡推理评测操作步骤](examples/task_workflows/EVAL_8H20.md)：TP1×DP8、1 pair、32 episode
  并发；首轮固定200题全部完成，正确150（最终答案匹配率75%），基础设施错误0、未提交0、模型调用313、工具执行51（用户汇总）。本地核对抽题未按表现筛选、标签不进prompt，三份清单的ID与规范化题干均无交集；远端逐题评分与耗时待审阅，不能据此排除预训练数据重叠。并发实际到达时序与内部调用交错不保证一致。
- 客户端逐题进度与调用墙钟计时已同步服务器：`episodes.jsonl`保存总耗时、累计inference／tool耗时及逐调用起止时间。两项本地正常路径检查、Ruff及真实2题评测通过，见[计时记录](../artifacts/hermes-qwen3-17b-20261023/timing-smoke-default.json)；本轮16题训练计时亦完整。本机缺`torchdata`的旧按需用例不计通过。
- [Qwen3-1.7B](examples/task_workflows/EVAL_8H20.md#9-%E5%88%87%E6%8D%A2%E5%88%B0-qwen3-17b)
  非thinking／64K YaRN，TP1×DP8／1
  pair／32并发：原8轮预算下固定200题完成，正确48（24%）、基础设施错误0；396次模型调用／211次工具执行。远端逐题记录确认总时长230.9s，单题平均33.0s／P95
  60.4s；114题仅一次模型调用，10题被标记未完成并按0分计，见[统计与边界](../artifacts/hermes-qwen3-17b-20261023/eval-200-analysis.json)。正式学习与到达率对照待开展。
- 按用户确认恢复 Hermes 原生默认：数学适配器未指定时不再覆盖轮数，已有容器启动 Agent
  时清除8轮环境变量。现有采样／纯评测两项检查、真实0.19实例90轮预算验证通过；Agent已重启且四组件健康，SGLang进程不变，见[生效记录](../artifacts/hermes-qwen3-17b-20261023/hermes-native-default.json)。新预算下未重跑评测，结果使用独立目录。
- 用户确认先用现有在线PPO。1.7B／64K
  YaRN／非thinking常驻16题／4次更新验证通过，见[验收记录](../artifacts/hermes-qwen3-17b-20261023/train-resident-smoke-16.json)；B40正式5轮运行OOM失败，不计通过。训练质量、同预算200题基线及到达率对照待完成；当前不代表完整DAPO算法。
- 后续暂不跑边界情况与故障注入测试；正常主链路的必要验证在改动集中完成后统一运行，已有用例与证据保留，见 [AGENTS.md](AGENTS.md)。
- 完整状态使用独享80／200GiB tmpfs，只证明同容器进程启停，未验证主机／容器丢失后的持久恢复；原始失败不计通过。
- 当前固定布局、horizon=1、Basic超时策略。完整Basic、Reference在线策略、独立拓扑调整、残留owner自动恢复仍待完成。
- GPU进程释放不等于归还Slurm配额；异步GPU·h收益与学习质量尚未验证。当前验收不替代上述研究实验。
