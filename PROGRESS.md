# AutoPilot 进度

更新：2026-10-24。开发分支：`areal-project/AReaL:lws/feat/elastic`。
[双机开发交接](examples/autopilot/DEVELOPMENT.md)／[架构](areal/elastic/README.md)／[研究计划](examples/autopilot/PLAN.md)。

**开发入口已迁移**

- 新分支原样承接已验证 `86323615`，保留 v2.1.0 基线与原历史；迁移后的运行代码、测试与配置保持一致。
- 本地 `origin` 指向官方仓库，个人仓库保留为 `fork`。两条旧 `lws/feat/autopilot*` 保留在 `0947bfc8`。
- 公司 checkout 尚待按交接文档保存本机未推送改动并切换跟踪；模型、SIF、数据和外部 `artifacts` 由各机器配置。

**当前实现与验证**

- 唯一训练入口为 `PPOTrainer → elastic/runtime.py`；固定布局下支持 Actor
  按需启停、完整状态恢复和权重发布，Rollout持续服务。FSDP／Megatron闭环及恢复已有GPU验证。
- `state/control/monitor/policy`
  分工；Basic与冻结Expert分离，AutoPilot提供revision／activation。原生PPO与FP32 master／moments语义保留。
- Hermes／CC × DAPO／SWE共享任务、环境与评分，训练沿用online PPO；Docker默认，Slurm／Singularity为显式执行方式。
- `lws/feat/elastic-envarena`
  新增[Hermes／DAPO Arena runner](examples/task_workflows/arena/README.md)。真实平台的单题CPU链路已通过：Hermes
  0.19.0、隔离Python工具、独立Reward与产物；模型响应为模拟接口，未验证真实推理或RL训练。
- 同一固定源 revision 的 DAPO `en/train`
  全量14,116题已发布，逐题保留原prompt与独立oracle，并核对发布配置；未重新划分既有实验题单，未运行全量评测或训练。
- Arena 客户端新增固定组合检查、任务执行后端与保存身份的恢复入口，复用共享 session／admission／reward／inbox
  回执流程。相关CPU正常路径通过（服务为接口替身）；真实平台只读核对全量题目映射及一题prompt通过，真实模型／Actor仍待验收。
  独立CPU接口探针支持在开发环境验证Arena回连，JSON／SSE与工具回合的本机检查通过，远端网络待实测。
- 数学工具分开保存
  `stdout`／`stderr`，`output`仅含标准输出；宿主支持Python3.6.8，`TASK_PROC_MODE=none`可关闭工具／评分容器的`/proc`挂载。
- 工作流41项离线正常路径、真实3.6.8宿主14项跨版本检查通过；其中容器命令与模型响应为替身，完整公司环境待验收。
- 逐调用计时／usage长度／工具依赖已记录；Rollout prefill／decode batch采集默认关闭，GPU兼容性与开销待验收。

**阻塞与下一步**

1. Arena 后续需验证真实模型可达性、AReaL
   token采集、reward／持久轨迹交接。工具使用独立命名空间和受限tmpfs；进程地址空间限制与原Docker聚合内存限制仍有语义差异。
1. 公司端先跑[CPU工具探针](examples/task_workflows/RUN_SLURM_SINGULARITY.md)，确认两路输出，再做16题纯推理评测；SWE先验证参考补丁评分。
1. 1.7B正式5轮／batch40／125更新的原运行第4次更新反传OOM，仅3次更新、998题正常结果和2题服务错误，无checkpoint，相关进程已清理。正式重跑未启动；`n_mbs=10`和分配器配置均未构成已验证修复，须覆盖原失败规模及64K完整前后向。
1. 显存验收后先做16题／4更新常驻训练，核对Trainer更新数、HF导出、逐调用计时与双方退出码，再推进正式训练和其余组合；[操作文档](examples/task_workflows/RUN_2X2_8GPU.md)。
1. 完整在线Reference策略、独立拓扑调整、Expert的1.7B dense适配、trace消费／batch关联及资源收益／学习质量实验仍待完成。

验证摘要与边界见[交接文档](examples/autopilot/DEVELOPMENT.md)；详细历史保留在[迁移前进度快照](https://github.com/areal-project/AReaL/blob/86323615a3a62fc22c7371b1ceaba526ec1ada93/PROGRESS.md)。
