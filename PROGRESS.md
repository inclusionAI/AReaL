# AutoPilot 进度

更新：2026-10-21。[架构](areal/elastic/README.md)／[计划](examples/autopilot/PLAN.md)／[执行接口](examples/autopilot/elastic/README.md)。

**当前状态**

- 主开发目录 `AReaL`；主线 `feature/autopilot-elastic`，基线 `v2.1.0`（`ecc8b0e4`）；历史备份
  `backup/elastic-before-squash-20261021`。
- `policy/basic` 提供可修订的规划与策略；`policy/reference` 提供独立冻结参考模型，不含 optimizer／调度策略。
- Elastic 能力归入 `areal/elastic/{state,control,telemetry,policy}`；不依赖 AutoPilot。
- `autopilot/revision` 仅保留通用 proposal、评测编排、接受、预算与存储；Elastic evaluator、worker、bundle 和
  native bridge 归入 `autopilot/domains/elastic`。
- 唯一执行仍为 `PPOTrainer → OnDemandActor`，默认 resident；通过公开控制接口串行激活事务，原生
  checkpoint／游标、发布确认和释放门禁保留。
- 旧执行器、旧导入兼容层和重复目录文档已移除；56
  个历史档案仍在[本地归档](../artifacts/repository-archive-20261021/README.md)。

**已验证**

- [本轮验收](examples/autopilot/elastic/validation/policy-layout-20261021.json)：`policy/{basic,reference}`
  已迁移，CPU **1000 通过／26 跳过**；本轮未提供档案，含 16 项档案测试跳过。
- 新增依赖隔离与公开控制边界回归；生命周期、事务恢复、候选执行和 revision 预算回归通过；完整 pre-commit 通过。
- Expert 16 步 raw／strict 预测、packing、RPC、冻结系数一致；cleanup=30／0 的旧新完整事件流、耗时与 GPU 秒一致，无
  refit。新实现仅在派生 catalog 增加精确兼容身份，原始归档 99 条文件记录未变。
- 原始 R10 CLI 回放计时对齐通过，严格 unsupported 仍为 `[9,11,16]`。源码迁移改变身份，旧接受 bundle／policy
  checkpoint 不能直接复用。

**阻塞与下一步**

- SSH 仍在跳板断开，本轮未启动 GPU 任务，远端已有资源状态未知。
- 原生当前固定布局、horizon=1；独立拓扑调整、协调器残留 owner／pair 自动恢复、异步 GPU·h 收益和质量仍待验。
- 恢复连接后核对资源，依次验 FSDP／Megatron 同布局三轮启停、真实 Agent Service → GSM8K online RL → Hermes。
