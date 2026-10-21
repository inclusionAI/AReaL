# AutoPilot 进度

更新：2026-10-21。[实施计划](examples/autopilot/PLAN.md)。

**当前状态**

- 主线 `feature/autopilot-elastic`；上游基线 AReaL `v2.1.0`（`ecc8b0e4`），本轮实现相对 `729d4a8e`。
  历史整理保留 `backup/elastic-before-squash-20261021`；压缩仅整理提交历史，不改变实现。
- 原生按需训练第 1–3 步已接主要机制，尚未完成；H20 SSH 连续三轮被跳板机断开。第 4 步独立拓扑调整不在本轮。
- 原生 PPOTrainer 新增 opt-in Actor 生命周期，默认 resident 保留：完整 batch 唤醒，原生 PPO 更新，完整
  checkpoint／游标提交后发布，安全空闲释放。
- 持久 episode admission／inbox、逐 token 版本、export／ack 与取消幂等已接入；未知生成保持隔离，迟到请求按会话身份拒绝。
- AWEX 已接双端断联、代次隔离、失败状态保留及全副本版本确认；完整 checkpoint 接口覆盖 FSDP／Megatron LR／RNG sidecar 和全
  rank 核验。
- Hermes 已接 episode 身份转发；GSM8K AgentRunnable 已接单样本推理与 completion ID 返回。CPU 服务和数据进度跨
  Actor 代次存活。

**已验证**

- [本轮回归](examples/autopilot/elastic/validation/native-on-demand-20261021.json)：macOS CPU
  **456 通过／9 跳过**，完整 pre-commit 通过；当前 Python 与测试快照一致。 包含三轮生命周期协议、取消／迟到请求及 GSM8K 服务 HTTP
  链路与真实 scorer；模型端为 stub，未运行 GPU 训练。 跳过为既有 schema／CUDA 用例；Linux-only Megatron/AWEX
  adapter 本地未运行，较早 Linux 358／9 仅覆盖接线前基础。
- GSM8K 固定 revision 经原生 formatter 读入 7,473 条 train 数据，前 96 条摘要与 scorer 正／负例已核验。
  Qwen2.5-1.5B 已在本地校验；远端传输未完成。此前 8×H20 空闲，当前资源和任务容器状态无法确认。
- [文献调研](examples/autopilot/research/related-work/README.md)：两主线／三层、36
  篇来源，proposal、证据矩阵及基线已更新；动态配置和修订净收益仍待实测。
- [整合前基线](examples/autopilot/elastic/validation/elastic-expert-integration-20261021.json)：634
  CPU 通过／1 CUDA 跳过；不作为本轮原生按需实现的验收。

**待完成**

- 恢复 SSH 后先核对任务容器、GPU 占用及模型续传；CPU 协调器崩溃后的 owner／pair 自动清理恢复待补，当前拒绝覆盖未确认 journal。
- 完成真实 Agent Service → GSM8K online RL → Hermes 多轮／工具；验证 FSDP／Megatron
  同布局恢复、三轮启停、故障、异步重叠、延迟、GPU·h 与资源清理。本轮 GPU 任务尚未启动。
- 正式 B=4、S=2、额度 12；1 个完整合格 batch 唤起，安全提交与发布后 idle≥60s 且驻留≥120s 释放。冒烟 S=0／idle=5s／驻留=0。
- 先讨论模块边界、协议／状态契约和证据归档规范，再决定整理范围；保留当前行为与验证边界。

历史入口：[M2](examples/autopilot/elastic/validation/joint-runtime-revision-20261019.json)／[R10](examples/autopilot/elastic/validation/r10-actor-replay-20261020.json)均不代替原生按需验收；
[Milestone](examples/autopilot/elastic/milestones/preparation-20261021/README.md)／[Expert 进度](areal/autopilot/elastic/policy/expert_designed/PROGRESS.md)。
