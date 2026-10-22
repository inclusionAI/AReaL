# AutoPilot 进度

更新：2026-10-22。[架构](areal/elastic/README.md)／[计划](examples/autopilot/PLAN.md)／[GPU 验收证据](../artifacts/gpu-native-acceptance-20261022/README.md)／[实测阶段时间线 SVG](../artifacts/elastic-timeline-20261022/README.md)。

**当前实现**

- 主目录 `AReaL`，分支 `feature/autopilot-elastic`，基线 `v2.1.0`（`ecc8b0e4`）。
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

- 后续暂不跑边界情况与故障注入测试；正常主链路的必要验证在改动集中完成后统一运行，已有用例与证据保留，见 [AGENTS.md](AGENTS.md)。
- 完整状态使用独享80／200GiB tmpfs，只证明同容器进程启停，未验证主机／容器丢失后的持久恢复；原始失败不计通过。
- 当前固定布局、horizon=1、Basic超时策略。完整Basic、Reference在线策略、独立拓扑调整、残留owner自动恢复仍待完成。
- GPU进程释放不等于归还Slurm配额；异步GPU·h收益与学习质量尚未验证。当前验收不替代上述研究实验。
