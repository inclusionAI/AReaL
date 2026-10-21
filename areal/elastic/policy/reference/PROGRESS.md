# Expert simulator／cost model 进度

更新：2026-10-21。[接口](README.md)／[设计](DESIGN.md)／[项目进度](../../../../PROGRESS.md)。

- 独立对照迁入 `areal.elastic.policy.reference`；Basic／revision 不依赖其实现。公式和基础 profiles
  保留，源码身份正常变化。
- [迁移审计](../../../../../artifacts/elastic-policy-layout-20261021/reference-migration.json)：NumPy
  1.26.4；16 步 raw／strict 预测、packing、RPC 与冻结系数一致；strict unsupported 为 `[9,11,16]`。
- cleanup=30／0 各完成 16 updates、77 groups；旧新完整事件流与 GPU 秒一致，均保留 21,817 条事件，耗时分别为
  1976.826s／1542.879s；无 refit。
- 原始 catalog 不变；仅审计后的派生 catalog 添加精确实现／数值环境兼容项。默认 `Catalog()` 只提供基础 profiles；档案测试显式设置
  `AREAL_TEST_EXPERT_CATALOG`。
- 本轮 CPU 回归 1000／26，含 16 项未提供档案的跳过；新目录兼容审计通过。原始回放与当前原生 GPU 验收分别记录；服务器不可访问，GPU 验收待补。
- [历史方法与证据](../../../../../artifacts/repository-archive-20261021/files/areal/autopilot/elastic/policy/expert_designed/PROGRESS.md)保留原适用版本和未知范围。跨配置、目标硬件独立精度和收益仍待验。

## 时间线绘图标准

- Workflow
  边界及绘图约定见[历史进度](../../../../../artifacts/repository-archive-20261021/files/areal/autopilot/elastic/policy/expert_designed/PROGRESS.md#%E6%97%B6%E9%97%B4%E7%BA%BF%E7%BB%98%E5%9B%BE%E6%A0%87%E5%87%86)；新认知在本文件修订并注明版本、证据、假设和未知。
- R10 到达回放与重建 Rollout 的条件模拟分别计分；保留原始计时、缺失边界和来源身份。
