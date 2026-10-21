# Elastic policy and reference model

[Basic](basic/README.md) 包含已有生命周期策略与离线 Actor＋Rollout 联合配置模型／搜索。
[expert_designed](expert_designed/README.md) 包含独立的固定参考模型、模拟器和 R10 档案，已合入 Elastic 主线。
Basic 与 Expert 保持独立实现；实验范围和后续计划见 [Tau2 交接](../../../../examples/autopilot/PLAN.md)。

```text
policy/
├── cli.py                       # 现有 Basic 离线场景入口
├── basic/
│   ├── cost_model/              # actor / rollout / transition 预测
│   ├── actor_cost.py            # actor 模型与工作流适配
│   ├── joint.py                 # 离线联合配置、固定工作量与结果
│   ├── simulator.py             # 工作流预测与驻留成本
│   ├── optimizer.py             # 配置与生命周期搜索
│   └── policy.py                # IdleTimeoutPolicy / BasicPolicy / JointPolicy
└── expert_designed/             # 独立固定参考模型、模拟器与档案
    ├── cost_model/
    │   ├── actor.py
    │   ├── rollout.py
    │   └── transition.py
    └── simulator.py
```

- `BasicPolicy` 通过 `SchedulingPolicy` 接入 `ElasticLoop`；`JointPolicy.plan()` 只返回离线搜索结果。
- expert 仅负责预测／模拟，不含 optimizer，也不实现调度策略；不作为 Basic 的代码依赖。
- [revision evaluator](../../revision/README.md) 负责评测反馈和接受判断，agent 据此修改 Basic。 expert
  作为固定对照，用于比较 Basic 各版本的表现。
- 公共数据类型位于 `autopilot/contracts.py`；Basic 实现导入使用 `policy.basic.*`，旧路径兼容层已移除。

现有标准库 CLI 仍可在仓库根目录运行：

```bash
python -S areal/autopilot/elastic/policy examples/autopilot/elastic/lifecycle_scenario.json
```

示例预测尚未经目标 H200／Archon 集群实测校准。GPU `TrainGroup` 只执行 pure-DP；named 混合布局仍需对应 executor。
