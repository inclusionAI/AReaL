# Basic policy

原生 `on_demand` 通过 `policy/adapter.py::PolicyAdapter` 统一装配和调用策略，默认使用
`IdleTimeoutPolicy`。默认空闲 60s 后请求释放， 支持零超时；最短驻留、发布、保存与进程清理由 Runtime 约束。完整 Basic 尚未成为默认策略。

| 文件                             | 职责                                             |
| -------------------------------- | ------------------------------------------------ |
| `policy.py`                      | 固定超时基线、模型生命周期决策和离线联合规划入口 |
| `optimizer.py`                   | 有限视野搜索、成本／时间约束、切换门槛与查询预算 |
| `simulator.py`                   | 阶段组合、负载预测、驻留 GPU 成本及联合模拟      |
| `actor_cost.py`                  | Actor 模型到阶段预测／生命周期策略的适配         |
| `joint.py`、`serial_planning.py` | 联合配置协议与离线严格串行模型／决策             |
| `profile.py`                     | 实测 workflow 装配与固定布局模拟边界             |
| `serialization.py`               | 公共规划输入的 JSON 解码                         |
| `cost_model/`                    | Actor、Rollout 与转换成本模型、校准和基础数据    |

`BasicPolicy.decide()` 处理生命周期，`JointPolicy.plan()` 返回离线联合计划；Basic 独立于 Reference。
运行历史统一来自默认 Monitor，超时基线也会积累数据；启用采集不自动切换模型搜索。 模型比较固定样本量和训练语义，缺少测量或状态时明确拒绝预测；配置转换需显式声明兼容。
原生实测模型适配目前仍要求显式 profile、固定布局、horizon=1，离线候选不代表 GPU 已支持重配置。

[成本模型](cost_model/README.md)／[原生执行接口](../../../../examples/autopilot/elastic/README.md)。
离线 CLI、R10 专用回放辅助模块与原详细说明见
[本地开发归档](../../../../../artifacts/elastic-policy-development-20261021/README.md)。
