# AReaL Elastic

Elastic 是 AReaL 的执行与规划能力。唯一训练执行路径仍是 `PPOTrainer → OnDemandActor`；本包不依赖 AutoPilot。

```text
elastic/
  contracts.py       观测、动作、负载与公共契约
  state/             checkpoint、消费游标、持久队列与在线 inbox
  control/           进程所有权、配置／策略激活事务、原生控制接口
  telemetry/         原始实测历史
  policy/
    basic/           Basic cost model、simulator、optimizer 与 policy
    reference/       独立 Expert 参考模型／simulator；不参与 Basic 实现
```

原生执行器通过公开 `NativeControlPort` 提供串行控制边界和 admission fence。 配置应用目前只接受同布局、同 GPU
池的身份确认；不支持实际拓扑调整。 训练、完整 checkpoint／游标提交、全副本权重发布确认和资源释放仍由原生执行器负责。

AutoPilot 的通用修订机制位于 [revision](../autopilot/revision/README.md)， Elastic evaluator、候选
worker、bundle 验证和原生策略桥接位于
[domains/elastic](../autopilot/domains/elastic/README.md)。源码迁移改变执行与评测身份， 旧 checkpoint／接受
bundle 不能仅修改路径后继续视为有效；Expert 校准须另做兼容审计。
