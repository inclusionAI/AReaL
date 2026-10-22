# Basic cost model

`ActorCostModel.estimate(config, workload)` 估计 `recompute_logp`、`compute_advantage`、
`train_step`；训练阶段包含全部 PPO minibatch 更新。模块独立于训练引擎与 Reference。

- `config.py`：完整后端／并行／精度／训练选项与硬件模型。
- `packing.py`：trajectory 原子分发、PPO 拆分、FFD、padding 与 DP 同步数量。
- `components.py`、`actor.py`：算子、通信、调度及三个 Actor 阶段的组合。
- `rollout.py`、`transition.py`：推理服务及配置转换的显式成本。
- `calibration.py`、`catalog.py`、`data/`：有限公开校准、基础 profile 与带来源的必要输入。

负载使用实际 Actor sequence／attention tokens，group、sample、sequence 和 loss tokens 分开计数。
缺少精确长度／dispatch 信息时标注代理；静态显存仅是下界。带宽单位为单向十进制 GB/s。
校准限定完整配置、负载与来源范围；跨硬件／后端外推需显式声明，公开总步时间不能辨识内部阶段。 目标硬件各阶段精度、运行中布局转换、训练质量和实际节省仍待独立验证。

日志／trace 解析位于 `elastic/monitor/trace.py`，CPU wall span 不作为 CUDA kernel 时间。 原
CLI、详细公式／使用说明与专用实验入口见
[本地开发归档](../../../../../../artifacts/elastic-policy-development-20261021/README.md)。
