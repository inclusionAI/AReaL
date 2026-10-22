# Reference model

独立冻结的 Expert 成本模型与异步 simulator，供实验对照；不含 optimizer 或在线调度策略。
`actor_lifecycle.policy=reference` 仍为待实现入口。Reference 不调用 Basic，精度需由目标数据验证。

| 文件                          | 职责                                                               |
| ----------------------------- | ------------------------------------------------------------------ |
| `cost_model/`                 | Actor、advantage、Rollout、发布、RPC 与转换成本，以及 packing 诊断 |
| `simulator.py`、`scenario.py` | 请求／组依赖、队列、版本、资源状态与异步时间／GPU 成本模拟         |
| `core.py`                     | 来源、覆盖范围、校准、可行性与数值实现身份                         |
| `catalog.py`、`trial.py`      | 基础 profile 装配、外部 trial 校验及条件模拟重建                   |
| `replay.py`                   | 已提供观测窗口的计数、版本与 allocation 记账                       |
| `profiles/`                   | 模型／硬件／系统／Actor／Rollout 的基础配置与来源                  |

主要入口：`ExpertActorModel.estimate()`、`ExpertRolloutModel.estimate()`、
`ExpertSimulator.simulate()`、`Catalog()`。默认 Catalog 提供基础 profiles；历史 trial 由调用方传外部 root。
缺失耗时保持 unsupported，零值须显式声明；同配置观测和跨配置预测分别验收。

校准绑定配置、源码、NumPy、计时边界和负载范围，加载不 refit；未知组合不能套用旧系数。 目标为 AReaL v2，九条 Tau2 历史由用户确认来自
v1；版本身份不代替执行兼容核对。 当前校准、跨配置、目标硬件精度、异步收益与训练质量的验证等级分别记录。

进度统一维护在[项目进度](../../../../PROGRESS.md)。设计长文、旧进度、详细接口和绘图约定见
[本地开发归档](../../../../../artifacts/elastic-policy-development-20261021/README.md)；
原始冻结数据和历史方法见 [原始归档](../../../../../artifacts/repository-archive-20261021/README.md)。
本次整理保留全部 Reference Python／profile 内容及数值实现身份。
