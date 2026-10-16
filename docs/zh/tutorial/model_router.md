# 模型请求路由（v2 SGLang）

默认 `rollout.model_router: disabled`，Data Proxy 仍直接访问原来绑定的推理实例。 现有 Gateway、Data Router
和轨迹聚合流程保持原来的行为。

## 开启方式

在已经使用 **v2 actor + v2 SGLang rollout** 的配置中增加：

```yaml
rollout:
  _version: v2
  model_router: global  # disabled | node | global
```

| 配置       | 首轮可选的推理实例                |
| ---------- | --------------------------------- |
| `disabled` | Data Proxy 原来绑定的实例         |
| `node`     | Data Proxy 所在服务头节点上的实例 |
| `global`   | 当前 Controller 管理的整个推理池  |

多节点 TP 按实例的服务头节点分池。`node` 要求代理的 guard 与服务头节点同机； 同机只有一个推理实例时，`node` 没有分流收益。

## 请求和数据如何流转

```text
Agent -> Gateway -> Data Proxy -> Model Router -> SGLang
                       |
                       +-> tokens、logp、reward、group 导出、RTensor
```

现有 Data Router 继续帮助 Gateway 定位 Data Proxy，`rollout.routing_strategy` 仍控制这一步的数据放置。新增的
`model_router` 只在 Data Proxy 构造好原生模型请求后 选择推理实例。轨迹记录、reward、组内处理和 dump 不移到 Model Router。

Controller 使用现有 RPCGuard 启停 Model Router：`node` 每个服务头节点一个， `global` 整个池一个。无需增加依赖或调度资源类型。

每条轨迹首轮调用模型时，选择未完成模型请求数最少的实例，同负载轮转选择； 后续轮次和 abort/resubmit 始终保持绑定，以复用 prefix cache。 一个
prompt 的不同 group 成员各有 session，可以分散到多个推理实例； 导出时仍按原来的 group 汇总。绑定键由 Data Proxy 的已认证
session 提供， 外部请求头不能覆盖。

首版使用 Router 自己记录的未完成请求数，不逐次查询 SGLang、KV cache 或 GPU 指标。 工具执行期间不算模型请求，已经绑定的长尾轨迹也不会被迁移。

## 权重更新和共卡切换

Controller 先关闭模型入口，再暂停物理推理实例并等待已转发请求结束，之后更新权重。 Data Proxy 和 Model Router
确认新版本后才恢复生成。每段响应携带准入时的版本， 被明确拒绝且未转发的请求会等待恢复，不消耗 abort/resubmit 次数。

训练和验证共用物理实例时，也共用同一组 Model Router。单独初始化 Controller 并复用 `server_infos` 时，需要同时传入训练
Controller 作为 `model_router_owner`。

共卡 offload 期间保持入口关闭，完整 `onload()` 成功后才能恢复； 此模式拒绝 offload 后只恢复部分
tags。传输结果不明、排空超时、部分实例控制失败或 权重更新失败，都会锁住推理池，需 recover；不会自动迁移或重试已准入请求。

## 支持范围和验证

- 当前支持本地 v2 SGLang 的原生非流式 `/generate`；图片、采样参数、 routed-expert 扩展字段及后端响应按原始字节转发。
- v1、vLLM、外部 `api_url` 暂不支持，开启时直接报配置错误。 公共路由模块留在 `areal/infra`，便于后续接入 v1。
- 推理池固定，开启后拒绝 `/configure_backend` 动态换后端。 全池切换使用 Controller 生命周期方法；直接调用单个 worker
  的管理接口不构成全池屏障。
- Router 的已认证 `/status` 可查看版本、绑定数和各实例未完成请求数。 session 导出并移除时释放绑定；异常遗弃的 session
  绑定保留到推理池销毁，避免破坏亲和性。

CPU 测试覆盖实际 HTTP 转发、组内图片与 RTensor 导出、轨迹绑定、版本屏障和失败处理。 上线前还需在目标多节点 GPU 环境验证 SGLang
对暂停后迟到请求的 abort 行为、数值一致性和吞吐。
