# AutoPilot revision

通用代码修订闭环：proposal → evaluator → acceptance → immutable revision store。 本包不依赖 Elastic
模型、执行器或领域类型；领域实现通过 `RevisionEvaluator` 协议接入。

- `contracts.py`：proposal、request、report 和内容身份。
- `interfaces.py`：领域提供初始源码、公开 API／参考材料、评测、接受规则和重放指纹。
- `agent.py`／`llm.py`：命令或 LLM proposer，只消费领域提供的公开上下文。
- `execution.py`：有超时、输出限制和子进程清理的 runner；worker 模块／命令由调用方指定。
- `loop.py`：有界反馈重试、冻结上下文、持久接受收据及原子 active pointer。

候选 Python 是受信代码，子进程限制不构成安全沙箱。接受后重新加载仍需领域校验。 历史标签隔离、模型准确性、优化器成本和原生策略部署见
[Elastic 适配器](../domains/elastic/README.md)。
