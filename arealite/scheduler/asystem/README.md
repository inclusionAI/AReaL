# AsystemScheduler 重构说明

本目录包含了重构后的 AsystemScheduler，使其与 LocalScheduler 的API结构保持一致。

## 文件结构

```
arealite/scheduler/asystem/
├── __init__.py          # AsystemScheduler 主类
├── rpc_client.py        # AsystemRPCClient - 分离的RPC客户端
├── test.py             # 使用示例和测试
├── test_asystem.py     # 基本功能测试
├── test_asystem_mock.py       # 基本模拟功能测试
├── client_old.py       # 原始的AsystemClient实现（已废弃）
└── README.md           # 本文档
```

## 主要变化

### 1. 结构对齐

重构前的 `AsystemClient` 继承自 `SchedulerClient`，现在的 `AsystemScheduler` 继承自 `Scheduler` 基类，与 `LocalScheduler` 保持一致。

### 2. RPC客户端分离

将HTTP请求逻辑从主调度器类中分离到独立的 `AsystemRPCClient` 类，遵循与 `LocalScheduler` 相同的模式：
- `LocalScheduler` 使用 `RPCClient`
- `AsystemScheduler` 使用 `AsystemRPCClient`

### 3. 统一的API接口

现在 `AsystemScheduler` 和 `LocalScheduler` 具有完全相同的方法签名：

```python
# 都继承自 Scheduler 基类
class AsystemScheduler(Scheduler):
class LocalScheduler(Scheduler):

# 相同的方法
def create_workers(self, scheduler_config: SchedulingConfig, *args, **kwargs)
def get_workers(self, timeout: float = 300.0) -> List[Worker]  
def delete_workers(self, name: str = None)
def create_engine(self, worker_id: str, engine_obj: Any, init_config: Any = None)
def call_engine(self, worker_id: str, method: str, *args, **kwargs)
```

### 4. 标准数据类型

使用 `arealite.scheduler.base` 模块中的标准数据类型：
- `SchedulingConfig` - 调度配置
- `ContainerSpec` - 容器规格
- `Worker` - 工作节点信息

## 使用方法

### 基本用法

```python
from arealite.scheduler.asystem import AsystemScheduler
from arealite.scheduler.base import SchedulingConfig, ContainerSpec

# 创建调度器
config = {
    "type": "asystem",
    "endpoint": "http://33.215.20.149:8081",
    "expr_name": "my_experiment",
    "trial_name": "trial_01"
}
scheduler = AsystemScheduler(config)

# 创建容器规格
spec = ContainerSpec(
    cpu=4000,
    gpu=1,
    mem=20000,
    container_image="/storage/openpsi/images/areal-latest.sif",
    cmd="python -m my_worker",
    env_vars={"NCCL_DEBUG": "INFO"},
    port=8080
)

# 创建调度配置
scheduling_config = SchedulingConfig(
    replicas=4,
    specs=[spec]
)

# 使用调度器
scheduler.create_workers(scheduling_config)
workers = scheduler.get_workers(timeout=300)
scheduler.create_engine(workers[0].id, my_engine, init_config)
result = scheduler.call_engine(workers[0].id, "my_method", arg1, arg2)
scheduler.delete_workers()
```

### 与 LocalScheduler 的兼容性

现在可以在相同的代码中无缝切换不同的调度器：

```python
def create_scheduler(scheduler_type: str):
    if scheduler_type == "local":
        from arealite.scheduler.local import LocalScheduler
        return LocalScheduler({"type": "local"})
    elif scheduler_type == "asystem":
        from arealite.scheduler.asystem import AsystemScheduler
        return AsystemScheduler({
            "type": "asystem",
            "expr_name": "test",
            "trial_name": "trial01"
        })

# 使用相同的API
scheduler = create_scheduler("asystem")  # 或 "local"
scheduler.create_workers(config)
workers = scheduler.get_workers()
# ... 其他操作完全相同
```

## 测试

运行基本功能测试：

```bash
python -m arealite.scheduler.asystem.test_asystem
```

运行完整示例：

```bash  
python -m arealite.scheduler.asystem.test
```

## 配置选项

AsystemScheduler 支持以下配置选项：

- `type`: 调度器类型，应设为 "asystem"
- `endpoint`: Asystem服务器地址，默认 "http://33.215.20.149:8081"
- `expr_name`: 实验名称
- `trial_name`: 试验名称

## 迁移指南

如果您之前使用 `AsystemClient`，请按以下步骤迁移到新的 `AsystemScheduler`：

### 1. 导入更改

```python
# 旧版本
from arealite.scheduler.asystem.client import AsystemClient

# 新版本  
from arealite.scheduler.asystem import AsystemScheduler
```

### 2. 初始化更改

```python
# 旧版本
client = AsystemClient("expr_name", "trial_name", "endpoint")

# 新版本
scheduler = AsystemScheduler({
    "type": "asystem",
    "endpoint": "endpoint", 
    "expr_name": "expr_name",
    "trial_name": "trial_name"
})
```

### 3. 方法调用更改

```python
# 旧版本
client.submit(instance_config)
server_infos = client.wait(timeout)
client.call("method_name", **kwargs)
client.stop("job_name")

# 新版本
scheduler.create_workers(scheduling_config)  
workers = scheduler.get_workers(timeout)
scheduler.call_engine(worker_id, "method_name", **kwargs)
scheduler.delete_workers()
```

## 注意事项

1. 新的API使用标准的数据类型，确保类型安全
2. RPC客户端已分离，便于测试和维护
3. 错误处理和日志记录得到改进
4. 与LocalScheduler完全兼容，便于开发和调试

## 未来计划

- 添加更多的错误恢复机制
- 支持更复杂的资源分配策略  
- 与其他调度器后端集成
- 性能优化和监控功能 