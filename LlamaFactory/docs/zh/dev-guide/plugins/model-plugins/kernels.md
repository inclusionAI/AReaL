# Kernels plugins

## 概览
LLaMA-Factory 通过 Kernels plugins 系统，依据不同硬件设备提供高性能计算内核（kernel）实现。该系统通过注册表机制管理所有 kernel，通过 `@register_kernel` 装饰器实现 kernel 定义后自动注册，由 `apply_kernel` 方法来使能指定的 kernel，`apply_default_kernels` 可使能注册表中当前环境所有可用的默认 kernels。

## 架构设计

### 核心组件

#### 1. Registry（注册表）

`Registry` 是一个用于管理所有 kernel 实现的静态类。它维护一个字典结构：`{kernel_id: KernelClass}`。

```python
# 注册表结构示例
{
    "npu_fused_rmsnorm": NpuRMSNormKernel,
    "npu_fused_swiglu": NpuSwiGluKernel,
    ...
}
```

#### 2. register_kernel (装饰器)

`@register_kernel` 是 `Registry.register` 的别名。所有 kernel 类均应使用该装饰器进行注册。

**注册机制**：
- 装饰器检查类是否继承自 `BaseKernel`。
- 检查类是否定义了 `_kernel_id` 和 `_device` 属性。
- 检查 `_device` 是否与当前运行环境的加速器类型匹配。如果不匹配，则跳过注册。
- 如果一切符合要求，将 kernel 类注册到全局注册表中。

#### 3. BaseKernel（基类）

所有 kernel 的实现都必须继承自 `BaseKernel` 抽象基类。`BaseKernel` 定义了 kernel 的基本属性和接口。

#### 4. 标识系统

**Kernel ID** (`_kernel_id`)：
每个 kernel 必须拥有一个唯一的字符串标识符，例如 `"npu_fused_rmsnorm"`。

**Device Type** (`_device`)：
kernel 必须声明其支持的设备类型，例如 `DeviceType.NPU` 或 `DeviceType.CUDA`。

## Kernel 系统 API 设计

### **Registry**：全局 kernel 注册表

`Registry` 类提供了注册和获取 kernel 的接口：

```python
class Registry:
    @classmethod
    def register(cls, kernel_cls: type[BaseKernel]) -> type[BaseKernel] | None:
        """注册一个 kernel 类"""
        ...

    @classmethod
    def get(cls, kernel_id: str) -> type[BaseKernel] | None:
        """根据 ID 获取 kernel 类"""
        ...
```

### **BaseKernel**

`BaseKernel` 定义了所有 kernel 必须实现的协议：

- `_kernel_id`: 类属性，kernel 的唯一标识符。
- `_device`: 类属性，kernel 支持的设备类型。
- `check_deps()`: 类方法，检查 kernel 的依赖项是否满足（如 `torch_npu` 是否安装）。
- `apply(**kwargs)`: 抽象类方法，实现 kernel 的具体应用逻辑。

```python
class BaseKernel(ABC):
    _kernel_id: Any = ""
    _device: DeviceType = DeviceType.CPU

    @classmethod
    def check_deps(cls) -> bool:
        """检查依赖项"""
        ...

    @classmethod
    @abstractmethod
    def apply(cls, **kwargs) -> HFModel:
        """应用 kernel 到模型"""
        ...
```

### **scan_all_kernels**

`scan_all_kernels` 函数会自动扫描 `ops` 目录下的所有 `.py` 文件并导入它们，从而触发 `@register_kernel` 装饰器完成自动注册。

### **apply_kernel**

对模型使能指定的 kernel。

```python
def apply_kernel(kernel_id: str, **kwargs) -> HFModel:
    """应用指定的 kernel 到模型

    Args:
        kernel_id: 目标 kernel 的 ID
        **kwargs: 传递给 kernel.apply 的参数，通常包含 model
    """
```

**用法示例**：
```python
from llamafactory.v1.plugins.model_plugins.kernels import apply_kernel

model = apply_kernel("npu_fused_rmsnorm", model=model)
```

### **apply_default_kernels**

对模型使能所有默认注册的 kernel。这是一个高级 API，通常在模型加载流程中自动调用。

```python
def apply_default_kernels(model: HFModel, include_kernels: str = None) -> HFModel:
    """应用所有默认 kernel

    Args:
        model: HFModel 实例
        include_kernels: 包含的 kernel ID 列表（逗号分隔字符串），或者 "auto"/True 表示全部
    """
```

## 扩展 Kernels

如果用户有针对特定模型或者设备的 kernel，可以按照下述步骤去实现并接入 LLaMA-Factory。

### 创建新 Kernel 的步骤

#### 1. 创建 Kernel 实现文件

在 `src/llamafactory/v1/plugins/model_plugins/kernels/ops` 下的相应子目录中创建新的 kernel 实现文件，例如 `mlp/cuda_swiglu.py`：

```python
import torch
from ......accelerator.helper import DeviceType
from ......utils.types import HFModel
from ...base import BaseKernel
from ...registry import register_kernel

# 实现具体的 kernel 函数
def _cuda_swiglu_forward(self, hidden_state):
    # ... CUDA 优化实现 ...
    pass

@register_kernel
class CudaSwiGluKernel(BaseKernel):
    _kernel_id = "cuda_fused_swiglu"
    _device = DeviceType.CUDA

    @classmethod
    def apply(cls, **kwargs) -> HFModel:
        model = kwargs.get("model")
        if model is None:
            raise ValueError("model is required")

        if not cls.check_deps():
            raise RuntimeError("Dependencies not met")

        # 遍历模型并替换 forward 方法
        for name, module in model.named_modules():
            # ... 匹配和替换逻辑 ...
            pass

        return model
```

#### 2. 自动发现

由于 `scan_all_kernels` 会自动扫描 `ops` 目录，只要文件位于该目录下且没有语法错误，系统启动时会自动导入并注册，无需手动修改注册表代码。

#### 3. 测试 Kernel

创建测试用例验证 kernel 的正确性：

```python
from llamafactory.v1.plugins.model_plugins.kernels import apply_kernel

# ... 加载模型 ...
model = apply_kernel("cuda_fused_swiglu", model=model)
# ... 验证 forward 是否被替换 ...
```

## 异常处理

### 依赖不可用

`BaseKernel.check_deps()` 默认会检查当前设备类型是否匹配。子类可以重写此方法以添加额外的依赖检查（如检查特定的库是否安装）。如果 `check_deps()` 返回 `False`，`apply()` 方法应当抛出异常或进行相应处理。

### Kernel ID 未找到

如果调用 `apply_kernel` 时传入了不存在的 `kernel_id`，会抛出 `ValueError`。
