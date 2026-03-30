# LLaMA-Factory Kernels 系统

## 概述

LLaMA-Factory Kernels 系统用于管理不同硬件设备提供的高性能计算内核（kernel）实现，该系统通过替换模型中的关键模块（如 RMSNorm、SwiGLU、RoPE、MoE 等）为硬件优化的版本，从而显著提升模型训练和推理的性能。

Kernels 系统采用基于注册表的自动发现机制，能够根据当前运行环境自动检测可用的硬件设备（NPU、CUDA 等），并使能相应的高性能 kernels。这种设计使得用户无需关心底层实现细节，只需简单调用接口即可获得性能提升。

## 核心特性

- **自动注册机制**：基于 `@register_kernel` 装饰器实现自动注册系统。系统启动时会自动扫描 `ops` 目录下的 kernel 实现，并将其注册到全局注册表中。

- **设备适配感知**：自动检测当前硬件设备（NPU、CUDA 等）并应用相应的优化。系统会跳过不支持的设备，确保在不同环境下都能正常工作。

- **模块化设计**：每个 kernel 独立实现，互不干扰。可以单独应用某个 kernel，也可以批量应用所有默认的 kernels。

- **后向兼容**：kernel 替换不修改模型权重，保持数值一致性。优化后的实现与原始实现保持精度一致（在浮点误差范围内）。

- **灵活扩展**：通过继承 `BaseKernel` 基类并使用装饰器，可以轻松添加新的 kernel 实现，支持新的硬件设备或优化算法。

## 使用方式

### 1. 通过训练 YAML 配置文件使用

要在训练过程中使能 kernels，只需在配置文件中增加如下配置，即可自动使能所有默认可用的 kernels：

```yaml
...
kernel_config:
    name: auto
    include_kernels: auto # choice: null/true/false/auto/kernel_id1,kernel_id2,kernel_id3, default is null

...
```

### 2. 调用 API 使能

#### 2.1 apply_default_kernels 使能所有默认 kernels

`apply_default_kernels` API 能够自动应用当前设备上所有默认注册的 kernels：

```python
from transformers import AutoModelForCausalLM
from llamafactory.v1.plugins.model_plugins.kernels import apply_default_kernels

# 加载模型
model = AutoModelForCausalLM.from_pretrained("qwen/qwen2.5-0.5B")

# 自动应用所有默认 kernels
model = apply_default_kernels(model, include_kernels="auto")
```

#### 2.2 apply_kernel 使能特定 kernel

如果需要更精细的控制，例如在某些场合单独应用某个 kernel，可以手动调用 `apply_kernel` 函数并传入 kernel ID：

```python
from transformers import AutoModelForCausalLM
from llamafactory.v1.plugins.model_plugins.kernels import apply_kernel

# 加载模型
model = AutoModelForCausalLM.from_pretrained("qwen/qwen2.5-0.5B")

# 手动应用各个 kernels
# 注意：kernel ID 必须与定义时的 _kernel_id 一致
model = apply_kernel("npu_fused_rope", model=model)
model = apply_kernel("npu_fused_rmsnorm", model=model)
model = apply_kernel("npu_fused_swiglu", model=model)
model = apply_kernel("npu_fused_moe", model=model)

### 3. 查询已注册的可用 kernels

可以通过 `get_default_kernels` 获取当前环境中所有已注册且可用的默认 kernel ID：

```python
from llamafactory.v1.plugins.model_plugins.kernels import get_default_kernels

# 获取默认 kernel 列表
available_kernels = get_default_kernels()
print(f"Available kernels: {available_kernels}")
# 输出示例: ['npu_fused_rmsnorm', 'npu_fused_swiglu', 'npu_fused_rope', 'npu_fused_moe']
```

### 当前已实现的 kernels

| Kernel ID | 功能 | 支持的设备 | 备注 |
|-----------|------|-----------|------|
| [npu_fused_rmsnorm](./fused-operators.md/#npufusedrmsnorm) | RMSNorm 融合算子 | NPU | NPU 设备的高性能 RMSNorm 实现 |
| [npu_fused_swiglu](./fused-operators.md/#npufusedswiglu) | SwiGLU 融合算子 | NPU | NPU 设备的高性能 SwiGLU 实现 |
| [npu_fused_rope](./fused-operators.md/#npufusedrope) | RoPE 融合算子 | NPU | NPU 设备的高性能 RoPE 实现 |
| [npu_fused_moe](./fused-operators.md/#npufusedmoe) | MoE 融合算子 | NPU | MoE 融合算子，适配 Qwen3-MoE 等模型 |

我们会持续适配更多的 kernels，如果您需要自己开发新的 kernels，请参考我们的 [Kernel 开发文档](../../dev-guide/plugins/model-plugins/kernels.md)，欢迎您向 LLaMA-Factory 贡献代码。
