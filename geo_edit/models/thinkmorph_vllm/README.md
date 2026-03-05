# ThinkMorph vLLM Inference

简化的ThinkMorph推理包装器，使用vLLM后端实现高效的图文交错生成。

## 特性

- ✅ **vLLM加速**: 使用vLLM的原生Bagel支持，显著提升文本生成速度
- ✅ **简单API**: 单个类完成所有功能，3行代码即可开始推理
- ✅ **图文交错**: 支持论文中的thinking过程，包含文本推理和视觉思考
- ✅ **批量处理**: 原生支持HuggingFace datasets，自动checkpoint和恢复
- ✅ **灵活配置**: 多个预设配置（快速/高质量/推理heavy等）

## 安装

```bash
# 安装依赖
pip install vllm transformers datasets pillow torch torchvision

# 添加模块到Python路径（在AReaL项目中）
# 或者直接在代码中：
import sys
sys.path.append("C:/Users/Antoine/code/AReaL/geo_edit/models")
```

## 快速开始

### 单样本推理

```python
from thinkmorph_vllm import VLLMInterleavedInference
from PIL import Image

# 初始化
inferencer = VLLMInterleavedInference(
    model_path="ThinkMorph/ThinkMorph-7B",
    tensor_parallel_size=1  # GPU数量
)

# 推理
image = Image.open("image.jpg")
outputs = inferencer.infer_single(
    image=image,
    text="What is in this image?",
    think=True  # 启用thinking模式
)

# 查看结果
for output in outputs:
    if isinstance(output, str):
        print(output)  # 文本输出
    else:
        output.show()  # 图像输出
```

### 批量处理Dataset

```python
from datasets import load_dataset

# 加载数据集
dataset = load_dataset("your-dataset", split="test")

# 批量推理
results = inferencer.infer_dataset(
    dataset=dataset,
    output_dir="./results",
    image_field="image",
    text_field="question",
    think=True,
    resume=True  # 支持断点续传
)

print(f"Processed {results['processed']} samples")
```

## 配置选项

### 预设配置

```python
from thinkmorph_vllm.configs import (
    DEFAULT_CONFIG,      # 平衡配置
    FAST_CONFIG,         # 快速推理（25步）
    HIGH_QUALITY_CONFIG, # 高质量（100步）
    REASONING_CONFIG,    # 推理heavy（更多token）
    EDITING_CONFIG,      # 图像编辑优化
)

# 使用预设
inferencer = VLLMInterleavedInference(
    model_path="ThinkMorph/ThinkMorph-7B",
    **FAST_CONFIG
)
```

### 自定义配置

```python
inferencer = VLLMInterleavedInference(
    model_path="ThinkMorph/ThinkMorph-7B",
    # vLLM设置
    tensor_parallel_size=2,  # 多GPU
    dtype="bfloat16",
    max_model_len=32768,
    # 推理配置
    max_think_tokens=4096,
    text_temperature=0.3,
    cfg_text_scale=4.0,      # 文本条件强度
    cfg_img_scale=2.0,       # 图像条件强度
    num_timesteps=50,        # 去噪步数
    cfg_renorm_type="text_channel",
)
```

## 参数说明

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_think_tokens` | 4096 | 思考过程的最大token数 |
| `text_temperature` | 0.3 | 文本生成温度（0-1） |
| `cfg_text_scale` | 4.0 | 文本CFG强度（4.0-8.0推荐） |
| `cfg_img_scale` | 2.0 | 图像CFG强度（1.0-2.0推荐） |
| `num_timesteps` | 50 | 图像生成的去噪步数 |
| `cfg_renorm_type` | "text_channel" | CFG归一化类型 |

### CFG归一化类型

- `"global"`: 全局归一化（T2I默认，编辑任务推荐）
- `"channel"`: 按通道归一化
- `"text_channel"`: 仅对文本条件归一化（可能导致模糊）

**提示**: 如果生成的图像模糊，尝试使用`"global"`或降低`cfg_renorm_min`。

## 使用模式

### 1. 纯文本理解（VQA）

```python
outputs = inferencer.infer_single(
    image=image,
    text="What is this?",
    think=True,
    understanding_output=True  # 只输出文本
)
```

### 2. 图文交错生成

```python
outputs = inferencer.infer_single(
    image=image,
    text="Analyze this image step by step",
    think=True,
    understanding_output=False  # 生成文本+图像
)
```

### 3. 视觉推理任务

```python
# 例如：拼图问题
puzzle_image = Image.open("jigsaw.jpg")
outputs = inferencer.infer_single(
    image=puzzle_image,
    text="这是一个2x2的拼图，部分被打乱。请分析正确的排列顺序。",
    think=True
)
```

## 文件结构

```
thinkmorph_vllm/
├── __init__.py          # 包入口，导出主类
├── inference.py         # 核心推理类（~600行）
├── configs.py           # 配置预设
├── example_usage.py     # 使用示例
└── README.md            # 本文档
```

## API参考

### VLLMInterleavedInference

**初始化参数**:
- `model_path` (str): 模型路径或HuggingFace ID
- `tensor_parallel_size` (int): 张量并行度（GPU数量）
- `dtype` (str): 数据类型，默认"bfloat16"
- `max_model_len` (int): 最大上下文长度
- 其他推理配置参数（见上文）

**方法**:
- `infer_single(image, text, think, understanding_output)`: 单样本推理
- `infer_dataset(dataset, output_dir, ...)`: 批量处理

### 输出格式

- **单样本**: 返回`List[Union[str, PIL.Image]]`，交错的文本和图像
- **批量**: 保存到`output_dir/`:
  - `{sample_id}_result.json`: 结果元数据
  - `{sample_id}_image_{idx}.png`: 生成的图像
  - `checkpoint.json`: 断点文件

## 注意事项

1. **图像生成**: 当前版本的图像生成部分返回占位符，完整实现需要访问Bagel的VAE和diffusion流程
2. **内存**: 确保有足够的GPU内存（推荐40GB+）
3. **模型下载**: 首次运行会从HuggingFace下载模型（~14GB）

## 故障排除

### OOM (Out of Memory)
- 减少`tensor_parallel_size`使用更多GPU
- 降低`max_model_len`
- 使用`FAST_CONFIG`减少生成步数

### 生成图像模糊
- 使用`cfg_renorm_type="global"`
- 降低`cfg_renorm_min`
- 降低`cfg_text_scale`

### 推理慢
- 使用`FAST_CONFIG`
- 增加`tensor_parallel_size`
- 降低`num_timesteps`

## 贡献

基于ByteDance的ThinkMorph项目，使用Apache 2.0许可证。

## 参考

- [ThinkMorph Paper](https://arxiv.org/abs/2510.27492)
- [ThinkMorph GitHub](https://github.com/ThinkMorph/ThinkMorph)
- [vLLM Documentation](https://docs.vllm.ai/)
