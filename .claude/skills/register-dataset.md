# register-dataset

注册新的 HuggingFace 数据集到 geo_edit 项目中。

## 使用方法

```
/register-dataset <huggingface_dataset_path>
```

例如：
```
/register-dataset MapEval/MapEval-Visual
```

## 执行步骤

### 1. 探索数据集结构

运行 Python 脚本获取数据集字段信息：

```python
from datasets import load_dataset
ds = load_dataset("<dataset_path>")
print(ds)  # 查看 splits
for split in ds.keys():
    print(f"Split: {split}")
    print(ds[split].features)  # 查看字段
print(ds[list(ds.keys())[0]][0])  # 查看第一条样本
```

检查 HuggingFace 仓库是否包含图片文件：

```python
from huggingface_hub import list_repo_files
files = list(list_repo_files("<dataset_path>", repo_type="dataset"))
print(files[:30])
```

确定以下关键字段：
- `id_key`: 用作唯一标识的字段（如果没有需要在预处理时添加）
- `image_key`: 图片字段名（如果图片需要单独下载，记录路径字段）
- `answer_key`: 答案字段名
- `question/prompt 字段`: 问题或提示文本字段
- 其他相关字段（如 options、classification、type 等）

**重要**：查看数据集的官方文档或示例代码，确认：
- 选项编号格式（0-indexed 还是 1-indexed）
- answer 字段的含义（是否有特殊值如 0 表示无答案）

### 2. 在 input_template.py 添加提示词模板

文件: `geo_edit/datasets/input_template.py`

根据数据集的任务类型，添加两个模板：
- `<DATASET_NAME>_INPUT_TEMPLATE` - 工具模式
- `<DATASET_NAME>_NOTOOL_INPUT_TEMPLATE` - 无工具模式

模板示例：
```python
<DATASET_NAME>_INPUT_TEMPLATE = """\
<任务描述和指导>

{question}
{options_text}

If you need to analyze the image in detail, you can use the available tools.

Please provide your final answer in <answer></answer> tags.
"""

<DATASET_NAME>_NOTOOL_INPUT_TEMPLATE = """\
<任务描述和指导>

{question}
{options_text}

Please analyze carefully and provide your final answer in <answer></answer> tags.
"""
```

### 3. 在 task_registry.py 注册数据集

文件: `geo_edit/datasets/task_registry.py`

1. 导入新模板：
```python
from geo_edit.datasets.input_template import (
    ...,
    <DATASET_NAME>_INPUT_TEMPLATE,
    <DATASET_NAME>_NOTOOL_INPUT_TEMPLATE,
)
```

2. 如果需要，添加辅助函数（如格式化选项）：
```python
def _format_<dataset_name>_options(item: Mapping[str, Any]) -> str:
    """Format options for <dataset_name> dataset."""
    options = item.get("options", [])
    if not options:
        return ""
    # 注意：根据数据集的 answer 格式选择正确的起始索引
    # 如果 answer=0 表示无答案，选项从 1 开始：enumerate(options, start=1)
    # 如果 answer 直接对应选项索引，从 0 开始：enumerate(options)
    option_lines = [f"{i}. {opt}" for i, opt in enumerate(options, start=1)]
    return "\n".join(option_lines)
```

3. 在 `DATASET_SPECS` 字典中添加新条目：
```python
"<dataset_name>": DatasetSpec(
    name="<dataset_name>",
    id_key="<id字段>",
    answer_key="<答案字段>",
    image_key="<图片字段>",
    prompt_template=<DATASET_NAME>_INPUT_TEMPLATE,
    notool_prompt_template=<DATASET_NAME>_NOTOOL_INPUT_TEMPLATE,
    template_fields={
        "question": "<问题字段>",
        "options_text": _format_<dataset_name>_options,  # 或直接字段名
    },
    task_kwargs_fields={
        "meta_info_extra": lambda item: {
            # 需要传递给任务的额外元信息（如 classification, type 等）
        },
    },
),
```

### 4. 创建数据预处理脚本（如需要）

如果数据集图片已内嵌且只需添加 id 字段，可以简化：

```python
"""Package <dataset_name> dataset to parquet format."""
from __future__ import annotations

import argparse
from pathlib import Path
from datasets import load_dataset


def package_<dataset_name>(out_dir: Path, split: str = "test") -> Path:
    out_parquet = out_dir / f"<dataset_name>_{split}_dataset.parquet"

    ds = load_dataset("<huggingface_path>")[split]

    # 添加 id 列（如果原数据集没有）
    ds = ds.add_column("id", list(range(len(ds))))

    ds.to_parquet(str(out_parquet))
    return out_parquet
```

如果图片需要单独下载，参考 `package_mapeval_visual.py`。

### 5. 创建评估脚本

文件: `geo_edit/evaluation/eval_<dataset_name>.py`

参考 `eval_mapeval_visual.py` 或 `eval_chartqa.py` 创建评估脚本。

**重要注意事项**：
1. 字段位置：`classification`、`type` 等字段可能直接在 record 顶层，也可能在 `meta_info_extra` 中，需要兼容两种情况：
   ```python
   classification = record.get("classification") or record.get("meta_info_extra", {}).get("classification", "unknown")
   ```

2. 答案提取：`output_text` 可能没有 `<answer>` 标签，需要 fallback：
   ```python
   extracted = extract_answer(output_str)
   predicted = parse_answer(extracted if extracted else output_str)
   ```

3. 使用 `iter_meta_info_files` 遍历每个子目录的 `meta_info.jsonl`，而不是读取 `global_meta_info.jsonl`

### 6. 验证注册

```bash
python -c "from geo_edit.datasets.task_registry import get_dataset_spec; print(get_dataset_spec('<dataset_name>'))"
```

### 7. 清理本地缓存

探索完成后，删除下载的 HuggingFace 数据集缓存：

```bash
rm -rf ~/.cache/huggingface/hub/datasets--<org>--<dataset_name>
```

## 常见问题避坑

### 选项编号格式
- **问题**：不同数据集选项编号方式不同
- **解决**：查看官方文档确认 answer 字段含义
  - MapEval-Visual: 选项从 1 开始，answer=0 表示无答案
  - 其他数据集可能从 0 开始

### 评估脚本字段读取
- **问题**：`classification`、`type` 等字段位置不固定
- **解决**：同时检查顶层和 `meta_info_extra`：
  ```python
  field = record.get("field") or record.get("meta_info_extra", {}).get("field", "unknown")
  ```

### 答案解析
- **问题**：模型输出可能没有 `<answer>` 标签
- **解决**：先尝试从标签提取，失败则解析原文：
  ```python
  extracted = extract_answer(text)
  result = parse(extracted if extracted else text)
  ```

### 数据集无 id 字段
- **问题**：原数据集没有 id 字段
- **解决**：在预处理脚本中添加：
  ```python
  ds = ds.add_column("id", list(range(len(ds))))
  ```

## 关键文件

- `geo_edit/datasets/task_registry.py` - 数据集注册
- `geo_edit/datasets/input_template.py` - 提示词模板
- `geo_edit/data_preprocess/` - 数据预处理脚本
- `geo_edit/evaluation/` - 评估脚本
- `geo_edit/utils/io_utils.py` - `iter_meta_info_files`, `load_records` 工具函数
