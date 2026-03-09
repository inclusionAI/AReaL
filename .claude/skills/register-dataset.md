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
- `id_key`: 用作唯一标识的字段
- `image_key`: 图片字段名（如果图片需要单独下载，记录路径字段）
- `answer_key`: 答案字段名
- `question/prompt 字段`: 问题或提示文本字段
- 其他相关字段（如 options、classification 等）

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
    option_lines = [f"{i}. {opt}" for i, opt in enumerate(options)]
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
            # 需要传递给任务的额外元信息
        },
    },
),
```

### 4. 创建数据预处理脚本（如需要）

如果数据集的图片需要单独下载，创建 `geo_edit/data_preprocess/package_<dataset_name>.py`:

```python
"""Package <dataset_name> dataset to parquet format."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasets import Dataset, Features, Image as HFImage, Sequence, Value, load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm


def package_<dataset_name>(out_dir: Path) -> Path:
    out_parquet = out_dir / "<dataset_name>_dataset.parquet"

    ds = load_dataset("<huggingface_path>")["<split>"]
    examples = []

    for i, item in enumerate(tqdm(ds, desc="Processing")):
        # 下载图片（如需要）
        image_path = hf_hub_download(
            repo_id="<huggingface_path>",
            filename=item["<image_path_field>"],
            repo_type="dataset",
        )
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        examples.append({
            "id": i,
            # ... 其他字段
            "image": {"bytes": image_bytes, "path": None},
        })

    features = Features({
        "id": Value("int64"),
        # ... 定义所有字段类型
        "image": HFImage(),
    })

    dataset = Dataset.from_list(examples, features=features)
    dataset.to_parquet(str(out_parquet))
    return out_parquet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir or ".").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    package_<dataset_name>(out_dir)


if __name__ == "__main__":
    main()
```

### 5. 验证注册

```bash
python -c "from geo_edit.datasets.task_registry import get_dataset_spec; print(get_dataset_spec('<dataset_name>'))"
```

### 6. 清理本地缓存

探索完成后，删除下载的 HuggingFace 数据集缓存：

```bash
rm -rf ~/.cache/huggingface/hub/datasets--<org>--<dataset_name>
```

例如：
```bash
rm -rf ~/.cache/huggingface/hub/datasets--MapEval--MapEval-Visual
```

## 关键文件

- `geo_edit/datasets/task_registry.py` - 数据集注册
- `geo_edit/datasets/input_template.py` - 提示词模板
- `geo_edit/data_preprocess/` - 数据预处理脚本
