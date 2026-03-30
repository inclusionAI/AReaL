# Data Plugins

## 1. Data Plugins 简介

## DataConverterPlugin

### 1. DataConverterPlugin 简介

DataConverter 负责将非标准格式的数据集转换为 v1 的标准 Messages 格式。这使得用户可以继续使用现有的数据集（如 Alpaca 格式），而无需手动转换。针对自定义格式的数据集，用户也可以通过构建对应的自定义 DataConverter 插件，来负责其数据格式标准化。

当前，LLaMA-Factory 已内置了 `Alpaca Converter` 和 `Pair Converter`，这两类数据集可以直接使用对应的 converter 进行标准化，无需自定义转换器。


### 2. Alpaca Converter 详解

#### 2.1 Alpaca 格式

Alpaca 格式是一种常见的指令微调数据格式：

```json
{
  "system": "You are a helpful assistant.",
  "instruction": "Describe a process of making crepes.",
  "input": "",
  "output": "Making crepes is an easy and delicious process..."
}
```

#### 2.2 Alpaca Converter 接口定义

```python
class AlpacaSample(TypedDict, total=False):
    """Alpaca 格式数据样本结构

    attr:
        system (str, 可选): 系统提示信息（system prompt），用于设定对话背景或模型行为。
        instruction (str, 可选): 用户指令（user instruction），通常为任务描述。
        input (str, 可选): 额外的输入内容（input text），可与 instruction 拼接。
        output (str, 可选): 模型生成的目标输出（expected response）。
    """
    ...


def alpaca_converter(raw_sample: AlpacaSample) -> SFTSample:
    """将 Alpaca 样本转换为 SFT（Supervised Fine-Tuning）标准样本格式

    `alpaca_converter` 将 Alpaca 数据集中一条样本转换为通用的 `SFTSample` 格式
    该格式用于监督微调（SFT）或多轮对话建模

    转换逻辑:
        - 若存在 `system` 字段，则生成一条系统消息，loss_weight = 0.0
        - 若存在 `instruction` 或 `input` 字段，则合并为一条用户消息，loss_weight = 0.0
        - 若存在 `output` 字段，则生成一条助手机器人回复消息，loss_weight = 1.0

    args:
        raw_sample (AlpacaSample): 原始 Alpaca 数据样本

    return:
        SFTSample: 转换后的标准化样本，格式如下:

            {
                "messages": [
                    {"role": "system", "content": [{"type": "text", "value": "..."}], "loss_weight": 0.0},
                    {"role": "user", "content": [{"type": "text", "value": "..."}], "loss_weight": 0.0},
                    {"role": "assistant", "content": [{"type": "text", "value": "..."}], "loss_weight": 1.0},
                ]
            }

    example:
        >>> raw = {"instruction": "请将以下句子翻译成英文：", "input": "你好", "output": "Hello"}
        >>> alpaca_converter(raw)
        {
            "messages": [
                {"role": "user", "content": [{"type": "text", "value": "请将以下句子翻译成英文：你好"}], "loss_weight": 0.0},
                {"role": "assistant", "content": [{"type": "text", "value": "Hello"}], "loss_weight": 1.0}
            ]
        }
    """

```

#### 2.3 转换过程

`alpaca_converter` 函数将 Alpaca 格式转换为标准格式，转换逻辑如下：

```python
def alpaca_converter(raw_sample: AlpacaSample) -> SFTSample:
    messages = []

    # 1. 添加系统提示词（如果存在）
    if "system" in raw_sample:
        messages.append({
            "role": "system",
            "content": [{"type": "text", "value": raw_sample["system"]}],
            "loss_weight": 0.0
        })

    # 2. 添加用户输入（instruction + input）
    if "instruction" in raw_sample or "input" in raw_sample:
        user_content = raw_sample.get("instruction", "") + raw_sample.get("input", "")
        messages.append({
            "role": "user",
            "content": [{"type": "text", "value": user_content}],
            "loss_weight": 0.0
        })

    # 3. 添加模型回复
    if "output" in raw_sample:
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "value": raw_sample["output"]}],
            "loss_weight": 1.0
        })

    return {"messages": messages}
```

#### 2.4 转换示例

**输入（Alpaca 格式）：**

```json
{
  "instruction": "What is the capital of France?",
  "input": "",
  "output": "The capital of France is Paris."
}
```

**输出（标准格式）：**

```json
{
  "messages": [
    {
      "role": "user",
      "content": [{"type": "text", "value": "What is the capital of France?"}],
      "loss_weight": 0.0
    },
    {
      "role": "assistant",
      "content": [{"type": "text", "value": "The capital of France is Paris."}],
      "loss_weight": 1.0
    }
  ]
}
```

### 3. 自定义转换器

#### 3.1 创建自定义转换器

如果用户有自己的数据格式，可以轻松添加自定义转换器将其标准化，实现过程可参考如下示例：

```python
# src/llamafactory/v1/plugins/data_plugins/converter.py

from typing import TypedDict, NotRequired
from ...extras.types import SFTSample

# 1. 定义输入格式的类型
class MyCustomSample(TypedDict, total=False):
    question: str
    answer: str
    context: NotRequired[str]

# 2. 实现转换逻辑
def custom_converter(raw_sample: MyCustomSample) -> SFTSample:
    messages = []

    # 构建用户消息
    user_text = raw_sample["question"]
    if "context" in raw_sample:
        user_text = f"Context: {raw_sample['context']}\n\nQuestion: {user_text}"

    messages.append({
        "role": "user",
        "content": [{"type": "text", "value": user_text}],
        "loss_weight": 0.0
    })

    # 构建助手消息
    messages.append({
        "role": "assistant",
        "content": [{"type": "text", "value": raw_sample["answer"]}],
        "loss_weight": 1.0
    })

    return {"messages": messages}

# 3. 注册 custom_converter
#src/llamafactory/v1/plugins/data_plugins/converter.py: CONVERTERS
CONVERTERS = {
    "alpaca": alpaca_converter,
    "custom": custom_converter,  # 添加自定义转换器
}
```

#### 3.2 使用自定义转换器

在 YAML 配置中指定转换器名称：

```yaml
my_dataset:
  file_name: custom_data.json
  converter: custom
```

---

## DataLoaderPlugin

### 1. DataLoaderPlugin 简介

`DataLoaderPlugin` 负责从本地文件加载数据集，当前支持如下文件格式：

- **JSON**: `.json`
- **JSONL**: `.jsonl`
- **CSV**: `.csv`
- **Parquet**: `.parquet`
- **Arrow**: `.arrow`
- **Text**: `.txt`

### 2. DataLoaderPlugin 接口定义

```python
@dataclass
class DataLoaderPlugin:
    """数据加载插件（DataLoaderPlugin）

    负责根据数据集信息（`DatasetInfo`）自动加载本地或远程数据集。  
    支持多种文件格式（如 CSV、JSON、Parquet、Text、Arrow），并可选择是否以流式方式加载。

    通常由 `DataEngine` 调用，用于统一封装数据加载逻辑。
    """

    args: DataArguments
    """数据参数对象，包含数据目录、缓存路径、分片等配置信息。"""


    def _get_builder_name(self, path: str) -> Literal["arrow", "csv", "json", "parquet", "text"]:
        """获取数据集文件格式

        根据输入文件路径自动判断应使用的 HuggingFace `load_dataset` 构建器类型。
        通过文件扩展名推断数据类型，例如 `.csv`、`.jsonl`、`.parquet`、`.txt` 等。

        args:
            path (str): 数据集文件路径，用于识别文件类型。

        return:
            Literal["arrow", "csv", "json", "parquet", "text"]:
                数据构建器名称，用于 `datasets.load_dataset()`。

        example:
            >>> _get_builder_name("data/train.jsonl")
            "json"
        """
        ...


    def auto_load_data(self, dataset_info: DatasetInfo) -> HFDataset:
        """根据传入的 `dataset_info` 自动选择合适的加载方式

        args:
            dataset_info (DatasetInfo): 数据集元信息，通常包含：
                - `file_name`: 数据文件路径
                - `split`: 数据划分（如 "train"、"test"）；
                - `streaming`: 是否启用流式加载

        return:
            HFDataset: 加载完成的 Hugging Face 数据集对象。

        example:
            >>> plugin = DataLoaderPlugin(args)
            >>> ds = plugin.auto_load_data({"file_name": "~/data.json", "split": "train"})
        """
        ...


    def load_data_from_file(self, filepath: str, split: str, streaming: bool) -> HFDataset:
        """从文件或目录加载数据集

        根据输入路径自动识别文件类型（CSV、JSON、Parquet、Text 等），  
        并通过 `datasets.load_dataset()` 加载数据集。  
        若 `streaming=True`，则将结果转换为迭代式数据集。

        args:
            filepath (str): 文件路径或目录路径。
            split (str): 数据划分名称（如 "train"、"validation"）。
            streaming (bool): 是否启用流式加载模式。

        return:
            HFDataset: 加载后的数据集对象。

        example:
            >>> plugin.load_data_from_file("data/train.json", "train", False)
        """
        ...

```

---

## DataIndexPlugin

### 1. DataIndexPlugin 简介

`DataIndexPlugin` 负责调整数据索引，支持通过配置 `size`, `weight` 等参数控制数据集样本数量和采样频率。

- 使用 `size` 参数 限制使用的样本数量：

```yaml
my_dataset:
  file_name: large_dataset.json
  size: 1000  # 只使用前 1000 个样本
```

- 使用 `weight` 参数调整数据集在混合数据中的采样频率：

```yaml
dataset_a:
  file_name: data_a.json
  weight: 1.0

dataset_b:
  file_name: data_b.json
  weight: 2.0  # dataset_b 的样本出现频率是 dataset_a 的 2 倍
```

**说明**：`weight` 参数适用于在多个数据集混合训练时，调整不同数据集的的采样频率

- 当 `weight=1.0` 时，数据集按原始比例采样
- 当 `weight=2.0` 时，该数据集的索引会复制 2 倍，使其样本出现频率翻倍

### 2. DataIndexPlugin 接口定义

```python
@dataclass
class DataIndexPlugin:
    """数据索引插件（DataIndexPlugin）

    根据 `size` 和 `weight` 调整数据索引列表，控制数据集的样本数量和采样频率  
    通常在多数据集混合训练时使用，以控制不同数据集在总体样本中的占比。

    在 `DataEngine.build_data_index` 中被自动调用，用于实现样本重采样或加权分布。
    """

    def adjust_data_index(
        self, data_index: list[tuple[str, int]], size: Optional[int], weight: Optional[float]
    ) -> list[tuple[str, int]]:
        """调整数据索引列表

        根据 `size` 或 `weight` 参数对输入的数据索引进行采样、扩展或缩减。  
        若两个参数同时存在，将依次执行基于大小和基于权重的调整。

        args:
            data_index (list[tuple[str, int]]):  
                数据索引列表，每个元素为 `(dataset_name, sample_index)`。  
            size (Optional[int]):  
                目标样本数量，若指定则根据该数量裁剪或重复样本。  
            weight (Optional[float]):  
                数据集权重，用于控制数据集在混合训练中的采样比例。

        return:
            list[tuple[str, int]]:  
                调整后的数据索引列表。

        example:
            >>> plugin = DataIndexPlugin()
            >>> adjusted = plugin.adjust_data_index([("ds1", i) for i in range(100)], size=50, weight=None)
            >>> len(adjusted)
            50
        """
        ...


    def adjust_by_size(self, data_index: list[tuple[str, int]], size: int) -> list[tuple[str, int]]:
        """根据目标大小调整数据索引

        通过裁剪或重复样本，使索引总数等于 `size`。  
        常用于统一不同数据集的样本数量。

        args:
            data_index (list[tuple[str, int]]):  
                原始数据索引列表。  
            size (int):  
                目标样本数量。

        return:
            list[tuple[str, int]]:  
                调整后长度等于 `size` 的数据索引列表。

        example:
            >>> plugin.adjust_by_size([("ds1", i) for i in range(10)], 20)
        """
        ...


    def adjust_by_weight(self, data_index: list[tuple[str, int]], weight: float) -> list[tuple[str, int]]:
        """根据权重调整数据索引

        通过加权采样或重复样本，使数据集样本出现频率符合指定权重。  
        常用于多数据源训练中按比例平衡样本。

        args:
            data_index (list[tuple[str, int]]):  
                原始数据索引列表。  
            weight (float):  
                数据集权重（相对比例，可与其他数据集共同归一化）。

        return:
            list[tuple[str, int]]:  
                调整后的加权数据索引列表。

        example:
            >>> plugin.adjust_by_weight([("ds1", i) for i in range(10)], 0.5)
        """
        ...

```
---

## DataSelectorPlugin

### 1. DataSelectorPlugin 简介

`DataSelectorPlugin` 为 `DataEngine`提供基于索引访问数据的功能，由 `DataEngine` 的 `__getitem__` 方法自动调用。


### 2. DataSelectorPlugin 接口定义

```python
@dataclass
class DataSelectorPlugin:
    """根据索引选择数据集样本。

    配合 `DataEngine` 使用，通过统一的 `data_index` 结构（包含数据集名与样本索引）来实现灵活的数据选择

    """

    data_index: list[tuple[str, int]]
    """数据索引列表，每个元素为 (dataset_name, sample_index)。"""


    def select(self, index: Union[slice, list[int], Any]) -> Union[tuple[str, int], list[tuple[str, int]]]:
        """选择数据集样本

        根据输入类型从 `data_index` 中选择对应的样本索引  
        支持三种索引方式：
            - 切片（slice）：返回对应范围内的样本
            - 索引列表（list[int]）：返回指定索引处的多个样本
            - 其他类型输入将触发异常。

        args:
            index (Union[slice, list[int], Any]): 数据样本索引
                可以是切片（`slice`）或索引列表

        return:
            Union[tuple[str, int], list[tuple[str, int]]]:
                - 若为单个索引：返回一个 `(dataset_name, sample_index)`
                - 若为多个索引或切片：返回多个样本的列表

        except:
        Raises:
            ValueError: 当输入索引类型不受支持时抛出。
        ...
```
