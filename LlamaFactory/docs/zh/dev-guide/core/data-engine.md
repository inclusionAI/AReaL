# DataEngine

## 1. DataEngine 简介


`DataEngine` 是 LLaMA-Factory v1 数据处理的核心类，继承自 PyTorch 的 `Dataset`，负责各种插件的接入，其他功能（如数据格式转换、数据加载等）均通过插件的形式实现并接入 `DataEngine`。

`DataEngine`接受一个唯一入参：`DataArguments` 实例，所有的元数据集信息均通过该参数配置传入。

## 2. DataEngine 与 DataArguments 接口定义

```python
@dataclass
class DataArguments:
    """ `DataEngine`初始化入参

    args:
        dataset (str): 数据集路径，远程数据集 repo id / dataset_info.yaml 路径，或本地数据集路径/dataset_info.yaml路径
        cutoff_len (int): 数据集截止长度，即数据集最大样本采样数量
    """
    ...


class DataEngine(Dataset):
    """数据引擎（DataEngine）

    `DataEngine` 负责数据集的加载与统一管理，支持：
        - 从本地路径或 Hugging Face Hub 加载数据
        - 通过插件机制加载自定义数据
        - 构建统一的数据索引
        - 支持流式（streaming）与非流式数据访问

    attr:
        args (DataArguments): 数据参数配置
        datasets (dict[str, HFDataset]): 数据集名称到数据对象的映射
        dataset_infos (dict[str, DatasetInfo]): 数据集名称到元信息的映射
        data_index (list[tuple[str, int]]): 数据索引列表，每项为 (dataset_name, sample_index)
        streaming (bool): 是否为流式数据集
    """

    def __init__(self, data_args: DataArguments) -> None:
        """初始化 `DataEngine`

        初始化时自动执行以下步骤：
            1. 调用 `get_dataset_info`， 从 `data_args` 读取并解析数据集元信息
            2. 调用 `load_dataset`，根据配置加载数据集
            3. 调用 `build_data_index`，构建统一的索引列表

        args:
            data_args (DataArguments): 数据参数配置对象
        """
        ...

    def get_dataset_info(self) -> None:
        """从配置文件或远程仓库加载数据集元信息

        根据 `self.args.dataset` 确定数据源，数据源支持如下选项：
            - 本地 YAML 配置文件路径
            - Hugging Face Hub 上的 YAML 配置文件路径
            - 本地数据集文件路径
            - Hugging Face Hub 数据集 repo id

        """
        ...

    def load_dataset(self) -> None:
        """根据数据集元信息加载所有数据集

        每个数据集条目可以包含以下字段：
            - `hf_hub_url`: 使用 `datasets.load_dataset` 加载
            - 本地数据文件：通过 `DataLoaderPlugin` 插件加载
            - `streaming`: 是否启用流式模式

        更新:
            self.datasets (dict): 数据集名称到已加载数据对象的映射
            self.streaming (bool): 如果任一数据集为流式模式，则设置为 True
        """
        ...

    def build_data_index(self) -> None:
        """构建统一的数据索引

        为所有数据集创建全局索引列表 `(dataset_name, sample_index)`

        当启用流式模式时，生成固定长度（例如 1000）的占位索引；
        否则，为每条样本建立索引。

        插件 `DataIndexPlugin` 可根据数据集大小或权重调整索引分布
        """
        ...

    def _convert_data_sample(self, raw_sample: dict[str, Any], dataset_name: str) -> Sample:
        """将原始样本转换为统一格式

        根据 `dataset_info` 中的 `converter` 字段，调用对应的转换插件，
        将原始样本标准化为统一的数据结构。

        args:
            raw_sample (dict[str, Any]): 原始数据样本
            dataset_name (str): 样本所属的数据集名称

        return:
            Sample: 转换后的标准化格式样本
        """
        ...

    def __len__(self) -> int:
        """返回数据集的总样本数

        return:
            int: 数据集长度
                如果为流式数据集，返回 `-1`
        """
        ...

    def __getitem__(self, index: Union[int, Any]) -> Union[Sample, list[Sample]]:
        """根据索引或选择器获取样本

        args:
            index (Union[int, Any]): 数据索引，int 或 list[int]

        return:
            Union[Sample, list[Sample]]: 单个样本或样本列表
        """
        ...

    def __iter__(self) -> Iterable:
        """返回数据集迭代器

        用于非流式数据集的顺序或随机访问
        流式模式下需要实现异步加载逻辑

        return:
            Iterable: 数据集迭代器。
        """
        ...

    async def __aiter__(self) -> AsyncIterable:
        """返回异步数据集迭代器

        用于流式数据集或异步数据加载场景
        允许在异步环境中以流的方式读取样本

        return:
            AsyncIterable: 异步迭代器，按顺序产出样本
        """
        ...


```

`DataArguments`  参数说明：

`dataset`: 数据集路径，支持本地或远程，当传入本地数据集文件路径时，需要满足该数据集为标准格式；否则需要传入 `dataset_info.yaml` 来配置数据集的 `converter` 等元信息，以告知 `DataEngine` 应当如何处理该数据。

`cutoff_len`: 数据集的截止长度，即该数据集的最大样本数量。

---

## 3. DataEngine 核心方法

### 3.1 `get_dataset_info`：加载数据元信息

根据 `dataset` 参数加载数据集配置，获取数据位置、数据格式、插件配置等所有数据元信息，在实例化 `DataEngine` 时会自动调用此方法。

### 3.2 加载数据集：`load_dataset`

遍历所有数据源，根据不同的数据源加载数据，在实例化 `DataEngine` 时会自动调用此方法。

```python
for key, value in self.dataset_infos.items():
    split = value.get("split", "train")
    streaming = value.get("streaming", False)

    if "hf_hub_url" in value:
        # 从 HF Hub 加载
        dataset = load_dataset(value["hf_hub_url"], split=split, streaming=streaming)
    else:
        # 使用 DataLoaderPlugin 加载本地文件
        dataset = DataLoaderPlugin(args=self.args).auto_load_data(value)

    self.datasets[key] = dataset
```

### 3.3 `build_data_index`：构建数据索引

为每个数据集创建索引列表 `[(dataset_name, sample_index), ...]`, `DataIndexPlugin`插件在此处被调用，可控制各数据集的采样频率、采样方式等，在实例化`DataEngine`时会自动调用此方法。

```python
for dataset_name, dataset in self.datasets.items():
    # 创建基础索引
    data_index = [(dataset_name, idx) for idx in range(len(dataset))]

    # 根据 size 和 weight 调整索引
    size = self.dataset_infos[dataset_name].get("size")
    weight = self.dataset_infos[dataset_name].get("weight")
    if size or weight:
        data_index = DataIndexPlugin().adjust_data_index(data_index, size, weight)

    self.data_index.extend(data_index)
```

### 3.4 `_convert_data_sample`：数据格式标准化

将原始数据转换为标准格式，`DataConverterPlugin`插件在此处被调用，具体调用的插件由 `get_dataset_info` 方法获取的 `converter` 信息指定，若 `converter` 为空则假定数据集为标准格式，此方法由`DataEngine`的 `__getitem__` 方法调用。

```python
def _convert_data_sample(self, raw_sample: dict, dataset_name: str) -> Sample:
    converter = self.dataset_infos[dataset_name].get("converter")
    if converter is not None:
        # 使用指定的转换器
        from ..plugins.data_plugins.converter import get_converter
        return {"_dataset_name": dataset_name, **get_converter(converter)(raw_sample)}
    else:
        # 已经是标准格式
        return {"_dataset_name": dataset_name, **raw_sample}
```

---

## 4. 初始化

`DataEngine` 初始化过程只需传入一个构建好的 `DataArguments` 即可，后续可通过该 `DataEngine` 访问数据集中的数据。

```python
from llamafactory.v1.config.data_args import DataArguments
from llamafactory.v1.core.data_engine import DataEngine

# 1. 创建数据参数
data_args = DataArguments(
    dataset="~/data/v1_sft_demo.jsonl",
    cutoff_len=2048
)

# 2. 初始化 Data Engine
data_engine = DataEngine(data_args=data_args)

# 3. 访问数据
sample = data_engine[0]  # 获取第一个样本
```

## 5. 数据访问方式

实例化后的`DataEngine`支持整数索引、列表索引、以及切片等访问方式，其数据读取用法可等价于 Python 列表。

```python
sample = data_engine[0]  # 获取第一个样本

sample = data_engine[0:10]  # 获取前 10 个样本

sample = data_engine[[0, 5, 10]]  # 获取指定索引的样本

```
