import torch
from abc import ABC


class DistributedBatchMemory(ABC):
    def __init__(self, dataset):
        """
        初始化 DistributedBatchMemory

        Args:
            dataset: 支持两种格式：
                1) list[Dict[str, torch.Tensor]]: 每个元素是一个样本的所有属性
                2) Dict[str, torch.Tensor]: 每个key对应一个属性，tensor的shape是batch_size
        """
        if isinstance(dataset, list):
            # 格式1: list[Dict[str, torch.Tensor]] - 直接存储
            self.dataset = dataset
        elif isinstance(dataset, dict):
            # 格式2: Dict[str, torch.Tensor] - 需要转换
            self.dataset = self._convert_dict_to_list(dataset)
        else:
            raise TypeError(f"dataset must be list or dict, got {type(dataset)}")

    def _convert_dict_to_list(self, dict_dataset):
        """
        将 Dict[str, torch.Tensor] 转换为 list[Dict[str, torch.Tensor]]

        Args:
            dict_dataset: Dict[str, torch.Tensor] 格式的数据

        Returns:
            list[Dict[str, torch.Tensor]] 格式的数据
        """
        if not dict_dataset:
            return []

        # 获取第一个tensor的长度作为batch_size
        first_tensor = next(iter(dict_dataset.values()))
        if not isinstance(first_tensor, torch.Tensor):
            raise TypeError(f"All values in dict_dataset must be torch.Tensor, got {type(first_tensor)}")

        batch_size = first_tensor.shape[0]

        # 检查所有tensor的batch_size是否一致
        for key, tensor in dict_dataset.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"All values in dict_dataset must be torch.Tensor, got {type(tensor)} for key '{key}'")
            if tensor.shape[0] != batch_size:
                raise ValueError(
                    f"All tensors must have same batch_size, got {tensor.shape[0]} for key '{key}' vs {batch_size}")

        # 转换为list格式
        converted_dataset = []
        for i in range(batch_size):
            sample = {}
            for key, tensor in dict_dataset.items():
                sample[key] = tensor[i:i + 1]  # 保持tensor维度为1
            converted_dataset.append(sample)

        return converted_dataset

    def split(self, dp_size: int) -> list:
        batches = []
        if self.dataset is not None:
            total = len(self.dataset)
            part_size = (total + dp_size - 1) // dp_size  # ceil division
            for i in range(dp_size):
                start = i * part_size
                end = min((i + 1) * part_size, total)
                sub_dataset = self.dataset[start:end]
                batch = DistributedBatchMemory(sub_dataset)
                batches.append(batch)
        else:
            raise ValueError("cannot split empty dataset")

        return batches

    def merge(self, batch):
        merged_dataset = (self.dataset or []) + (batch.dataset or [])
        if not merged_dataset:
            merged_dataset = None
        new_batch = DistributedBatchMemory(merged_dataset)
        return new_batch

    def __getstate__(self):
        state = {
            'dataset': self.dataset,
        }

        return state

    def __setstate__(self, state):
        self.dataset = state['dataset']

    def __getitem__(self, key):
        # list[Dict[str, torch.Tensor]]
        if isinstance(key, int):
            return self.dataset[key]
        elif isinstance(key, str):
            # 返回所有item[key] cat后的tensor
            tensor_list = [item[key] for item in self.dataset]
            return torch.cat(tensor_list, dim=0)
        else:
            raise TypeError(f"Key must be int or str, got {type(key)}")

    def __setitem__(self, key, value):
        """
        如果 key 是 int，value 应为 Dict[str, Tensor]，直接赋值。
        如果 key 是 str，value 应为 Tensor，shape[0] == batchsize，
        按 batchsize 拆分后分别写入 self.dataset[i][key]。
        """
        if isinstance(key, int):
            if not isinstance(value, dict):
                raise ValueError("When key is int, value must be a dict of str->Tensor.")
            self.dataset[key] = value
        elif isinstance(key, str):
            if not isinstance(value, torch.Tensor):
                raise ValueError("When key is str, value must be a torch.Tensor.")
            batchsize = len(self.dataset)
            if value.shape[0] != batchsize:
                raise ValueError(
                    f"Tensor batchsize mismatch: value.shape[0]={value.shape[0]}, dataset batchsize={batchsize}")
            for i in range(batchsize):
                self.dataset[i][key] = value[i:i + 1]  # 保持shape为(1,...)
        else:
            raise TypeError(f"Key must be int or str, got {type(key)}")

    def __delitem__(self, key):
        """
        If key is int, delete self.dataset[key].
        If key is str, 删除 self.dataset 所有元素的该属性。
        """
        if isinstance(key, int):
            del self.dataset[key]
        elif isinstance(key, str):
            for item in self.dataset:
                if key in item:
                    del item[key]
        else:
            raise TypeError(f"Key must be int or str, got {type(key)}")
