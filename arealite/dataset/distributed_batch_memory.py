import torch
from abc import ABC


class DistributedBatchMemory(ABC):
    def __init__(self, dataset):
        self.dataset = dataset  # list[Dict[str, torch.Tensor]]

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
            # Return a list of tensors for this key
            return [item[key] for item in self.dataset]
        else:
            raise TypeError(f"Key must be int or str, got {type(key)}")

    def __setitem__(self, key, value):
        """
        If key is int, set self.dataset[key] = value.
        If key is str, value 应为 list，批量设置 self.dataset[i][key] = value[i]。
        """
        if isinstance(key, int):
            self.dataset[key] = value
        elif isinstance(key, str):
            if not isinstance(value, list) or len(value) != len(self.dataset):
                raise ValueError("Value must be a list with the same length as dataset.")
            for i, v in enumerate(value):
                self.dataset[i][key] = v
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
