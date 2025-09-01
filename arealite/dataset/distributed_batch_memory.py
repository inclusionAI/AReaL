import torch

from realhf.base.datapack import ffd_allocate


class DistributedBatchMemory:
    def __init__(self, dataset):
        if isinstance(dataset, list):
            self.dataset = self._convert_list_to_dict(dataset)
        elif isinstance(dataset, dict):
            self._validate_dict_dataset(dataset)
            self.dataset = dataset
        else:
            raise TypeError(f"dataset must be list or dict, got {type(dataset)}")

    def _validate_dict_dataset(self, dataset):
        """验证 Dict[str, Tensor] 格式的完整性"""
        if not dataset:
            return
        batch_size = next(iter(dataset.values())).shape[0]
        for key, tensor in dataset.items():
            if tensor.shape[0] != batch_size:
                raise ValueError(
                    f"Batch size mismatch for key '{key}': expected {batch_size}, got {tensor.shape[0]}"
                )

    def _convert_list_to_dict(self, list_dataset):
        """将 list[Dict] 转换为 Dict[str, Tensor]"""
        if not list_dataset:
            return {}
        keys = list_dataset[0].keys()
        dict_dataset = {}
        for key in keys:
            tensors = [sample[key] for sample in list_dataset]
            dict_dataset[key] = torch.cat(tensors, dim=0)
        return dict_dataset

    def split(self, dp_size: int) -> list:
        """分割数据集"""
        if not self.dataset:
            raise ValueError("Cannot split empty dataset")

        batch_size = next(iter(self.dataset.values())).shape[0]
        part_size = (batch_size + dp_size - 1) // dp_size  # 向上取整
        batches = []
        for i in range(dp_size):
            start = i * part_size
            end = min(start + part_size, batch_size)
            split_data = {k: v[start:end].clone() for k, v in self.dataset.items()}
            batches.append(DistributedBatchMemory(split_data))
        return batches

    def split_by_groups(self, group_size: int, n: int) -> list:
        """
        split data into n parts, each parts containing group_size continuous elements
        examples:
        tensor = [0,1,...,510,511], n=32， group_size = 8
        index=0 => [0,1,...,6,7,256,257,...,262,263]
        index=1 => [8,9,...,14,15, 264,265,...,270,271]
        index=31 => [248,249,...,254,255,504,505,...,510,511]

        returns:
            List[DistributedBatchMemory]
        """
        total = next(iter(self.dataset.values())).shape[0]
        assert total % group_size == 0, "tensor length must be devided by group_size"
        total_chunks = total // group_size
        assert total_chunks % n == 0, "chunk size must be devided by n"

        k = total_chunks // n  # num of groups of each part
        batches = []
        for i in range(n):
            indices = []
            for j in range(k):
                chunk_idx = i + j * n
                start = chunk_idx * group_size
                end = start + group_size
                indices.extend(range(start, end))

            split_data = {k: v[indices] for k, v in self.dataset.items()}
            batches.append(DistributedBatchMemory(split_data))

        return batches

    def split_by_groups_1(self, group_size: int, n: int) -> list:
        total = next(iter(self.dataset.values())).shape[0]
        assert total % n == 0, "tensor length must be devided by n"
        # 512/4 = 128
        count_of_each_part = total // n
        # fixme
        assert (
            count_of_each_part % (group_size * n) == 0
        ), "count of each part must be devided by (n * group_size)"
        # 128/4 = 32
        count_of_each_part_of_each_n = count_of_each_part // n
        indexes = [[] for _ in range(n)]
        for part in range(n):
            # [0, 128, 256, 384]
            start_index = count_of_each_part * part
            inner_indices = []
            for i in range(n):
                # [0, 128, 256, 384] + range(32..64..96..128)
                tmp_inner_indices = list(
                    range(
                        count_of_each_part_of_each_n * i,
                        count_of_each_part_of_each_n * (i + 1),
                    )
                )
                inner_indices.append(
                    list(map(lambda item: item + start_index, tmp_inner_indices))
                )
                indexes[i].extend(inner_indices[i])

        batches = []
        print(f"indexes: {indexes}")
        for part in range(n):
            split_data = {k: v[indexes[part]] for k, v in self.dataset.items()}
            batches.append(DistributedBatchMemory(split_data))

        return batches

    def split_by_groups_2(self, group_size: int, n: int) -> list:
        total = next(iter(self.dataset.values())).shape[0]
        assert total % n == 0, "tensor length must be devided by n"
        # 512/4
        count_of_each_part = total // n
        # fixme
        assert (
            count_of_each_part % (group_size * n) == 0
        ), "count of each part must be devided by (n * group_size)"
        indexes = [[] for _ in range(n)]
        # 128/8 = 16
        group_count_of_each_part = count_of_each_part // group_size
        print(f"group_count_of_each_part: {group_count_of_each_part}")
        # group = 16, n=4
        for group_num in range(group_count_of_each_part):
            for i in range(n):
                tmp_index = list(
                    range(
                        i * count_of_each_part + group_num * group_size,
                        i * count_of_each_part + (group_num + 1) * group_size,
                    )
                )
                indexes[group_num // n].extend(tmp_index)

        batches = []
        print(f"indexes: {indexes}")
        for part in range(n):
            split_data = {k: v[indexes[part]] for k, v in self.dataset.items()}
            batches.append(DistributedBatchMemory(split_data))

        return batches

    def split_by_seqlen_ffd_2(self, group_size: int, n: int) -> list:
        # 按推理组先将数据按顺序切成n份
        batches = self.split_by_groups_2(group_size, n)
        result = DistributedBatchMemory({})
        for batch in batches:
            result = result.merge(batch)
        return result._split_by_seqlen_ffd_helper(group_size, n)

    def split_by_seqlen_ffd_1(self, group_size: int, n: int) -> list:
        # 按推理组先将数据按顺序切成n份
        batches = self.split_by_groups_1(group_size, n)

        results = [DistributedBatchMemory({}) for _ in range(n)]
        for i, batch in enumerate(batches):
            # 将推理组内的数据均分成n份
            datas = batch._split_by_seqlen_ffd_helper(group_size, n)
            for j, data in enumerate(datas):
                results[j] = results[j].merge(data)

        return results

    def split_by_seqlen_ffd(self, group_size: int, n: int) -> list:
        # 按推理组先将数据按顺序切成n份
        batches = self.split(n)

        results = [DistributedBatchMemory({}) for _ in range(n)]
        for i, batch in enumerate(batches):
            # 将推理组内的数据均分成n份
            datas = batch._split_by_seqlen_ffd_helper(group_size, n)
            for j, data in enumerate(datas):
                results[j] = results[j].merge(data)

        return results

    def _split_by_seqlen_ffd_helper(self, group_size: int, n: int) -> list:
        total = next(iter(self.dataset.values())).shape[0]
        assert total % group_size == 0, "tensor length must be devided by group_size"

        seqlen = self.dataset["seqlen"]
        # 转二维，每行 group_size 个元素
        reshaped = seqlen.view(-1, group_size)
        # 对每行求和
        # [10, 40, 60, 43, 90, 133,45, 65]
        group_total_lens = reshaped.sum(dim=1)
        print(f"group_total_lens: {group_total_lens}")
        # 返回indexes
        # [[0,4],[1,5],[2,6],[3,7]]
        unsorted_group_rebalanced_indexs = ffd_allocate(
            group_total_lens.tolist(), int(1e12), n
        )
        group_rebalanced_indexs = sorted(
            [sorted(g) for g in unsorted_group_rebalanced_indexs]
        )
        print(f"group_rebalanced_indexs: {group_rebalanced_indexs}")
        batches = []
        for i in range(n):
            indexes = []
            # print(f"group_rebalanced_indexs[i]: {group_rebalanced_indexs[i]}")
            for group_index in group_rebalanced_indexs[i]:
                tmp_indexs = list(
                    range(
                        group_size * group_index, group_size * group_index + group_size
                    )
                )
                # print(f"tmp_indexs: {tmp_indexs}")
                indexes.extend(tmp_indexs)
            # print(f"indexes: {indexes}")
            split_data = {k: v[indexes] for k, v in self.dataset.items()}
            batches.append(DistributedBatchMemory(split_data))
        return batches

    def merge(self, other):
        """合并另一个批次的数据"""
        merged_data = {k: v for k, v in self.dataset.items()}
        for k, v in other.dataset.items():
            if k in merged_data:
                merged_data[k] = torch.cat([merged_data[k], v], dim=0)
            else:
                merged_data[k] = v
        return DistributedBatchMemory(merged_data)

    def __getstate__(self):
        return {"dataset": self.dataset}

    def __setstate__(self, state):
        self.dataset = state["dataset"]

    def __getitem__(self, key):
        if isinstance(key, int):
            return {k: v[key] for k, v in self.dataset.items()}
        elif isinstance(key, str):
            return self.dataset[key]
        else:
            raise TypeError("Key must be int or str")

    def __setitem__(self, key, value):
        """支持两种赋值方式：
        - str键: 更新整个属性张量
        - int索引: 需要将数据转换为列表格式后更新（效率较低，建议避免）
        """
        if isinstance(key, str):
            # 更新整个属性张量
            if not isinstance(value, torch.Tensor):
                raise ValueError("值必须为torch.Tensor类型")
            if self.dataset:
                expected_batch_size = next(iter(self.dataset.values())).shape[0]
                if value.shape[0] != expected_batch_size:
                    raise ValueError(
                        f"张量的批处理大小不匹配。期望{expected_batch_size}, 实际{value.shape[0]}"
                    )
            self.dataset[key] = value
        else:
            raise TypeError("键必须为str类型以更新属性张量")

    def __delitem__(self, key):
        """支持两种删除方式：
        - int索引: 删除指定位置的样本
        - str键: 删除整个属性
        """
        if isinstance(key, int):
            # 转换为列表格式进行删除
            list_dataset = self._convert_dict_to_list(self.dataset)
            del list_dataset[key]
            self.dataset = self._convert_list_to_dict(list_dataset)
        elif isinstance(key, str):
            # 直接删除整个属性
            if key in self.dataset:
                del self.dataset[key]
        else:
            raise TypeError(f"键必须为int或str类型, 实际类型为{type(key)}")

    def _convert_dict_to_list(self, dict_dataset):
        """将字典格式的数据集转换为列表格式（用于按索引删除）"""
        if not dict_dataset:
            return []
        batch_size = next(iter(dict_dataset.values())).shape[0]
        list_dataset = []
        for i in range(batch_size):
            sample = {k: v[i] for k, v in dict_dataset.items()}
            list_dataset.append(sample)
        return list_dataset

    def __str__(self):
        if not self.dataset:
            return "DistributedBatchMemory<empty>"

        batch_size = next(iter(self.dataset.values())).shape[0]
        keys = list(self.dataset.keys())
        shapes = {k: v.shape for k, v in self.dataset.items()}
        return f"DistributedBatchMemory<batch_size={batch_size}, keys={keys}, shapes={shapes}, values={self.dataset.items()}>"

    def __repr__(self):
        return self.__str__()


if __name__ == "__main__":
    # 示例：创建一个长度为 512 的张量
    x = torch.arange(512)
    input_ = {"input": x}
    data = DistributedBatchMemory(input_)
    batches = data.split_by_groups(group_size=8, n=32)

    print(batches)
    # 输出: tensor([  0,   1,   2,   3,   4,   5,   6,   7, 256, 257, 258, 259, 260, 261, 262, 263]) ....
