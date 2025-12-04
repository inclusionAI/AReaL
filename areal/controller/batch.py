from __future__ import annotations

import asyncio
import uuid
from typing import Any

import torch
from torch import Tensor

from areal.api.controller_api import DistributedBatch
from areal.controller.batch_client import BatchDataClient
from areal.controller.batch_metadata import (
    BatchMetadata,
    ScalarMetadata,
    ShardMetadata,
    TensorMetadata,
)
from areal.utils.batch_utils import (
    convert_dict_to_list,
    convert_list_to_dict,
    validate_dict_dataset,
)
from areal.utils.data import concat_padded_tensors
from areal.utils.datapack import ffd_allocate
from areal.utils.errors import FrameworkError


class DistributedBatchMemory(DistributedBatch):
    """Distributed batch memory with metadata-driven data access.

    This class separates metadata (data shape, location) from actual data.
    The control plane only passes metadata, and actual data is fetched on-demand
    via HTTP from distributed nodes.

    Attributes
    ----------
    dataset : dict[str, torch.Tensor | Any] | None
        The actual data (lazy-loaded, None until get_data() is called)
    metadata : BatchMetadata | None
        Metadata describing the distributed batch
    _is_local : bool
        Whether this batch has local data (not distributed)
    """

    dataset = None
    metadata: BatchMetadata | None = None
    _is_local: bool = True  # True if data is stored locally (not distributed)

    # Shared client for fetching data
    _client: BatchDataClient | None = None

    @classmethod
    def _calculate_total_batch_size_from_shards(
        cls, shards: list[ShardMetadata]
    ) -> int:
        """Calculate total_batch_size from shards, handling different keys.

        When shards have different keys, simple summation may be incorrect.
        This method uses a heuristic to determine the correct total batch size.

        Parameters
        ----------
        shards : list[ShardMetadata]
            List of shard metadata

        Returns
        -------
        int
            Total batch size

        Notes
        -----
        Strategy:
        1. If all shards have the same keys: sum all batch_sizes
        2. If shards have different keys:
           - Prefer shards with "primary" fields (input_ids, attention_mask)
           - If no primary fields, use the maximum batch_size across all shards
           - Issue a warning about potential inaccuracy
        """
        if not shards:
            return 0

        # Check if all shards have the same keys
        all_keys_sets = [set(shard.fields.keys()) for shard in shards]
        same_keys = all(keys == all_keys_sets[0] for keys in all_keys_sets)

        if same_keys:
            # All shards have the same keys, simple summation is correct
            return sum(shard.batch_size for shard in shards)

        # Different keys: use heuristic
        from areal.utils import logging

        logger = logging.getLogger(__name__)

        # Primary fields that typically represent the main data
        primary_fields = {"input_ids", "attention_mask", "pixel_values"}

        # Find shards with primary fields
        primary_shards = [
            shard
            for shard in shards
            if any(field in shard.fields for field in primary_fields)
        ]

        if primary_shards:
            # Use shards with primary fields
            total = sum(shard.batch_size for shard in primary_shards)
            logger.warning(
                f"Shards have different keys. Calculating total_batch_size ({total}) "
                f"based on shards with primary fields {primary_fields}. "
                f"Total shards: {len(shards)}, primary shards: {len(primary_shards)}"
            )
            return total
        else:
            # No primary fields found, use maximum batch_size
            # This assumes shards may represent the same samples with different attributes
            total = max(shard.batch_size for shard in shards)
            logger.warning(
                f"Shards have different keys and no primary fields found. "
                f"Using maximum batch_size ({total}) across {len(shards)} shards. "
                f"This may not be accurate if shards represent different sample sets."
            )
            return total

    @classmethod
    def get_client(cls) -> BatchDataClient:
        """Get or create the shared batch data client.

        Returns
        -------
        BatchDataClient
            Shared client instance
        """
        if cls._client is None:
            cls._client = BatchDataClient()
        return cls._client

    @classmethod
    def from_dict(cls, dict_dataset: dict[str, Tensor | Any]):
        """Create a DistributedBatchMemory from dictionary format dataset.

        This creates a local batch (not distributed) with data stored in memory.

        Parameters
        ----------
        dict_dataset : Dict[str, Union[Tensor, Any]]
            Dictionary format dataset, where values can be Tensor, scalar, or list types

        Returns
        -------
        DistributedBatchMemory
            New DistributedBatchMemory instance
        """
        validate_dict_dataset(dict_dataset)
        instance = cls.__new__(cls)
        instance.dataset = dict_dataset
        instance.metadata = None
        instance._is_local = True
        return instance

    @classmethod
    def from_metadata(cls, metadata: BatchMetadata) -> DistributedBatchMemory:
        """Create a DistributedBatchMemory from metadata (without actual data).

        The data will be fetched lazily when get_data() is called.

        Parameters
        ----------
        metadata : BatchMetadata
            Metadata describing the distributed batch

        Returns
        -------
        DistributedBatchMemory
            New DistributedBatchMemory instance with metadata only
        """
        instance = cls.__new__(cls)
        instance.dataset = None
        instance.metadata = metadata
        instance._is_local = False
        return instance

    @classmethod
    def create_metadata_for_local_data(
        cls,
        dict_dataset: dict[str, Tensor | Any],
        node_id: str,
        node_addr: str,
        global_step: int = 0,
    ) -> BatchMetadata:
        """Create metadata for local data.

        This is a helper to create metadata that describes locally stored data,
        which can be useful when transitioning from local to distributed storage.

        Parameters
        ----------
        dict_dataset : dict[str, Tensor | Any]
            Local dataset
        node_id : str
            Identifier of the current node
        node_addr : str
            Network address (host:port) of the current node
        global_step : int, optional
            Global training step, by default 0

        Returns
        -------
        BatchMetadata
            Metadata describing the local data
        """
        # Get batch size
        batch_size = 0
        if dict_dataset:
            first_value = next(iter(dict_dataset.values()))
            if isinstance(first_value, torch.Tensor):
                batch_size = first_value.shape[0]
            elif isinstance(first_value, list):
                batch_size = len(first_value)
            else:
                batch_size = 1

        # Create field metadata
        fields = {}
        for key, value in dict_dataset.items():
            if isinstance(value, torch.Tensor):
                fields[key] = TensorMetadata(
                    shape=tuple(value.shape),
                    dtype=str(value.dtype),
                    device=str(value.device),
                )
            elif isinstance(value, list):
                fields[key] = ScalarMetadata(value_type="list", length=len(value))
            else:
                fields[key] = ScalarMetadata(value_type=type(value).__name__, length=1)

        # Create shard metadata
        shard_id = str(uuid.uuid4())
        shard = ShardMetadata(
            node_id=node_id,
            node_addr=node_addr,
            shard_id=shard_id,
            batch_size=batch_size,
            fields=fields,
        )

        # Create batch metadata
        batch_id = str(uuid.uuid4())
        metadata = BatchMetadata(
            batch_id=batch_id,
            global_step=global_step,
            total_batch_size=batch_size,
            shards=[shard],
        )

        return metadata

    @classmethod
    def from_list(cls, list_dataset: list[dict[str, Tensor | Any]]):
        """Create a DistributedBatchMemory from list format dataset.

        Parameters
        ----------
        list_dataset : List[Dict[str, Union[Tensor, Any]]]
            List format dataset

        Returns
        -------
        DistributedBatchMemory
            New DistributedBatchMemory instance
        """
        dict_dataset = convert_list_to_dict(list_dataset)
        return cls.from_dict(dict_dataset)

    def chunk(self, dp_size: int) -> list[DistributedBatchMemory]:
        """Split the dataset across data parallel processes.

        This function preserves the original order of data, ensuring that
        the sequence of samples in the concatenated result matches the
        original dataset order.

        Supports both metadata mode and local data mode.
        """
        # Metadata mode: split shards across dp_size groups
        if self.metadata is not None:
            return self._chunk_metadata(dp_size)

        # Local data mode: split actual data
        if not self.dataset:
            raise FrameworkError(
                "FrameworkError",
                "DistributedBatchMemoryError",
                "Cannot split empty dataset",
            )

        total = self._get_total_size()
        part_size = (total + dp_size - 1) // dp_size
        batches = []
        for i in range(dp_size):
            start = i * part_size
            end = min(start + part_size, total)
            split_data = {}
            for k, v in self.dataset.items():
                if isinstance(v, torch.Tensor):
                    split_data[k] = v[start:end].clone()
                elif isinstance(v, list):
                    split_data[k] = v[start:end]
                else:
                    # For scalar values, keep as-is
                    split_data[k] = v
            batch = self.__class__.__new__(self.__class__)
            batch.dataset = split_data
            batch.metadata = None
            batch._is_local = True
            batches.append(batch)
        return batches

    def _chunk_metadata(self, dp_size: int) -> list[DistributedBatchMemory]:
        """Split metadata across data parallel processes.

        与本地 ``chunk`` 语义一致: 在全局样本维度上按顺序平均划分为 ``dp_size`` 份。

        和只按 shard 维度贪心分配不同, 这里允许 **一个物理 shard 被拆成多个
        逻辑子 shard**。子 shard 通过 ``offset`` 和 ``batch_size`` 表示其在原始
        shard 中的子区间, 从而实现真正意义上的"按 batch_size 均分"。

        注意：当 shards 包含不同 fields 时，我们假设具有相同 batch_size 和
        sample_offset 的 shards 对应同一批样本的不同字段，应该被分到同一个 group。
        """
        if self.metadata is None:
            raise FrameworkError(
                "FrameworkError",
                "DistributedBatchMemoryError",
                "No metadata to split",
            )

        total = self.metadata.total_batch_size
        if dp_size <= 0:
            raise FrameworkError(
                "FrameworkError",
                "DistributedBatchMemoryError",
                "dp_size must be positive",
            )

        # 检查是否所有 shards 有相同的 fields
        all_fields_sets = [set(shard.fields.keys()) for shard in self.metadata.shards]
        same_fields = all(fields == all_fields_sets[0] for fields in all_fields_sets)

        if not same_fields:
            # Different fields: use field-aware chunking
            from areal.utils import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                "Shards have different fields, using field-aware chunking. "
                "This may group shards representing the same samples together."
            )
            return self._chunk_metadata_with_different_fields(dp_size)

        # Same fields: use original logic
        # 每个 group 目标样本数, 与本地 chunk 保持一致
        part_size = (total + dp_size - 1) // dp_size

        groups: list[list[ShardMetadata]] = [[] for _ in range(dp_size)]
        group_sizes = [0] * dp_size

        group_idx = 0
        remaining_groups = dp_size
        used_in_group = 0

        for shard in self.metadata.shards:
            # 起始 offset 可能非 0 (例如来源于上一次拆分)
            base_offset = shard.offset
            remaining_in_shard = shard.batch_size

            # 当前 shard 可能被拆到多个 group 中
            while remaining_in_shard > 0 and group_idx < dp_size:
                # 若只剩最后一个 group, 把所有剩余样本都放进去
                if remaining_groups == 1:
                    take = remaining_in_shard
                else:
                    space = part_size - used_in_group
                    if space <= 0:
                        # 当前 group 已满, 切换到下一个 group, 并重新估算后续目标大小
                        remaining_groups -= 1
                        group_idx += 1
                        used_in_group = 0
                        if group_idx >= dp_size:
                            break
                        left_samples = total - sum(group_sizes[:group_idx])
                        part_size = (
                            left_samples + remaining_groups - 1
                        ) // remaining_groups
                        continue
                    take = min(space, remaining_in_shard)

                if take <= 0:
                    break

                # 为当前 group 创建一个子 shard
                logical_offset = base_offset + (shard.batch_size - remaining_in_shard)
                sub_shard = ShardMetadata(
                    node_id=shard.node_id,
                    node_addr=shard.node_addr,
                    shard_id=shard.shard_id,
                    batch_size=take,
                    offset=logical_offset,
                    fields=shard.fields,
                )
                groups[group_idx].append(sub_shard)
                group_sizes[group_idx] += take

                remaining_in_shard -= take
                used_in_group += take

                # 当前 group 达到目标大小, 切换到下一个 group
                if (
                    remaining_in_shard > 0
                    and used_in_group >= part_size
                    and remaining_groups > 1
                ):
                    remaining_groups -= 1
                    group_idx += 1
                    used_in_group = 0
                    if group_idx >= dp_size:
                        break
                    left_samples = total - sum(group_sizes[:group_idx])
                    part_size = (
                        left_samples + remaining_groups - 1
                    ) // remaining_groups

            if group_idx >= dp_size:
                break

        # 构造带 metadata 的子 batch
        batches: list[DistributedBatchMemory] = []
        for i in range(dp_size):
            new_metadata = BatchMetadata(
                batch_id=f"{self.metadata.batch_id}_chunk_{i}",
                global_step=self.metadata.global_step,
                total_batch_size=group_sizes[i],
                shards=groups[i],
            )
            batch = self.__class__.__new__(self.__class__)
            batch.dataset = None
            batch.metadata = new_metadata
            batch._is_local = False
            batches.append(batch)

        return batches

    def _chunk_metadata_with_different_fields(
        self, dp_size: int
    ) -> list[DistributedBatchMemory]:
        """Split metadata when shards have different fields.

        策略：将 shards 分为 primary 和 additional 两类。
        - Primary shards: 包含 input_ids/attention_mask 等主要字段
        - Additional shards: 只包含额外字段（如 prox_logp）

        只对 primary shards 进行分配，然后为每个 group 添加对应的 additional shards。
        """
        from areal.utils import logging

        logger = logging.getLogger(__name__)

        # 定义主要字段
        primary_fields = {"input_ids", "attention_mask", "pixel_values"}

        # 分离 primary 和 additional shards
        primary_shards = []
        additional_shards = []

        for shard in self.metadata.shards:
            if any(field in shard.fields for field in primary_fields):
                primary_shards.append(shard)
            else:
                additional_shards.append(shard)

        logger.info(
            f"Field-aware chunking: {len(primary_shards)} primary shards, "
            f"{len(additional_shards)} additional shards"
        )

        if not primary_shards:
            # 没有 primary shards，回退到普通分配
            logger.warning("No primary shards found, using all shards for chunking")
            primary_shards = self.metadata.shards
            additional_shards = []

        # 计算基于 primary shards 的总样本数
        total = sum(shard.batch_size for shard in primary_shards)
        part_size = (total + dp_size - 1) // dp_size

        # 对 primary shards 进行分配
        groups: list[list[ShardMetadata]] = [[] for _ in range(dp_size)]
        group_sizes = [0] * dp_size
        group_sample_ranges = [
            (0, 0) for _ in range(dp_size)
        ]  # (start, end) sample indices

        group_idx = 0
        remaining_groups = dp_size
        used_in_group = 0
        current_sample_idx = 0  # 追踪当前样本索引

        for shard in primary_shards:
            base_offset = shard.offset
            remaining_in_shard = shard.batch_size

            # 当前 shard 可能被拆到多个 group 中
            while remaining_in_shard > 0 and group_idx < dp_size:
                # 若只剩最后一个 group, 把所有剩余样本都放进去
                if remaining_groups == 1:
                    take = remaining_in_shard
                else:
                    space = part_size - used_in_group
                    if space <= 0:
                        # 当前 group 已满, 切换到下一个 group
                        remaining_groups -= 1
                        group_idx += 1
                        used_in_group = 0
                        if group_idx >= dp_size:
                            break
                        left_samples = total - sum(group_sizes[:group_idx])
                        part_size = (
                            left_samples + remaining_groups - 1
                        ) // remaining_groups
                        continue
                    take = min(space, remaining_in_shard)

                if take <= 0:
                    break

                # 为当前 group 创建一个子 shard
                logical_offset = base_offset + (shard.batch_size - remaining_in_shard)
                sub_shard = ShardMetadata(
                    node_id=shard.node_id,
                    node_addr=shard.node_addr,
                    shard_id=shard.shard_id,
                    batch_size=take,
                    offset=logical_offset,
                    fields=shard.fields,
                )
                groups[group_idx].append(sub_shard)

                # Update group sample range
                if group_sizes[group_idx] == 0:
                    group_sample_ranges[group_idx] = (
                        current_sample_idx,
                        current_sample_idx + take,
                    )
                else:
                    start, _ = group_sample_ranges[group_idx]
                    group_sample_ranges[group_idx] = (start, current_sample_idx + take)

                group_sizes[group_idx] += take
                current_sample_idx += take

                remaining_in_shard -= take
                used_in_group += take

                # 当前 group 达到目标大小, 切换到下一个 group
                if (
                    remaining_in_shard > 0
                    and used_in_group >= part_size
                    and remaining_groups > 1
                ):
                    remaining_groups -= 1
                    group_idx += 1
                    used_in_group = 0
                    if group_idx >= dp_size:
                        break
                    left_samples = total - sum(group_sizes[:group_idx])
                    part_size = (
                        left_samples + remaining_groups - 1
                    ) // remaining_groups

            if group_idx >= dp_size:
                break

        # 为每个 group 添加对应的 additional shards
        # 需要根据样本范围来分配，确保严格保序
        # 计算 additional shards 的样本范围（按顺序累积）
        additional_sample_ranges = []
        current_add_sample_idx = 0
        for add_shard in additional_shards:
            add_start = current_add_sample_idx
            add_end = current_add_sample_idx + add_shard.batch_size
            additional_sample_ranges.append((add_start, add_end, add_shard))
            current_add_sample_idx = add_end

        # 为每个 group 分配对应的 additional shards
        for group_idx in range(dp_size):
            group_start, group_end = group_sample_ranges[group_idx]

            for add_start, add_end, add_shard in additional_sample_ranges:
                # 检查 additional shard 是否与当前 group 的样本范围重叠
                overlap_start = max(group_start, add_start)
                overlap_end = min(group_end, add_end)

                if overlap_start < overlap_end:
                    # 有重叠，需要切片 additional shard
                    # 计算在 original shard 中的 offset 和 length
                    offset_in_shard = overlap_start - add_start
                    length = overlap_end - overlap_start

                    sub_shard = ShardMetadata(
                        node_id=add_shard.node_id,
                        node_addr=add_shard.node_addr,
                        shard_id=add_shard.shard_id,
                        batch_size=length,
                        offset=add_shard.offset + offset_in_shard,
                        fields=add_shard.fields,
                    )
                    groups[group_idx].append(sub_shard)
                    logger.debug(
                        f"Added additional shard slice to group {group_idx}: "
                        f"shard_id={add_shard.shard_id}, "
                        f"batch_size={length}, offset={sub_shard.offset}, "
                        f"covers samples [{overlap_start}, {overlap_end})"
                    )

        # 构造带 metadata 的子 batch
        batches: list[DistributedBatchMemory] = []
        for i in range(dp_size):
            new_metadata = BatchMetadata(
                batch_id=f"{self.metadata.batch_id}_chunk_{i}",
                global_step=self.metadata.global_step,
                total_batch_size=group_sizes[i],  # 只计算 primary shards 的 batch_size
                shards=groups[i],
            )
            batch = self.__class__.__new__(self.__class__)
            batch.dataset = None
            batch.metadata = new_metadata
            batch._is_local = False
            batches.append(batch)

        return batches

    def chunk_by_ffd(
        self, group_size: int, dp_size: int
    ) -> list[DistributedBatchMemory]:
        """Split data by sequence length using First Fit Decreasing algorithm

        Parameters
        ----------
        group_size : int
            Size of each group
        dp_size : int
            Number of data parallel processes

        Returns
        -------
        list[DistributedBatchMemory]
            List of DistributedBatchMemory objects

        Notes
        -----
        For metadata mode, this method will fall back to simple chunking
        since we cannot determine sequence lengths without fetching the data.
        """
        # Metadata mode: fall back to simple chunking
        # FFD requires sequence length information which is not available in metadata yet, it will be implemented in the future
        if self.metadata is not None:
            return self.chunk(dp_size)

        # Local data mode: use FFD algorithm
        total_size = self._get_total_size()
        if total_size % group_size != 0:
            raise FrameworkError(
                "FrameworkError",
                "DistributedBatchMemoryError",
                "Total size must be divisible by group_size",
            )

        # Handle seqlen calculation for both tensor and scalar types
        if "seqlen" in self.dataset.keys():
            seqlen = self.dataset["seqlen"]
            if isinstance(seqlen, torch.Tensor):
                reshaped = seqlen.view(-1, group_size)
                group_total_lens = reshaped.sum(dim=1)
            else:
                # Handle scalar/list case
                seqlen_list = (
                    seqlen if isinstance(seqlen, list) else [seqlen] * total_size
                )
                reshaped = [
                    seqlen_list[i : i + group_size]
                    for i in range(0, len(seqlen_list), group_size)
                ]
                group_total_lens = [sum(group) for group in reshaped]
        elif "attention_mask" in self.dataset.keys():
            attention_mask = self.dataset["attention_mask"]
            if isinstance(attention_mask, torch.Tensor):
                seqlen = attention_mask.sum(1)
                reshaped = seqlen.view(-1, group_size)
                group_total_lens = reshaped.sum(dim=1)
            else:
                # Fallback for scalar types - assume equal length
                group_total_lens = [group_size] * (total_size // group_size)
        else:
            # Fallback when neither seqlen nor attention_mask exists
            group_total_lens = [group_size] * (total_size // group_size)

        unsorted_group_rebalanced_indexs = ffd_allocate(
            group_total_lens, int(1e12), dp_size
        )
        group_rebalanced_indexs = sorted(
            [sorted(g) for g in unsorted_group_rebalanced_indexs]
        )
        batches = []
        for i in range(dp_size):
            indexes = []
            for group_index in group_rebalanced_indexs[i]:
                tmp_indexs = list(
                    range(
                        group_size * group_index, group_size * group_index + group_size
                    )
                )
                indexes.extend(tmp_indexs)
            split_data = {}
            for k, v in self.dataset.items():
                if isinstance(v, torch.Tensor):
                    split_data[k] = v[indexes]
                elif isinstance(v, list):
                    split_data[k] = [v[i] for i in indexes]
                else:
                    # For scalar values, keep as-is (they represent single sample)
                    split_data[k] = v
            batch = self.__class__.__new__(self.__class__)
            batch.dataset = split_data
            batch.metadata = None
            batch._is_local = True
            batches.append(batch)
        return batches

    def union(self, other: DistributedBatchMemory) -> DistributedBatchMemory:
        """Merge another batch with this one.

        Supports both metadata mode and local data mode.
        """
        # Both are in local data mode
        if self.dataset is not None and other.dataset is not None:
            return self._union_local_data(other)

        # Both are in metadata mode
        if self.metadata is not None and other.metadata is not None:
            return self._union_metadata(other)

        # Mixed mode: not supported
        raise FrameworkError(
            "FrameworkError",
            "DistributedBatchMemoryError",
            "Cannot union batches in different modes (metadata vs local data)",
        )

    def _union_metadata(self, other: DistributedBatchMemory) -> DistributedBatchMemory:
        """Merge two batches in metadata mode."""
        if self.metadata is None or other.metadata is None:
            raise FrameworkError(
                "FrameworkError",
                "DistributedBatchMemoryError",
                "Both batches must have metadata for union",
            )

        # Combine shards from both batches
        all_shards = self.metadata.shards + other.metadata.shards
        max_global_step = max(self.metadata.global_step, other.metadata.global_step)

        # Calculate total_batch_size with consideration for different keys
        total_batch_size = self._calculate_total_batch_size_from_shards(all_shards)

        # Create new metadata
        new_metadata = BatchMetadata(
            batch_id=str(uuid.uuid4()),
            global_step=max_global_step,
            total_batch_size=total_batch_size,
            shards=all_shards,
        )

        batch = self.__class__.__new__(self.__class__)
        batch.dataset = None
        batch.metadata = new_metadata
        batch._is_local = False
        return batch

    def _union_local_data(
        self, other: DistributedBatchMemory
    ) -> DistributedBatchMemory:
        """Merge two batches in local data mode."""
        merged_data = {k: v for k, v in self.dataset.items()}
        for k, v in other.dataset.items():
            if k in merged_data:
                if isinstance(merged_data[k], torch.Tensor) and isinstance(
                    v, torch.Tensor
                ):
                    merged_data[k] = torch.cat([merged_data[k], v], dim=0)
                elif isinstance(merged_data[k], list) and isinstance(v, list):
                    merged_data[k] = merged_data[k] + v
                else:
                    # Handle mixed types or scalar values
                    if isinstance(merged_data[k], list):
                        merged_data[k].append(v)
                    else:
                        merged_data[k] = [merged_data[k], v]
            else:
                merged_data[k] = v
        batch = self.__class__.__new__(self.__class__)
        batch.dataset = merged_data
        batch.metadata = None
        batch._is_local = True
        return batch

    def _get_total_size(self) -> int:
        """Get the total size of the dataset, supporting both tensor and scalar types.

        Returns
        -------
        int
            The total size (batch size) of the dataset
        """
        # Metadata mode: return total_batch_size from metadata
        if self.metadata is not None:
            return self.metadata.total_batch_size

        # Local data mode: calculate from dataset
        if not self.dataset:
            return 0

        first_value = next(iter(self.dataset.values()))
        if isinstance(first_value, torch.Tensor):
            return first_value.shape[0]
        elif isinstance(first_value, list):
            return len(first_value)
        else:
            # For scalar values, assume it's a single sample
            return 1

    def get_data(self) -> dict[str, torch.Tensor | Any]:
        """Get all data from the DistributedBatchMemory.

        If data is stored locally, returns it directly.
        If data is distributed (has metadata), fetches it from remote nodes
        via HTTP and assembles the complete dataset.

        Returns
        -------
        Dict[str, torch.Tensor | Any]
            Dictionary where keys are field names and values are tensors or
            other data types containing all values for that field across the
            entire batch.
        """
        # If we already have local data, return it
        if self.dataset is not None:
            return self.dataset

        # If we have metadata, fetch data from remote nodes
        if self.metadata is not None:
            client = self.get_client()

            # Use asyncio.run() to fetch data in a dedicated event loop.
            # NOTE: get_data() is a synchronous API and is expected to be called
            # from non-async contexts. If it is called from within an active event
            # loop, we raise a clear error.
            async def _fetch_batch():
                return await client.fetch_batch(self.metadata)

            try:
                self.dataset = asyncio.run(_fetch_batch())
            except RuntimeError as exc:
                # e.g. "asyncio.run() cannot be called from a running event loop"
                raise RuntimeError(
                    "get_data() cannot be called from within an async context when "
                    "fetching remote data. Please call aget_data() instead."
                ) from exc
            return self.dataset

        # No data and no metadata
        return {}

    async def aget_data(self) -> dict[str, torch.Tensor | Any]:
        """Async version of get_data().

        This is useful when calling from async contexts to avoid blocking.

        Returns
        -------
        Dict[str, torch.Tensor | Any]
            Dictionary where keys are field names and values are tensors or
            other data types.
        """
        # If we already have local data, return it
        if self.dataset is not None:
            return self.get_data()

        # If we have metadata, fetch data from remote nodes
        if self.metadata is not None:
            client = self.get_client()
            self.dataset = await client.fetch_batch(self.metadata)
            return self.dataset

        # No data and no metadata
        return {}

    @classmethod
    def concat(cls, data: list[DistributedBatchMemory]) -> DistributedBatchMemory:
        """Concatenate multiple DistributedBatchMemory objects

        Parameters
        ----------
        data : list[DistributedBatchMemory]
            List of DistributedBatchMemory objects to concatenate

        Returns
        -------
        DistributedBatchMemory
            Single concatenated DistributedBatchMemory object
        """
        assert data is not None and len(data) != 0

        # If all have metadata (distributed), concatenate metadata
        if all(item.metadata is not None for item in data):
            all_shards = []
            max_global_step = 0
            for item in data:
                all_shards.extend(item.metadata.shards)
                max_global_step = max(max_global_step, item.metadata.global_step)

            # Calculate total_batch_size with consideration for different keys
            total_batch_size = cls._calculate_total_batch_size_from_shards(all_shards)

            result = DistributedBatchMemory.__new__(DistributedBatchMemory)
            result.dataset = None
            result.metadata = BatchMetadata(
                batch_id=str(uuid.uuid4()),
                global_step=max_global_step,
                total_batch_size=total_batch_size,
                shards=all_shards,
            )
            result._is_local = False
            return result

        # Otherwise, concatenate local data
        # Note: concat_padded_tensors assumes all dicts have the same keys
        # If they don't, we need to handle it differently
        datasets = [k.dataset for k in data]

        # Check if all datasets have the same keys
        if datasets:
            all_keys = set()
            for dataset in datasets:
                all_keys.update(dataset.keys())

            same_keys = all(set(dataset.keys()) == all_keys for dataset in datasets)

            if same_keys and "attention_mask" in all_keys:
                # All datasets have the same keys, use concat_padded_tensors
                merged_data = concat_padded_tensors(datasets)
            else:
                # Different keys, merge manually
                from areal.utils import logging

                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Datasets have different keys, merging manually. "
                    f"All keys: {sorted(all_keys)}"
                )
                merged_data = {}
                for key in sorted(all_keys):
                    values_to_concat = []
                    for dataset in datasets:
                        if key in dataset:
                            values_to_concat.append(dataset[key])

                    if not values_to_concat:
                        continue

                    first_value = values_to_concat[0]
                    if isinstance(first_value, torch.Tensor):
                        # Check if tensors need padding (multi-dimensional with varying lengths)
                        if first_value.ndim > 1:
                            # Assume dim=1 is the sequence dimension
                            max_length = max(
                                tensor.shape[1] for tensor in values_to_concat
                            )
                            need_padding = any(
                                tensor.shape[1] < max_length
                                for tensor in values_to_concat
                            )

                            if need_padding:
                                # Pad tensors to max_length before concatenating
                                padded_tensors = []
                                for tensor in values_to_concat:
                                    if tensor.shape[1] < max_length:
                                        # Pad along sequence dimension (dim=1)
                                        pad_width = max_length - tensor.shape[1]
                                        # Determine pad value based on key
                                        if key == "attention_mask":
                                            pad_value = 0
                                        else:
                                            pad_value = 0.0

                                        # Create padding for dim=1 (sequence dimension)
                                        n_dim = tensor.ndim
                                        pad_mode = (0,) * (2 * (n_dim - 2)) + (
                                            0,
                                            pad_width,
                                        )
                                        padded_tensor = torch.nn.functional.pad(
                                            tensor, pad_mode, value=pad_value
                                        )
                                        padded_tensors.append(padded_tensor)
                                    else:
                                        padded_tensors.append(tensor)
                                merged_data[key] = torch.cat(padded_tensors, dim=0)
                            else:
                                # All tensors have same shape, directly concat
                                merged_data[key] = torch.cat(values_to_concat, dim=0)
                        else:
                            # 1D tensor, directly concat
                            merged_data[key] = torch.cat(values_to_concat, dim=0)
                    elif isinstance(first_value, list):
                        merged_list = []
                        for v in values_to_concat:
                            merged_list.extend(v)
                        merged_data[key] = merged_list
                    else:
                        merged_data[key] = first_value
        else:
            merged_data = {}

        result = DistributedBatchMemory.__new__(DistributedBatchMemory)
        result.dataset = merged_data
        result.metadata = None
        result._is_local = True
        return result

    @classmethod
    async def aclear(cls, global_step: int, node_addrs: set[str] | None = None):
        """Clear old batch data from distributed nodes.

        This performs garbage collection of batch data with step < global_step.

        Parameters
        ----------
        global_step : int
            Clear all data with step less than this value
        node_addrs : set[str] | None, optional
            Set of node addresses to clear. If None, no clearing is performed.
            Addresses should be in "host:port" format.
        """
        if node_addrs is None or len(node_addrs) == 0:
            return

        client = cls.get_client()
        await client._aclear(node_addrs, global_step)

    @classmethod
    def clear(cls, global_step: int, node_addrs: set[str] | None = None):
        """Synchronous version of clear().

        Parameters
        ----------
        global_step : int
            Clear all data with step less than this value
        node_addrs : set[str] | None, optional
            Set of node addresses to clear. If None, no clearing is performed.
        """
        if node_addrs is None or len(node_addrs) == 0:
            return

        loop = asyncio.get_event_loop()
        if loop.is_running():
            raise RuntimeError(
                "clear_sync() cannot be called from within an async context. "
                "Please use clear() instead."
            )
        else:
            loop.run_until_complete(cls.clear(global_step, node_addrs))

    def __getstate__(self):
        return {
            "dataset": self.dataset,
            "metadata": self.metadata,
            "_is_local": self._is_local,
        }

    def __setstate__(self, state):
        self.dataset = state.get("dataset")
        self.metadata = state.get("metadata")
        self._is_local = state.get("_is_local", True)

    def __getitem__(self, key):
        if isinstance(key, int):
            return {k: v[key] for k, v in self.dataset.items()}
        elif isinstance(key, str):
            return self.dataset[key]
        else:
            raise FrameworkError(
                "FrameworkError",
                "DistributedBatchMemoryError",
                "Key must be int or str",
            )

    def __setitem__(self, key, value):
        """Support two assignment methods:
        - str key: update entire attribute tensor
        - int index: requires converting data to list format for update (less efficient, avoid if possible)
        """
        if isinstance(key, str):
            # Update entire attribute tensor or scalar/list value
            if self.dataset:
                expected_total_size = self._get_total_size()
                if isinstance(value, torch.Tensor):
                    if value.shape[0] != expected_total_size:
                        raise FrameworkError(
                            "FrameworkError",
                            "DistributedBatchMemoryError",
                            f"The batch size of the tensor does not match. Expected {expected_total_size}, actual {value.shape[0]}",
                        )
                elif isinstance(value, list):
                    if len(value) != expected_total_size:
                        raise FrameworkError(
                            "FrameworkError",
                            "DistributedBatchMemoryError",
                            f"The batch size of the list does not match. Expected {expected_total_size}, actual {len(value)}",
                        )
            self.dataset[key] = value
        else:
            raise FrameworkError(
                "FrameworkError", "DistributedBatchMemoryError", "key must be str"
            )

    def __delitem__(self, key):
        """Support two deletion methods:
        - int index: delete sample at specified position
        - str key: delete entire attribute
        """
        if isinstance(key, int):
            # Convert to list format for deletion
            list_dataset = convert_dict_to_list(self.dataset)
            del list_dataset[key]
            self.dataset = convert_list_to_dict(list_dataset)
        elif isinstance(key, str):
            # Delete entire attribute directly
            if key in self.dataset:
                del self.dataset[key]
        else:
            raise FrameworkError(
                "FrameworkError",
                "DistributedBatchMemoryError",
                f"key: {type(key)} must be str or int",
            )

    def __str__(self):
        if self.metadata is not None:
            # Show metadata information
            return (
                f"DistributedBatchMemory<metadata: {self.metadata}, "
                f"is_local={self._is_local}, data_loaded={self.dataset is not None}>"
            )

        if not self.dataset:
            return "DistributedBatchMemory<empty>"

        total_size = self._get_total_size()
        keys = list(self.dataset.keys())
        shapes = {}
        for k, v in self.dataset.items():
            if isinstance(v, torch.Tensor):
                shapes[k] = v.shape
            elif isinstance(v, list):
                shapes[k] = f"list[{len(v)}]"
            else:
                shapes[k] = f"scalar({type(v).__name__})"
        return f"DistributedBatchMemory<total_size={total_size}, keys={keys}, shapes={shapes}>"

    def __len__(self):
        """Return the total size."""
        return self._get_total_size()

    def __repr__(self):
        return self.__str__()
