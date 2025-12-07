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
    ShardMetadata,
)
from areal.utils import logging
from areal.utils.batch_utils import (
    convert_dict_to_list,
    convert_list_to_dict,
    validate_dict_dataset,
)
from areal.utils.data import concat_padded_tensors
from areal.utils.datapack import ffd_allocate
from areal.utils.errors import FrameworkError

logger = logging.getLogger("DistributedBatchMemory")


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
    """

    dataset = None
    metadata: BatchMetadata | None = None

    # Shared client for fetching data
    _client: BatchDataClient | None = None

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
        return instance

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
            batches.append(batch)
        return batches

    @staticmethod
    def _group_shards_by_keys(
        shards: list[ShardMetadata],
    ) -> tuple[list[list[ShardMetadata]], int]:
        """Group shards by fields.keys() and calculate total batch size.
        Shards with identical keys (fields.keys()) are grouped together.
        Shards with any different keys belong to different groups.
        """
        if not shards:
            return [], 0

        # Group shards by their fields.keys() (using sorted tuple as key)
        from collections import defaultdict

        groups_dict: dict[tuple[str, ...], list[ShardMetadata]] = defaultdict(list)
        for shard in shards:
            keys_tuple = tuple(sorted(shard.fields.keys()))
            groups_dict[keys_tuple].append(shard)

        groups_list = list(groups_dict.values())

        # Validate: different groups should not have overlapping keys
        group_keys_list = [set(group[0].fields.keys()) for group in groups_list]
        for i in range(len(group_keys_list)):
            for j in range(i + 1, len(group_keys_list)):
                overlap = group_keys_list[i] & group_keys_list[j]
                assert not overlap, (
                    f"Groups {i} and {j} have overlapping keys: {overlap}"
                )

        # Calculate total batch size for each group
        group_totals = [
            sum(shard.batch_size for shard in group) for group in groups_list
        ]

        # Validate: all groups should have the same total batch_size
        # This ensures consistency across different field groups
        if len(group_totals) > 1:
            assert len(set(group_totals)) == 1, (
                f"Different groups have inconsistent total batch_sizes: {group_totals}"
            )

        # Return groups and total batch size (use first group's total if multiple groups)
        total_batch_size = group_totals[0] if group_totals else 0
        return groups_list, total_batch_size

    @staticmethod
    def _chunk_shard_group(
        shard_group: list[ShardMetadata],
        dp_size: int,
    ) -> list[list[ShardMetadata]]:
        """Chunk a single shard group across dp_size data parallel processes.

        Splits shards in the group evenly across dp_size groups while preserving
        sample ordering. A physical shard can be split into multiple logical
        sub-shards using offset and batch_size to represent sub-ranges.

        Parameters
        ----------
        shard_group : list[ShardMetadata]
            List of shards in the same group (same fields.keys())
        dp_size : int
            Number of data parallel processes

        Returns
        -------
        list[list[ShardMetadata]]
            List of dp_size groups, each containing chunked shards
        """
        if not shard_group:
            return [[] for _ in range(dp_size)]

        # Calculate total batch size for this group
        total = sum(shard.batch_size for shard in shard_group)

        # Pre-calculate sample ranges for each dp group
        part_size = (total + dp_size - 1) // dp_size
        group_ranges = []
        for i in range(dp_size):
            start = i * part_size
            end = min(start + part_size, total)
            group_ranges.append((start, end))

        # Distribute shards to groups based on sample ranges
        dp_groups: list[list[ShardMetadata]] = [[] for _ in range(dp_size)]
        current_sample_idx = 0

        for shard in shard_group:
            shard_start = current_sample_idx
            shard_end = current_sample_idx + shard.batch_size
            current_sample_idx = shard_end

            # Find which dp groups this shard overlaps with
            for dp_idx, (group_start, group_end) in enumerate(group_ranges):
                overlap_start = max(shard_start, group_start)
                overlap_end = min(shard_end, group_end)

                if overlap_start < overlap_end:
                    # Calculate offset within the original shard
                    offset_in_shard = overlap_start - shard_start
                    length = overlap_end - overlap_start

                    sub_shard = ShardMetadata(
                        node_id=shard.node_id,
                        node_addr=shard.node_addr,
                        shard_id=shard.shard_id,
                        batch_size=length,
                        offset=shard.offset + offset_in_shard,
                        fields=shard.fields,
                    )
                    dp_groups[dp_idx].append(sub_shard)

        return dp_groups

    def _chunk_metadata(self, dp_size: int) -> list[DistributedBatchMemory]:
        """Split metadata across data parallel processes."""

        if self.metadata is None:
            raise FrameworkError(
                "FrameworkError",
                "DistributedBatchMemoryError",
                "No metadata to split",
            )

        if dp_size <= 0:
            raise FrameworkError(
                "FrameworkError",
                "DistributedBatchMemoryError",
                "dp_size must be positive",
            )

        # Step 1: Group shards by fields.keys()
        key_groups, _ = self._group_shards_by_keys(self.metadata.shards)

        # Step 2: Chunk each group across dp_size processes
        # Result: list[list[list[ShardMetadata]]] - [key_group][dp_idx][shards]
        chunked_groups = []
        for group in key_groups:
            chunked_group = self._chunk_shard_group(group, dp_size)
            chunked_groups.append(chunked_group)

        # Step 3: Merge results from different groups
        # Combine shards from all key groups for each dp process
        # Since different groups have non-overlapping keys but same total batch_size,
        # the merged batch_size should equal each group's batch_size (not sum)
        merged_chunks: list[list[ShardMetadata]] = [[] for _ in range(dp_size)]

        # Calculate batch_size for each dp process from the first group
        # (all groups should have the same chunk sizes after chunking)
        if chunked_groups:
            chunk_sizes = [
                sum(shard.batch_size for shard in chunked_groups[0][dp_idx])
                for dp_idx in range(dp_size)
            ]

            # Validate: all groups should have the same chunk sizes for each dp process
            for group_idx, chunked_group in enumerate(chunked_groups):
                for dp_idx in range(dp_size):
                    group_dp_size = sum(
                        shard.batch_size for shard in chunked_group[dp_idx]
                    )
                    assert group_dp_size == chunk_sizes[dp_idx], (
                        f"Group {group_idx} dp_idx {dp_idx} has batch_size {group_dp_size}, "
                        f"expected {chunk_sizes[dp_idx]}"
                    )
        else:
            chunk_sizes = [0] * dp_size

        # Merge shards from all groups for each dp process
        for chunked_group in chunked_groups:
            for dp_idx in range(dp_size):
                merged_chunks[dp_idx].extend(chunked_group[dp_idx])

        # Create chunked batches
        batches = []
        for i in range(dp_size):
            new_metadata = BatchMetadata(
                batch_id=f"{self.metadata.batch_id}_chunk_{i}",
                global_step=self.metadata.global_step,
                total_batch_size=chunk_sizes[i],
                shards=merged_chunks[i],
            )
            batch = self.__class__.__new__(self.__class__)
            batch.dataset = None
            batch.metadata = new_metadata
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
        # Combine shards from both batches
        all_shards = self.metadata.shards + other.metadata.shards
        max_global_step = max(self.metadata.global_step, other.metadata.global_step)

        # Calculate total_batch_size (validates different fields have same total)
        _, total_batch_size = self._group_shards_by_keys(all_shards)

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

    def _merge_shards(
        self, shard_data_list: list[dict[str, torch.Tensor | Any]]
    ) -> dict[str, torch.Tensor | Any]:
        """Merge shard data into a complete dataset."""
        if not shard_data_list:
            return {}

        # Check if all shards have the same keys
        all_keys = set()
        for shard_data in shard_data_list:
            all_keys.update(shard_data.keys())

        same_keys = all(
            set(shard_data.keys()) == all_keys for shard_data in shard_data_list
        )

        if same_keys and "attention_mask" in all_keys:
            return concat_padded_tensors(shard_data_list)
        else:
            return self._merge_shards_with_different_keys(shard_data_list, all_keys)

    def _merge_shards_with_different_keys(
        self,
        shard_data_list: list[dict[str, torch.Tensor]],
        all_keys: set[str],
    ) -> dict[str, torch.Tensor]:
        """Merge shards that may have different keys."""
        result = {}

        for key in sorted(all_keys):
            values_to_concat = []

            for shard_data in shard_data_list:
                if key in shard_data:
                    values_to_concat.append(shard_data[key])

            if not values_to_concat:
                continue

            first_value = values_to_concat[0]
            if first_value.ndim > 1:
                max_length = max(tensor.shape[1] for tensor in values_to_concat)
                need_padding = any(
                    tensor.shape[1] < max_length for tensor in values_to_concat
                )

                if need_padding:
                    pad_value = 0 if key == "attention_mask" else 0.0
                    padded_tensors = []
                    for tensor in values_to_concat:
                        if tensor.shape[1] < max_length:
                            pad_width = max_length - tensor.shape[1]
                            n_dim = tensor.ndim
                            pad_mode = (0,) * (2 * (n_dim - 2)) + (0, pad_width)
                            padded_tensors.append(
                                torch.nn.functional.pad(
                                    tensor, pad_mode, value=pad_value
                                )
                            )
                        else:
                            padded_tensors.append(tensor)
                    result[key] = torch.cat(padded_tensors, dim=0)
                else:
                    result[key] = torch.cat(values_to_concat, dim=0)
            else:
                result[key] = torch.cat(values_to_concat, dim=0)

        return result

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
        if self.dataset is not None:
            return self.dataset

        if self.metadata is not None:
            client = self.get_client()

            # NOTE: get_data() is synchronous and cannot be called from async context.
            # Use asyncio.run() to fetch data in a dedicated event loop.
            async def _fetch_shards():
                shard_data_list = await client.fetch_shards(self.metadata)
                return self._merge_shards(shard_data_list)

            try:
                self.dataset = asyncio.run(_fetch_shards())
            except RuntimeError as exc:
                raise RuntimeError(
                    "get_data() cannot be called from within an async context when "
                    "fetching remote data. Please call aget_data() instead."
                ) from exc
            return self.dataset

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
        if self.dataset is not None:
            return self.get_data()

        if self.metadata is not None:
            client = self.get_client()
            shard_data_list = await client.fetch_shards(self.metadata)
            self.dataset = self._merge_shards(shard_data_list)
            return self.dataset

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

        # Check if we have mixed modes (some with metadata, some without)
        has_metadata = [item.metadata is not None for item in data]
        if not all(has_metadata) and any(has_metadata):
            raise FrameworkError(
                "FrameworkError",
                "DistributedBatchMemoryError",
                "Cannot concatenate batches with mixed modes. "
                "All batches must be either in metadata mode or local data mode.",
            )

        # If all have metadata (distributed), concatenate metadata
        if all(item.metadata is not None for item in data):
            all_shards = []
            max_global_step = 0
            for item in data:
                all_shards.extend(item.metadata.shards)
                max_global_step = max(max_global_step, item.metadata.global_step)

            # Calculate total_batch_size (validates different fields have same total)
            _, total_batch_size = cls._group_shards_by_keys(all_shards)

            result = DistributedBatchMemory.__new__(DistributedBatchMemory)
            result.dataset = None
            result.metadata = BatchMetadata(
                batch_id=str(uuid.uuid4()),
                global_step=max_global_step,
                total_batch_size=total_batch_size,
                shards=all_shards,
            )
            return result

        # Concatenate local data
        datasets = [k.dataset for k in data]
        if not datasets:
            merged_data = {}
        else:
            # Verify all datasets have the same keys
            all_keys = set()
            for dataset in datasets:
                all_keys.update(dataset.keys())

            same_keys = all(set(dataset.keys()) == all_keys for dataset in datasets)
            if not same_keys:
                key_sets = [set(dataset.keys()) for dataset in datasets]
                raise FrameworkError(
                    "FrameworkError",
                    "DistributedBatchMemoryError",
                    f"All datasets must have the same keys. "
                    f"Found key sets: {[sorted(ks) for ks in key_sets]}",
                )

            # All datasets have the same keys, use concat_padded_tensors
            merged_data = concat_padded_tensors(datasets)

        result = DistributedBatchMemory.__new__(DistributedBatchMemory)
        result.dataset = merged_data
        result.metadata = None
        return result

    @classmethod
    async def aclear(cls, global_step: int, node_addrs: set[str] | None = None):
        """Clear old batch data from distributed nodes."""
        if node_addrs is None or len(node_addrs) == 0:
            return

        client = cls.get_client()
        await client.clear_batches(node_addrs, global_step)

    @classmethod
    def clear(cls, global_step: int, node_addrs: set[str] | None = None):
        """Synchronous version of clear()."""
        if node_addrs is None or len(node_addrs) == 0:
            return

        asyncio.run(cls.aclear(global_step, node_addrs))

    def __getstate__(self):
        return {
            "dataset": self.dataset,
            "metadata": self.metadata,
        }

    def __setstate__(self, state):
        self.dataset = state.get("dataset")
        self.metadata = state.get("metadata")

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
                f"data_loaded={self.dataset is not None}>"
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
