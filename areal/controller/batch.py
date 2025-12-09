from __future__ import annotations

import asyncio
import uuid
from collections import defaultdict
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from torch import Tensor

from areal.api.controller_api import DistributedBatch
from areal.controller.batch_metadata import (
    BatchMetadata,
    ShardMetadata,
)

if TYPE_CHECKING:
    from areal.controller.batch_client import BatchDataClient
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


class BatchStatus(Enum):
    """Explicit status enum for DistributedBatchMemory.

    Attributes
    ----------
    LOCAL : auto
        Data stored locally in memory
    REMOTE : auto
        Only metadata; data fetched on-demand via HTTP
    EMPTY : auto
        Neither data nor metadata present (invalid/empty state)
    """

    LOCAL = auto()  # Data stored locally in memory
    REMOTE = auto()  # Only metadata; data fetched on-demand
    EMPTY = auto()  # Neither present (invalid state)


class DistributedBatchMemory(DistributedBatch):
    """Distributed batch memory with metadata-driven data access.

    This class separates metadata (data shape, location) from actual data.
    The control plane only passes metadata, and actual data is fetched on-demand
    via HTTP from distributed nodes.

    The class supports two statuses:
    - LOCAL status: Data is stored locally in memory (dataset is not None)
    - REMOTE status: Only metadata is present; data fetched on-demand via HTTP

    Use the `status` property to check the current status, and `is_local`/`is_remote`
    for convenience checks.

    Attributes
    ----------
    dataset : dict[str, torch.Tensor | Any] | None
        The actual data (lazy-loaded, None until get_data() is called)
    metadata : BatchMetadata | None
        Metadata describing the distributed batch
    """

    # Shared client for fetching data (singleton pattern)
    _client: ClassVar[BatchDataClient | None] = None

    def __init__(
        self,
        dataset: dict[str, torch.Tensor | Any] | None = None,
        metadata: BatchMetadata | None = None,
    ):
        """Initialize a DistributedBatchMemory instance.

        Parameters
        ----------
        dataset : dict[str, torch.Tensor | Any] | None
            The actual data stored locally. If provided, batch is in LOCAL status.
        metadata : BatchMetadata | None
            Metadata describing the distributed batch. If provided without dataset,
            batch is in REMOTE status.
        """
        self.dataset = dataset
        self.metadata = metadata

    @property
    def status(self) -> BatchStatus:
        """Get the current status of this batch (LOCAL, REMOTE, or EMPTY)."""
        has_data = self.dataset is not None and len(self.dataset) > 0
        has_meta = self.metadata is not None

        if has_data:
            return BatchStatus.LOCAL
        if has_meta:
            return BatchStatus.REMOTE
        return BatchStatus.EMPTY

    @property
    def is_local(self) -> bool:
        """Check if data is available locally (no fetch needed)."""
        return self.status == BatchStatus.LOCAL

    @property
    def is_remote(self) -> bool:
        """Check if this batch is in metadata-only status.

        Returns
        -------
        bool
            True if batch is in REMOTE status
        """
        return self.status == BatchStatus.REMOTE

    def _require_status(self, *allowed_statuses: BatchStatus, operation: str) -> None:
        """Assert that current status is one of the allowed status."""
        if self.status not in allowed_statuses:
            raise FrameworkError(
                "FrameworkError",
                "BatchStatusError",
                f"Operation '{operation}' requires status {[m.name for m in allowed_statuses]}, "
                f"but current status is {self.status.name}",
            )

    def _require_same_status(
        self, other: DistributedBatchMemory, operation: str
    ) -> None:
        """Assert that both batches are in the same status."""
        if self.status != other.status:
            raise FrameworkError(
                "FrameworkError",
                "BatchStatusError",
                f"Operation '{operation}' requires both batches in same status. "
                f"Self is {self.status.name}, other is {other.status.name}",
            )

    @classmethod
    def get_client(cls) -> BatchDataClient:
        """Get or create the shared batch data client.

        Returns
        -------
        BatchDataClient
            Shared client instance
        """
        if cls._client is None:
            # Import here to avoid circular dependency
            from areal.controller.batch_client import BatchDataClient

            cls._client = BatchDataClient()
        return cls._client

    @classmethod
    def from_dict(cls, dict_dataset: dict[str, Tensor | Any]):
        """Create a DistributedBatchMemory from dictionary format dataset.

        This creates a LOCAL status batch with data stored in memory.

        Parameters
        ----------
        dict_dataset : Dict[str, Union[Tensor, Any]]
            Dictionary format dataset, where values can be Tensor, scalar, or list types

        Returns
        -------
        DistributedBatchMemory
            New DistributedBatchMemory instance in LOCAL status
        """
        validate_dict_dataset(dict_dataset)
        return cls(dataset=dict_dataset, metadata=None)

    @classmethod
    def from_metadata(cls, metadata: BatchMetadata) -> DistributedBatchMemory:
        """Create a DistributedBatchMemory from metadata (without actual data).

        This creates a REMOTE status batch. The data will be fetched lazily
        when get_data() is called.

        Parameters
        ----------
        metadata : BatchMetadata
            Metadata describing the distributed batch

        Returns
        -------
        DistributedBatchMemory
            New DistributedBatchMemory instance in REMOTE status
        """
        return cls(dataset=None, metadata=metadata)

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

        Supports both REMOTE status (metadata) and LOCAL status (data).

        Parameters
        ----------
        dp_size : int
            Number of data parallel processes

        Returns
        -------
        list[DistributedBatchMemory]
            List of chunked batches

        Raises
        ------
        FrameworkError
            If batch is in EMPTY status
        """
        # REMOTE status: split shards across dp_size groups
        if self.is_remote:
            return self._chunk_metadata(dp_size)

        # LOCAL status: split actual data
        self._require_status(BatchStatus.LOCAL, operation="chunk")

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
            batches.append(self.__class__(dataset=split_data, metadata=None))
        return batches

    @staticmethod
    def _infer_shard_size(shard: ShardMetadata) -> int:
        """从 tensor 元数据推断 shard 大小（使用首维度）。"""
        if not shard.fields:
            return 0
        first_meta = next(iter(shard.fields.values()))
        if not first_meta.shape:
            return 0
        return first_meta.shape[0]

    @classmethod
    def _group_shards_by_keys(
        cls, shards: list[ShardMetadata]
    ) -> tuple[list[list[ShardMetadata]], int]:
        """Group shards by fields.keys() and calculate total batch size.
        Shards with identical keys (fields.keys()) are grouped together.
        Shards with any different keys belong to different groups.
        """
        if not shards:
            return [], 0

        # Group shards by their fields.keys() (using sorted tuple as key)
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

        group_totals = [
            sum(cls._infer_shard_size(shard) for shard in group)
            for group in groups_list
        ]
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
        """Evenly split ``shard_group`` into ``dp_size`` contiguous parts."""
        if dp_size <= 0:
            raise FrameworkError(
                "FrameworkError",
                "DistributedBatchMemoryError",
                "dp_size must be positive",
            )

        total = len(shard_group)
        if total == 0:
            return [[] for _ in range(dp_size)]

        base = total // dp_size
        remainder = total % dp_size

        dp_groups: list[list[ShardMetadata]] = []
        start = 0
        for idx in range(dp_size):
            # 前 remainder 个分组多分一个
            size = base + (1 if idx < remainder else 0)
            end = start + size
            dp_groups.append(shard_group[start:end])
            start = end

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
        shards_by_field_keys, _ = self._group_shards_by_keys(self.metadata.shards)

        # Step 2: Chunk each field group across dp_size processes
        # Result: list[list[list[ShardMetadata]]] - [field_group_idx][dp_rank_idx][shards]
        chunked_per_dp = []
        for field_group in shards_by_field_keys:
            dp_chunks = self._chunk_shard_group(field_group, dp_size)
            chunked_per_dp.append(dp_chunks)

        # Step 3: Merge results from different field groups
        # Combine shards from all field groups for each dp rank
        # Since different groups have non-overlapping keys but same total batch_size,
        # the merged batch_size should equal each group's batch_size (not sum)
        shards_per_dp_rank: list[list[ShardMetadata]] = [[] for _ in range(dp_size)]

        # Merge shards from all field groups for each dp rank
        for dp_chunks in chunked_per_dp:
            for dp_idx in range(dp_size):
                shards_per_dp_rank[dp_idx].extend(dp_chunks[dp_idx])

        # Create chunked batches
        batches = []
        for i in range(dp_size):
            new_metadata = BatchMetadata(
                batch_id=f"{self.metadata.batch_id}_chunk_{i}",
                shards=shards_per_dp_rank[i],
            )
            batches.append(self.__class__(dataset=None, metadata=new_metadata))

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
        For REMOTE status, this method will fall back to simple chunking
        since we cannot determine sequence lengths without fetching the data.
        """
        # REMOTE status: fall back to simple chunking
        # TODO: FFD requires sequence length information which is not available in metadata yet
        if self.is_remote:
            return self.chunk(dp_size)

        # LOCAL status: use FFD algorithm
        self._require_status(BatchStatus.LOCAL, operation="chunk_by_ffd")

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
            batches.append(self.__class__(dataset=split_data, metadata=None))
        return batches

    def union_(self, other: DistributedBatchMemory) -> DistributedBatchMemory:
        """In-place merge. Mutates ``self`` and returns it."""
        self._require_same_status(other, operation="union_")

        if self.is_remote:
            self._union_metadata(other)
        else:
            self._union_local_data(other)

        return self

    def _union_metadata(self, other: DistributedBatchMemory) -> None:
        """Merge two batches in metadata status by modifying self in-place."""
        # Combine shards from both batches
        all_shards = self.metadata.shards + other.metadata.shards

        # Update self.metadata directly
        self.metadata = BatchMetadata(
            batch_id=str(uuid.uuid4()),
            shards=all_shards,
        )
        self.dataset = None

    def _union_local_data(self, other: DistributedBatchMemory) -> None:
        """Merge two batches in local data status by modifying self in-place."""
        # Merge data directly into self.dataset
        for k, v in other.dataset.items():
            if k in self.dataset:
                if isinstance(self.dataset[k], torch.Tensor) and isinstance(
                    v, torch.Tensor
                ):
                    self.dataset[k] = torch.cat([self.dataset[k], v], dim=0)
                elif isinstance(self.dataset[k], list) and isinstance(v, list):
                    self.dataset[k] = self.dataset[k] + v
                else:
                    # Handle mixed types or scalar values
                    if isinstance(self.dataset[k], list):
                        self.dataset[k].append(v)
                    else:
                        self.dataset[k] = [self.dataset[k], v]
            else:
                self.dataset[k] = v
        self.metadata = None

    def _get_total_size(self) -> int:
        """Get the total size of the dataset"""
        if self.metadata is not None:
            _, total_size = self._group_shards_by_keys(self.metadata.shards)
            return total_size

        # Local data status: calculate from dataset
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

    def _merge_shards(self, shard_data_list: list[dict[str, Any]]) -> dict[str, Any]:
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
        shard_data_list: list[dict[str, Any]],
        all_keys: set[str],
    ) -> dict[str, Any]:
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

        # Check if we have mixed statuses (some REMOTE, some LOCAL)
        has_metadata = [item.is_remote for item in data]
        if not all(has_metadata) and any(has_metadata):
            raise FrameworkError(
                "FrameworkError",
                "DistributedBatchMemoryError",
                "Cannot concatenate batches with mixed statuses. "
                "All batches must be either in REMOTE status or LOCAL status.",
            )

        # If all are in REMOTE status, concatenate metadata
        if all(item.is_remote for item in data):
            all_shards = []
            for item in data:
                all_shards.extend(item.metadata.shards)

            return cls(
                dataset=None,
                metadata=BatchMetadata(
                    batch_id=str(uuid.uuid4()),
                    shards=all_shards,
                ),
            )

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

        return cls(dataset=merged_data, metadata=None)

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
        if self.is_remote:
            raise FrameworkError(
                "FrameworkError",
                "DistributedBatchMemoryError",
                "Cannot access items in REMOTE status. Call get_data() first to fetch data.",
            )

        if self.dataset is None:
            raise FrameworkError(
                "FrameworkError",
                "DistributedBatchMemoryError",
                "Dataset is empty.",
            )

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
        if not isinstance(key, str):
            raise FrameworkError(
                "FrameworkError", "DistributedBatchMemoryError", "key must be str"
            )

        # Special handling for DistributedBatchMemory values
        # The key is expected to match the field key in value's metadata (set via result_key)
        if isinstance(value, DistributedBatchMemory):
            # Merge using in-place union
            self.union_(value)
            return

        if self.metadata is not None:
            raise FrameworkError(
                "FrameworkError",
                "DistributedBatchMemoryError",
                "Cannot assign regular value to metadata-status batch. "
                "Use union() with a DistributedBatchMemory object, or get_data() first.",
            )

        # Local data status: proceed with assignment
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

        # Ensure dataset exists
        if self.dataset is None:
            self.dataset = {}
        self.dataset[key] = value

    def __delitem__(self, key):
        """Support two deletion methods:
        - int index: delete sample at specified position
        - str key: delete entire attribute
        """
        if self.is_remote:
            raise FrameworkError(
                "FrameworkError",
                "DistributedBatchMemoryError",
                "Cannot delete items in REMOTE status. Call get_data() first to fetch data.",
            )

        if self.dataset is None:
            raise FrameworkError(
                "FrameworkError",
                "DistributedBatchMemoryError",
                "Dataset is empty.",
            )

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
        status_name = self.status.name
        total_size = self._get_total_size()

        if self.status == BatchStatus.EMPTY:
            return f"DistributedBatchMemory<status={status_name}, empty>"

        if self.is_remote:
            # Show metadata information for REMOTE status
            return (
                f"DistributedBatchMemory<status={status_name}, "
                f"size={total_size}, "
                f"batch_id={self.metadata.batch_id}, "
                f"num_shards={len(self.metadata.shards)}, "
                f"data_loaded={self.dataset is not None}>"
            )

        # LOCAL status: show data details
        keys = list(self.dataset.keys())
        shapes = {}
        for k, v in self.dataset.items():
            if isinstance(v, torch.Tensor):
                shapes[k] = v.shape
            elif isinstance(v, list):
                shapes[k] = f"list[{len(v)}]"
            else:
                shapes[k] = f"scalar({type(v).__name__})"
        return (
            f"DistributedBatchMemory<status={status_name}, "
            f"size={total_size}, keys={keys}, shapes={shapes}>"
        )

    def __len__(self):
        """Return the total size."""
        return self._get_total_size()

    def __repr__(self):
        return self.__str__()
