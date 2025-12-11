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
    ShardId,
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

    def _group_shards_by_task_id(
        self, shards: list[ShardMetadata]
    ) -> dict[str, list[ShardMetadata]]:
        """Group shards by task_id."""
        task_id_to_shards: dict[str, list[ShardMetadata]] = defaultdict(list)
        for shard in shards:
            task_id = shard.shard_id.task_id
            task_id_to_shards[task_id].append(shard)
        return task_id_to_shards

    def _chunk_metadata(self, dp_size: int) -> list[DistributedBatchMemory]:
        """Split metadata across data parallel processes.

        Groups shards by task_id and distributes task groups across dp_size processes.
        """

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

        task_id_to_shards = self._group_shards_by_task_id(self.metadata.shards)
        task_groups = list(task_id_to_shards.items())

        if not task_groups:
            batches = []
            for i in range(dp_size):
                new_metadata = BatchMetadata(
                    batch_id=f"{self.metadata.batch_id}_chunk_{i}",
                    shards=[],
                )
                batches.append(self.__class__(dataset=None, metadata=new_metadata))
            return batches

        # Distribute task_id groups across dp_size processes
        shards_per_dp_rank: list[list[ShardMetadata]] = [[] for _ in range(dp_size)]
        task_id_counts_per_rank = [0] * dp_size

        for task_id, shards in task_groups:
            target_rank = min(range(dp_size), key=lambda i: task_id_counts_per_rank[i])
            shards_per_dp_rank[target_rank].extend(shards)
            task_id_counts_per_rank[target_rank] += 1

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
        all_shards = self.metadata.shards + other.metadata.shards
        self.metadata = BatchMetadata(
            batch_id=str(uuid.uuid4()),
            shards=all_shards,
        )
        self.dataset = None

    def _union_local_data(self, other: DistributedBatchMemory) -> None:
        """Merge two batches in local data status by modifying self in-place."""
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
        """Get the total size of the dataset."""
        if self.metadata is not None:
            if not self.metadata.shards:
                return 0
            task_id_to_shards = self._group_shards_by_task_id(self.metadata.shards)

            total_size = 0
            for shards in task_id_to_shards.values():
                if (
                    shards
                    and shards[0].tensor_metadata
                    and shards[0].tensor_metadata.shape
                ):
                    total_size += shards[0].tensor_metadata.shape[0]

            return total_size

        if not self.dataset:
            return 0

        first_value = next(iter(self.dataset.values()))
        if isinstance(first_value, torch.Tensor):
            return first_value.shape[0]
        elif isinstance(first_value, list):
            return len(first_value)
        else:
            return 1

    def _merge_shards(self, shard_data_list: list[dict[str, Any]]) -> dict[str, Any]:
        """Merge shard data into a complete dataset."""
        if not shard_data_list:
            return {}

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
        """Merge shards that may have different keys.

        Handles padding for tensors with different sequence lengths.
        """
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
        """Concatenate multiple DistributedBatchMemory objects."""
        assert data is not None and len(data) != 0

        has_metadata = [item.is_remote for item in data]
        if not all(has_metadata) and any(has_metadata):
            raise FrameworkError(
                "FrameworkError",
                "DistributedBatchMemoryError",
                "Cannot concatenate batches with mixed statuses. "
                "All batches must be either in REMOTE status or LOCAL status.",
            )

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

        datasets = [k.dataset for k in data]
        if not datasets:
            merged_data = {}
        else:
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

            merged_data = concat_padded_tensors(datasets)

        return cls(dataset=merged_data, metadata=None)

    @classmethod
    async def aclear(
        cls,
        target: DistributedBatchMemory | list[str],
        node_addrs: set[str],
    ):
        """Clear old batch data from distributed nodes."""
        if not node_addrs:
            return

        client = cls.get_client()

        # Extract shard_ids from target
        if isinstance(target, DistributedBatchMemory):
            if target.metadata is None or not target.metadata.shards:
                return
            shard_ids = [shard.shard_id for shard in target.metadata.shards]
        elif isinstance(target, list):
            shard_ids = [ShardId.from_string(s) for s in target]
        else:
            raise TypeError(
                f"target must be DistributedBatchMemory or list[str], got {type(target)}"
            )

        await client.clear_batches(node_addrs, shard_ids)

    @classmethod
    def clear(
        cls,
        target: DistributedBatchMemory | list[str],
        node_addrs: set[str],
    ):
        """Synchronous version of clear()."""
        asyncio.run(cls.aclear(target, node_addrs))

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

        if self.dataset is None:
            self.dataset = {}
        self.dataset[key] = value

    def __delitem__(self, key):
        """Delete item by int index or str key."""
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
            list_dataset = convert_dict_to_list(self.dataset)
            del list_dataset[key]
            self.dataset = convert_list_to_dict(list_dataset)
        elif isinstance(key, str):
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
            return (
                f"DistributedBatchMemory<status={status_name}, "
                f"size={total_size}, "
                f"batch_id={self.metadata.batch_id}, "
                f"num_shards={len(self.metadata.shards)}, "
                f"data_loaded={self.dataset is not None}>"
            )

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
