import asyncio
from dataclasses import dataclass
from typing import Any, Literal

import aiohttp
import numpy as np
import orjson
import torch

from areal.utils import logging

TOKENIZER_ARCHIVE_INLINE_THRESHOLD = 512 * 1024
TOKENIZER_ZSTD_THRESHOLD = 20 * 1024 * 1024
TokenizerCompression = Literal["zip", "zstd"]

logger = logging.getLogger("SyncRPCServer")


@dataclass(frozen=True)
class ShardId:
    """Unique identifier for a tensor shard."""

    task_id: str
    key: str

    def __str__(self) -> str:
        """Convert to string format: task_id:key"""
        return f"{self.task_id}:{self.key}"

    @classmethod
    def from_string(cls, s: str) -> "ShardId":
        """Parse from string format: task_id:key"""
        parts = s.split(":", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid ShardId string: {s}")
        return cls(task_id=parts[0], key=parts[1])


@dataclass
class ShardInfo:
    """Metadata for a single shard of an RTensor."""

    shard_id: ShardId
    node_addr: str
    shape: list[int]
    dtype: str


@dataclass
class ShardLayout:
    shard_id: ShardId
    size: int


@dataclass
class BatchLayout:
    layout: list[ShardLayout]

    @staticmethod
    def find_in_structure(obj):
        """Find first BatchLayout in a nested structure."""
        if isinstance(obj, BatchLayout):
            return obj
        if isinstance(obj, dict):
            for v in obj.values():
                result = BatchLayout.find_in_structure(v)
                if result:
                    return result
        if isinstance(obj, list):
            for item in obj:
                result = BatchLayout.find_in_structure(item)
                if result:
                    return result
        return None

    @staticmethod
    def exists_in_structure(obj):
        """Check if any BatchLayout exists in a nested structure."""
        return BatchLayout.find_in_structure(obj) is not None


@dataclass
class RTensor:
    """Single tensor distributed as CPU shards across nodes."""

    shards: list[ShardInfo]
    _data: torch.Tensor | None = None

    def get_key(self) -> str:
        """Get key from first shard (all shards must have same key)."""
        if not self.shards:
            raise ValueError("Cannot get key from empty RTensor")
        key = self.shards[0].shard_id.key
        # Validate consistency
        for shard in self.shards[1:]:
            if shard.shard_id.key != key:
                raise ValueError(f"Inconsistent keys: {key} vs {shard.shard_id.key}")
        return key

    def to_local(self) -> torch.Tensor:
        """Fetch all shards via HTTP, concatenate along dim 0."""
        if self._data is not None:
            return self._data

        # Fetch all shards first
        tensors = self._fetch()

        # NOTE: we assume that dim 1 is sequence dim
        if len(tensors[0].shape) > 1:
            # Pad along dim 1
            max_len = max(t.shape[1] for t in tensors)
            padded_tensors = []
            for t in tensors:
                if t.shape[1] < max_len:
                    pad_size = max_len - t.shape[1]
                    pad_tensor = torch.zeros(
                        (t.shape[0], pad_size, *t.shape[2:]),
                        dtype=t.dtype,
                    )
                    t = torch.cat([t, pad_tensor], dim=1)
                padded_tensors.append(t)
            tensors = padded_tensors

        self._data = torch.cat(tensors, dim=0)
        return self._data

    def _fetch(self):
        """Fetch all shards synchronously."""

        async def _fetch_all():
            async with aiohttp.ClientSession() as session:
                return await asyncio.gather(
                    *[
                        RTensor._fetch_tensor(session, s.shard_id, s.node_addr)
                        for s in self.shards
                    ]
                )

        return asyncio.run(_fetch_all())

    @staticmethod
    async def _fetch_tensor(session, shard_id: ShardId, node_addr: str) -> torch.Tensor:
        # Avoid circular import
        from areal.scheduler.rpc.serialization import deserialize_value

        url = f"http://{node_addr}/data/{shard_id}"
        async with session.get(url) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Failed to fetch shard from {url}: {resp.status}")
            data_bytes = await resp.read()
            serialized_data = orjson.loads(data_bytes)
            return deserialize_value(serialized_data, fetch_remote=False)

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, task_id: str, key: str, node_addr: str):
        # Called by inference engine, which produces individual outputs
        tensor = tensor.detach().cpu()
        info = ShardInfo(
            shard_id=ShardId(task_id=task_id, key=key),
            node_addr=node_addr,
            shape=list(tensor.shape),
            dtype=str(tensor.dtype),
        )
        return cls(shards=[info], _data=tensor)

    @classmethod
    def from_batched(
        cls,
        batch_tensor: torch.Tensor,
        layout: BatchLayout,
        key: str,
        node_addr: str,
    ):
        # Called by train engine, which produces batched output
        batch_tensor = batch_tensor.detach().cpu()
        offsets = np.cumsum([0] + [shard.size for shard in layout.layout])
        # no clone here because they are read-only slices
        tensors = [batch_tensor[a:b] for a, b in zip(offsets[:-1], offsets[1:])]

        shards = []
        for tensor, shard_layout in zip(tensors, layout.layout):
            info = ShardInfo(
                shard_id=ShardId(task_id=shard_layout.shard_id.task_id, key=key),
                node_addr=node_addr,
                shape=list(tensor.shape),
                dtype=str(batch_tensor.dtype),
            )
            shards.append(info)
        return cls(shards=shards, _data=batch_tensor)

    @staticmethod
    def cat(rtensors: list["RTensor"], dim: int = 0) -> "RTensor":
        """Concatenate RTensors along existing dimension."""
        if not rtensors:
            raise ValueError("Cannot concatenate empty list of RTensors")
        if dim != 0:
            raise NotImplementedError("Only dim=0 concatenation supported")
        return RTensor(shards=[shard for r in rtensors for shard in r.shards])

    def store_shards(self, storage_dict, storage_lock):
        assert self._data is not None, "No data to store for RTensor"

        with storage_lock:
            offsets = np.cumsum([0] + [s.shape[0] for s in self.shards])
            for shard_info, s, e in zip(self.shards, offsets[:-1], offsets[1:]):
                storage_dict[shard_info.shard_id] = self._data[s:e]

    @staticmethod
    def from_engine_output(
        output: Any,
        input_layouts: Any,
        key: str | None,
        task_id: int | None,
        node_addr: str,
        storage_dict,
        storage_lock,
    ):
        if output is None:
            return None

        if isinstance(output, torch.Tensor):
            # Determine if batched from input layouts
            batch_layout = BatchLayout.find_in_structure(input_layouts)

            if batch_layout:
                assert task_id is None
                rtensor = RTensor.from_batched(
                    output, batch_layout, key=key, node_addr=node_addr
                )
            else:
                assert task_id is not None
                rtensor = RTensor.from_tensor(
                    output, task_id=task_id, key=key, node_addr=node_addr
                )

            rtensor.store_shards(storage_dict, storage_lock)
            return rtensor

        if isinstance(output, dict):
            return {
                k: RTensor.from_engine_output(
                    v, input_layouts, k, task_id, node_addr, storage_dict, storage_lock
                )
                for k, v in output.items()
            }

        if isinstance(output, list):
            return [
                RTensor.from_engine_output(
                    item,
                    input_layouts,
                    key,
                    task_id,
                    node_addr,
                    storage_dict,
                    storage_lock,
                )
                for item in output
            ]

        if isinstance(output, tuple):
            return tuple(
                RTensor.from_engine_output(
                    item,
                    input_layouts,
                    key,
                    task_id,
                    node_addr,
                    storage_dict,
                    storage_lock,
                )
                for item in output
            )

        return output
