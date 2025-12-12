import asyncio
from collections import defaultdict
from dataclasses import dataclass
from threading import Lock
from typing import Any

import aiohttp
import numpy as np
import orjson
import torch


@dataclass(frozen=True)
class TensorShardId:
    """Unique identifier for a tensor shard."""

    task_id: str
    tensor_name: str

    def __str__(self) -> str:
        """Convert to string format: task_id:tensor_name"""
        return f"{self.task_id}:{self.tensor_name}"

    @classmethod
    def from_string(cls, s: str) -> "TensorShardId":
        """Parse from string format: task_id:tensor_name"""
        parts = s.split(":", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid TensorShardId string: {s}")
        return cls(task_id=parts[0], tensor_name=parts[1])


@dataclass
class TensorShardInfo:
    """Metadata for a single shard of an RTensor."""

    shard_id: TensorShardId
    node_addr: str
    size: int  # Batch size (shape[0]) of this shard


@dataclass
class RTensor:
    """Single tensor distributed as CPU shards across nodes."""

    shards: list[TensorShardInfo]
    data: torch.Tensor | None

    def get_name(self) -> str:
        """Get tensor_name from first shard (all shards must have same tensor_name)."""
        if not self.shards:
            raise ValueError("Cannot get tensor_name from empty RTensor")
        tensor_name = self.shards[0].shard_id.tensor_name
        # Validate consistency
        for shard in self.shards[1:]:
            if shard.shard_id.tensor_name != tensor_name:
                raise ValueError(
                    f"Inconsistent tensor_names: {tensor_name} vs {shard.shard_id.tensor_name}"
                )
        return tensor_name

    def to_local(self) -> torch.Tensor:
        """Fetch all shards via HTTP, concatenate along dim 0."""
        if not self.data.is_meta:
            return self.data

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

        self.data = torch.cat(tensors, dim=0)
        return self.data

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
    async def _fetch_tensor(
        session: aiohttp.ClientSession, shard_id: TensorShardId, node_addr: str
    ) -> torch.Tensor:
        # Avoid circular import
        from areal.scheduler.rpc.serialization import deserialize_value

        url = f"http://{node_addr}/data/{shard_id}"
        async with session.get(url) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Failed to fetch shard from {url}: {resp.status}")
            data_bytes = await resp.read()
            serialized_data = orjson.loads(data_bytes)
            return deserialize_value(serialized_data)

    @classmethod
    def from_single(cls, tensor: torch.Tensor, shard_id: TensorShardId, node_addr: str):
        # Called by inference engine, which produces individual objs
        if not tensor.is_cpu:
            raise ValueError("RTensor shards must be on CPU")
        info = TensorShardInfo(
            shard_id=shard_id,
            node_addr=node_addr,
            size=tensor.shape[0],
        )
        # Store locally
        store(shard_id, tensor)
        return cls(shards=[info], data=tensor.to("meta"))

    @classmethod
    def from_batched(
        cls,
        batch_tensor: torch.Tensor,
        layout: "RTensor",
        tensor_name: str,
        node_addr: str,
    ):
        # Called by train engine, which produces batched obj
        batch_tensor = batch_tensor.detach().cpu()
        offsets = np.cumsum([0] + [shard.size for shard in layout.shards])
        # no clone here because they are read-only slices
        tensors = [batch_tensor[a:b] for a, b in zip(offsets[:-1], offsets[1:])]

        global _storage, _storage_lock
        shards = []
        for tensor, shard_info in zip(tensors, layout.shards):
            sid = TensorShardId(
                task_id=shard_info.shard_id.task_id, tensor_name=tensor_name
            )
            info = TensorShardInfo(
                shard_id=sid,
                node_addr=node_addr,
                size=tensor.shape[0],
            )
            shards.append(info)
            # Store locally
            store(sid, tensor)
        return cls(shards=shards, data=batch_tensor.to("meta"))

    @staticmethod
    def cat(rtensors: list["RTensor"]) -> "RTensor":
        """Concatenate RTensors along existing dimension."""
        if not rtensors:
            return RTensor(shards=[], data=torch.tensor([]).to("meta"))
        return RTensor(
            shards=[shard for r in rtensors for shard in r.shards],
            data=torch.cat([r.data for r in rtensors]),
        )

    @staticmethod
    def find_in_structure(obj) -> "RTensor":
        """Find first RTensor in a nested structure."""
        if isinstance(obj, RTensor):
            return obj
        if isinstance(obj, dict):
            for v in obj.values():
                result = RTensor.find_in_structure(v)
                if result:
                    return result
        if isinstance(obj, list):
            for item in obj:
                result = RTensor.find_in_structure(item)
                if result:
                    return result
        return None

    @staticmethod
    def rtensorize(
        obj: Any,
        layouts: Any,
        node_addr: str,
        tensor_name: str | None = None,
        task_id: int | None = None,
    ) -> Any:
        if isinstance(obj, torch.Tensor):
            if tensor_name is None:
                raise ValueError(
                    "tensor_name must be provided when rtensorizing a tensor"
                )

            # Determine if batched from input layouts
            layout_rtensor = RTensor.find_in_structure(layouts)

            if layout_rtensor:
                if task_id is not None:
                    raise ValueError(
                        "task_id must be None when using batched layout (e.g., TrainEngine outputs)"
                    )
                return RTensor.from_batched(
                    obj,
                    layout=layout_rtensor,
                    tensor_name=tensor_name,
                    node_addr=node_addr,
                )

            if task_id is None:
                raise ValueError(
                    "task_id must be provided when using single shard layout (e.g., InferenceEngine outputs)"
                )
            shard_id = TensorShardId(task_id=task_id, tensor_name=tensor_name)
            return RTensor.from_single(obj, shard_id=shard_id, node_addr=node_addr)

        if isinstance(obj, dict):
            return {
                k: RTensor.rtensorize(
                    obj=v,
                    layouts=layouts,
                    tensor_name=k,
                    task_id=task_id,
                    node_addr=node_addr,
                )
                for k, v in obj.items()
            }

        if isinstance(obj, list):
            return [
                RTensor.rtensorize(
                    obj=item,
                    layouts=layouts,
                    tensor_name=tensor_name,
                    task_id=task_id,
                    node_addr=node_addr,
                )
                for item in obj
            ]

        if isinstance(obj, tuple):
            return tuple(
                RTensor.rtensorize(
                    obj=item,
                    layouts=layouts,
                    tensor_name=tensor_name,
                    task_id=task_id,
                    node_addr=node_addr,
                )
                for item in obj
            )

        return obj

    @staticmethod
    def localize(obj: Any) -> Any:
        """Convert RTensors to local tensors in nested structures.

        Inverse of rtensorize() - fetches remote data and converts to local tensors.
        """
        if isinstance(obj, RTensor):
            return obj.to_local()

        if isinstance(obj, dict):
            return {k: RTensor.localize(v) for k, v in obj.items()}

        if isinstance(obj, list):
            return [RTensor.localize(item) for item in obj]

        if isinstance(obj, tuple):
            return tuple(RTensor.localize(item) for item in obj)

        return obj


# Global tensor data storage for distributed batch
# Storage: shard_id -> dict[str, Tensor]
_storage: dict[TensorShardId, RTensor] = {}
_storage_lock = Lock()
_storage_stats: dict[TensorShardId, int] = defaultdict(int)


def store(shard_id: TensorShardId, tensor: torch.Tensor):
    """Store a tensor shard in global storage."""
    global _storage, _storage_lock, _storage_stats
    with _storage_lock:
        _storage[shard_id] = tensor
        _storage_stats[shard_id] = tensor.nbytes


def fetch(shard_id: TensorShardId) -> torch.Tensor:
    """Retrieve a tensor shard from global storage."""
    global _storage, _storage_lock
    with _storage_lock:
        tensor = _storage.get(shard_id)
        if tensor is None:
            raise KeyError(f"Shard {shard_id} not found in storage")
        return tensor


def remove(shard_id: TensorShardId) -> int:
    """Remove a tensor shard from global storage."""
    global _storage, _storage_lock, _storage_stats
    with _storage_lock:
        if shard_id in _storage:
            del _storage[shard_id]
            del _storage_stats[shard_id]
            return 1
        return 0


def storage_stats() -> dict[str, int]:
    """Get current storage stats: shard_id -> size in bytes."""
    global _storage_stats, _storage_lock, _storage
    with _storage_lock:
        return dict(num_tensors=len(_storage), total_bytes=sum(_storage_stats.values()))
