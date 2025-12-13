import asyncio
import uuid
from collections import defaultdict
from dataclasses import dataclass
from threading import Lock
from typing import Any

import aiohttp
import numpy as np
import orjson
import torch

from areal.utils.datapack import ffd_allocate


@dataclass
class TensorShardInfo:
    """Metadata for a single shard of an RTensor."""

    shard_id: str
    node_addr: str
    size: int  # Batch size (shape[0]) of this shard
    seqlen: int  # Cumulative sequence length of this shard (attn_mask.sum())


@dataclass
class RTensor:
    """Single tensor distributed as CPU shards across nodes."""

    shards: list[TensorShardInfo]
    data: torch.Tensor | None

    def to_local(self) -> torch.Tensor:
        """Fetch all shards via HTTP, concatenate along dim 0."""
        if not self.data.is_meta:
            return self.data
        # Fetch all shards first
        tensors = self._fetch()
        self.data = _pad_cat_dim0(tensors)
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
        session: aiohttp.ClientSession, shard_id: str, node_addr: str
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

    @staticmethod
    def split_tensor(
        batch_tensor: torch.Tensor, layout: "RTensor"
    ) -> list[torch.Tensor]:
        offsets = np.cumsum([0] + [shard.size for shard in layout.shards])
        if offsets[-1] != batch_tensor.shape[0]:
            raise ValueError(
                f"Batched tensor size {batch_tensor.shape[0]} does not match layout total size {offsets[-1]}"
            )
        # no clone here because they are read-only slices
        return [batch_tensor[a:b] for a, b in zip(offsets[:-1], offsets[1:])]

    @classmethod
    def from_batched(cls, batch_tensor: torch.Tensor, layout: "RTensor"):
        if not batch_tensor.is_cpu:
            raise ValueError("RTensor shards must be on CPU")

        tensors = cls.split_tensor(batch_tensor, layout)

        shards = []
        for tensor, shard_info in zip(tensors, layout.shards):
            sid = str(uuid.uuid4())
            info = TensorShardInfo(
                shard_id=sid,
                node_addr=shard_info.node_addr,
                size=shard_info.size,
                seqlen=shard_info.seqlen,
            )
            shards.append(info)

            # Store locally
            store(sid, tensor)

        return cls(shards=layout.shards, data=batch_tensor.to("meta"))

    @staticmethod
    def cat(rtensors: list["RTensor"]) -> "RTensor":
        """Concatenate RTensors along existing dimension."""
        if not rtensors:
            return RTensor(shards=[], data=torch.tensor([]).to("meta"))
        return RTensor(
            shards=[shard for r in rtensors for shard in r.shards],
            data=_pad_cat_dim0([r.data for r in rtensors]),
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
        if isinstance(obj, (tuple, list)):
            for item in obj:
                result = RTensor.find_in_structure(item)
                if result:
                    return result
        return None

    @staticmethod
    def rtensorize(obj: Any, layouts: Any, node_addr: str | None = None) -> Any:
        # Determine if batched from input layouts
        layout_rtensor = RTensor.find_in_structure(layouts)
        if layout_rtensor is None:
            if not isinstance(obj, dict):
                raise RuntimeError(
                    "When input does not contain RTensor, "
                    "we expect to extract layouts from a dict batch "
                    "returned by InferenceEngine"
                )
            attn_mask = obj.get("attention_mask", None)
            if attn_mask is None:
                raise RuntimeError("`attention_mask` is not found")
            layout_rtensor = RTensor(
                shards=[
                    TensorShardInfo(
                        shard_id="",
                        node_addr=node_addr,
                        size=attn_mask.shape[0],
                        seqlen=int(attn_mask.sum()),
                    )
                ],
                data=None,
            )

        if isinstance(obj, torch.Tensor):
            return RTensor.from_batched(obj, layout=layout_rtensor)

        if isinstance(obj, dict):
            return {
                k: RTensor.rtensorize(obj=v, layouts=layout_rtensor)
                for k, v in obj.items()
            }

        if isinstance(obj, list):
            return [
                RTensor.rtensorize(obj=item, layouts=layout_rtensor) for item in obj
            ]

        if isinstance(obj, tuple):
            return tuple(
                RTensor.rtensorize(obj=item, layouts=layout_rtensor) for item in obj
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

    @staticmethod
    def data_parallel_dispatch(
        obj: Any, dp_size: int, group_indices: list[list[int]] | None = None
    ) -> list[Any]:
        if group_indices is None:
            layout_rtensor = RTensor.find_in_structure(obj)
            if layout_rtensor is not None:
                seqlens = [s.seqlen for s in layout_rtensor.shards]
                # Use FFD to allocate shards to DP groups
                group_indices = ffd_allocate(
                    seqlens, capacity=int(1e12), min_groups=dp_size
                )
            # else: no RTensors found, will replicate scalars without group_indices

        if isinstance(obj, RTensor):
            tensors = RTensor.split_tensor(obj.data, obj)
            # Split shards according to group assignments
            split_rtensors = []
            for group_idxs in group_indices:
                # Collect shards for this group
                group_shards = [obj.shards[i] for i in group_idxs]
                group_data = _pad_cat_dim0([tensors[i] for i in group_idxs])
                split_rtensors.append(RTensor(shards=group_shards, data=group_data))
            return split_rtensors

        if isinstance(obj, dict):
            # Split each value, return list of dicts
            split_values = {
                k: RTensor.data_parallel_dispatch(v, dp_size, group_indices)
                for k, v in obj.items()
            }
            return [{k: split_values[k][i] for k in obj.keys()} for i in range(dp_size)]

        if isinstance(obj, list):
            # Split each element
            split_elements = [
                RTensor.data_parallel_dispatch(elem, dp_size, group_indices)
                for elem in obj
            ]
            return [
                [split_elements[j][i] for j in range(len(obj))] for i in range(dp_size)
            ]

        if isinstance(obj, tuple):
            # Split each element
            split_elements = [
                RTensor.data_parallel_dispatch(elem, dp_size, group_indices)
                for elem in obj
            ]
            return [
                tuple(split_elements[j][i] for j in range(len(obj)))
                for i in range(dp_size)
            ]

        # Non-RTensor objects: replicate to all groups
        return [obj] * dp_size

    @staticmethod
    def data_parallel_merge(results: list[Any]) -> Any:
        if not results:
            return None

        first = results[0]

        # Check for raw tensors - not allowed
        if isinstance(first, torch.Tensor):
            raise TypeError(
                "Regular tensors not allowed in merge - only RTensors. "
                "Engine outputs should be automatically converted to RTensors."
            )

        if isinstance(first, RTensor):
            return RTensor.cat(results)

        if isinstance(first, dict):
            merged = {}
            for key in first.keys():
                values = [r[key] for r in results]
                merged[key] = RTensor.data_parallel_merge(values)
            return merged

        if isinstance(first, list):
            merged = []
            for i in range(len(first)):
                elements = [r[i] for r in results]
                merged.append(RTensor.data_parallel_merge(elements))
            return merged

        if isinstance(first, tuple):
            merged = []
            for i in range(len(first)):
                elements = [r[i] for r in results]
                merged.append(RTensor.data_parallel_merge(elements))
            return tuple(merged)

        # Scalars: return first (assume synchronized)
        return first

    @staticmethod
    def collect_shards(obj: Any) -> dict[str, list[str]]:
        """Collect shard IDs grouped by node address from nested structure.

        Returns dict mapping node_addr -> list of shard_ids
        """
        shards_by_node = {}

        def _collect(o):
            if isinstance(o, RTensor):
                for shard in o.shards:
                    if shard.node_addr not in shards_by_node:
                        shards_by_node[shard.node_addr] = []
                    shards_by_node[shard.node_addr].append(shard.shard_id)
            elif isinstance(o, dict):
                for v in o.values():
                    _collect(v)
            elif isinstance(o, (list, tuple)):
                for item in o:
                    _collect(item)

        _collect(obj)
        return shards_by_node


def _pad_cat_dim0(tensors: list[torch.Tensor]) -> torch.Tensor:
    # Pad from dim 1 to dim N-1 if needed
    # Get the maximum shape
    shape = [0 for _ in range(tensors[0].ndim - 1)]
    for t in tensors:
        if t.ndim != len(shape) + 1:
            raise ValueError(
                f"Shard dimension mismatch: expected {len(shape) + 1}, got {t.ndim}"
            )
        for i in range(1, t.ndim):
            shape[i - 1] = max(shape[i - 1], t.shape[i])
    # Pad
    padded_tensors = []
    for t in tensors:
        pad_sizes = []
        for i in range(1, t.ndim):
            pad_size = shape[i - 1] - t.shape[i]
            pad_sizes.append(pad_size)
        if any(pad_sizes):
            pad = []
            for pad_size in reversed(pad_sizes):
                pad.extend([0, pad_size])
            pt = torch.nn.functional.pad(t, tuple(pad), "constant", 0)
            padded_tensors.append(pt)
            continue
        padded_tensors.append(t)

    return torch.cat(padded_tensors, dim=0)


# Global tensor data storage for distributed batch
# Storage: shard_id -> dict[str, Tensor]
_storage: dict[str, RTensor] = {}
_storage_lock = Lock()
_storage_stats: dict[str, int] = defaultdict(int)


def store(shard_id: str, tensor: torch.Tensor):
    """Store a tensor shard in global storage."""
    global _storage, _storage_lock, _storage_stats
    with _storage_lock:
        _storage[shard_id] = tensor
        _storage_stats[shard_id] = tensor.nbytes


def fetch(shard_id: str) -> torch.Tensor:
    """Retrieve a tensor shard from global storage."""
    global _storage, _storage_lock
    with _storage_lock:
        tensor = _storage.get(shard_id)
        if tensor is None:
            raise KeyError(f"Shard {shard_id} not found in storage")
        return tensor


def remove(shard_id: str) -> int:
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
