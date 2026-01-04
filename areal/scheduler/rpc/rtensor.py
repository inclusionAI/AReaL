from __future__ import annotations

import asyncio
import uuid
from collections import defaultdict
from dataclasses import dataclass
from threading import Lock
from typing import Any, Protocol

import aiohttp
import numpy as np
import orjson
import ray
import torch

from areal.utils.datapack import ffd_allocate, flat2d


class TensorBackend(Protocol):
    def fetch(self, shards: list[TensorShardInfo]) -> list[torch.Tensor]:
        """Fetch tensors for the given shards.

        Parameters
        ----------
        shards : list[TensorShardInfo]
            List of shard metadata to fetch

        Returns
        -------
        list[torch.Tensor]
            List of tensors corresponding to each shard
        """
        ...

    def store(self, tensor: torch.Tensor) -> Any:
        """Store a tensor and return its shard ID.

        Parameters
        ----------
        tensor : torch.Tensor
            The tensor to store

        Returns
        -------
        Any
            Shard ID (str for HTTP backend, ray.ObjectRef for Ray backend)
        """
        ...

    async def delete(self, node_addr: str, shard_ids: list[Any]) -> None:
        """Delete shards from storage.

        Parameters
        ----------
        node_addr : str
            The node address where shards are stored
        shard_ids : list[Any]
            List of shard IDs to delete
        """
        ...


@dataclass
class TensorShardInfo:
    """Metadata for a single shard of an RTensor.

    This is a pure data class containing only shard metadata.
    All storage operations are handled by TensorBackend implementations.

    Attributes
    ----------
    size : int
        Batch size (shape[0]) of this shard
    seqlens : list[int]
        Sequence lengths of this shard (from attention_mask)
    shard_id : Any
        Unique identifier for the shard (str for HTTP, ray.ObjectRef for Ray)
    node_addr : str
        Network address where shard is stored (empty for Ray backend)
    """

    size: int
    seqlens: list[int]
    shard_id: Any
    node_addr: str


class HttpTensorBackend:
    def fetch(self, shards: list[TensorShardInfo]) -> list[torch.Tensor]:
        """Fetch all shards via HTTP."""

        async def _fetch_all():
            async with aiohttp.ClientSession() as session:
                return await asyncio.gather(
                    *[
                        self._fetch_tensor(session, s.shard_id, s.node_addr)
                        for s in shards
                    ]
                )

        return asyncio.run(_fetch_all())

    async def _fetch_tensor(
        self, session: aiohttp.ClientSession, shard_id: str, node_addr: str
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

    def store(self, tensor: torch.Tensor) -> str:
        """Store tensor in local storage, return UUID shard_id."""
        shard_id = str(uuid.uuid4())
        _store_local(shard_id, tensor)
        return shard_id

    async def delete(self, node_addr: str, shard_ids: list[str]) -> None:
        """Delete shards via HTTP DELETE request."""
        async with aiohttp.ClientSession() as session:
            async with session.delete(
                f"http://{node_addr}/data/clear", json={"shard_ids": shard_ids}
            ) as resp:
                if resp.status == 200:
                    await resp.json()


class RayTensorBackend:
    def fetch(self, shards: list[TensorShardInfo]) -> list[torch.Tensor]:
        """Fetch all shards from Ray object store."""
        return ray.get([s.shard_id for s in shards])

    def store(self, tensor: torch.Tensor) -> ray.ObjectRef:
        """Store tensor in Ray object store, return ObjectRef."""
        return ray.put(tensor)

    async def delete(self, _node_addr: str, shard_ids: list[ray.ObjectRef]) -> None:
        """Free objects from Ray object store."""
        ray.internal.free(shard_ids)


_backend: TensorBackend | None = None


def get_backend() -> TensorBackend:
    global _backend
    if _backend is None:
        if ray.is_initialized():
            _backend = RayTensorBackend()
        else:
            _backend = HttpTensorBackend()
    return _backend


def set_backend(backend: TensorBackend | None) -> None:
    global _backend
    _backend = backend


@dataclass
class RTensor:
    shards: list[TensorShardInfo]
    data: torch.Tensor

    def to_local(self) -> torch.Tensor:
        if not self.data.is_meta:
            return self.data
        # Fetch all shards first
        tensors = self._fetch()
        self.data = _pad_cat_dim0(tensors)
        return self.data

    def _fetch(self) -> list[torch.Tensor]:
        return get_backend().fetch(self.shards)

    @staticmethod
    def split_tensor(batch_tensor: torch.Tensor, layout: RTensor) -> list[torch.Tensor]:
        offsets = np.cumsum([0] + [shard.size for shard in layout.shards])
        if offsets[-1] != batch_tensor.shape[0]:
            raise ValueError(
                f"Batched tensor size {batch_tensor.shape[0]} does not match "
                f"layout total size {offsets[-1]}"
            )
        # no clone here because they are read-only slices
        return [batch_tensor[a:b] for a, b in zip(offsets[:-1], offsets[1:])]

    def split(self) -> list[RTensor]:
        tensors = RTensor.split_tensor(self.data, self)
        return [RTensor(shards=[s], data=t) for s, t in zip(self.shards, tensors)]

    @classmethod
    def from_batched(
        cls, batch_tensor: torch.Tensor, layout: RTensor, node_addr: str
    ) -> RTensor:
        if not batch_tensor.is_cpu and not batch_tensor.is_meta:
            raise ValueError("RTensor shards must be on CPU or meta device")

        tensors = cls.split_tensor(batch_tensor, layout)
        backend = get_backend()

        shards = []
        for tensor, shard_info in zip(tensors, layout.shards):
            # Truncate at the maximum sequence length to prevent over-padding
            if tensor.ndim > 1:
                tensor = tensor[:, : max(shard_info.seqlens)]
            # Store locally
            shard_id = backend.store(tensor)
            info = TensorShardInfo(
                size=shard_info.size,
                seqlens=shard_info.seqlens.copy(),
                shard_id=shard_id,
                node_addr=node_addr,
            )
            shards.append(info)

        return cls(shards=shards, data=batch_tensor.to("meta"))

    @staticmethod
    def cat(rtensors: list[RTensor | torch.Tensor], dim: int = 0) -> RTensor:
        n_tensors = len(rtensors)
        if n_tensors == 0:
            return RTensor(shards=[], data=torch.tensor([]).to("meta"))
        n_rtensors = len([x for x in rtensors if isinstance(x, RTensor)])

        # All RTensors
        if n_tensors == n_rtensors:
            if dim != 0:
                raise ValueError(
                    "RTensor.cat for multiple RTensors only supports dim=0"
                )
            if any(t.data is None for t in rtensors):
                raise RuntimeError("Cannot concat rtensors with None data")
            return RTensor(
                shards=[shard for r in rtensors for shard in r.shards],
                data=_pad_cat_dim0([r.data for r in rtensors]),
            )

        # hybrid RTensor and normal tensors
        if n_rtensors != 1:
            raise ValueError(
                "RTensor.cat only support concatenating a single RTensor "
                "with other torch.Tensor"
            )
        rt = [x for x in rtensors if isinstance(x, RTensor)][0]
        return RTensor(
            shards=rt.shards,
            data=torch.cat(
                [r.data if isinstance(r, RTensor) else r for r in rtensors], dim=dim
            ),
        )

    @staticmethod
    def extract_layout(obj: Any, layouts: Any, node_addr: str | None) -> RTensor | None:
        """Extract layout RTensor from object or create from attention_mask.

        Parameters
        ----------
        obj : Any
            Object potentially containing tensors
        layouts : Any
            Layouts potentially containing RTensor
        node_addr : str | None
            Node address for creating new shard info

        Returns
        -------
        RTensor | None
            Layout RTensor or None if not found
        """
        # Determine if batched from input layouts
        layout_rtensor = _find_in_structure(layouts, RTensor)
        result_tensor = _find_in_structure(obj, torch.Tensor)
        if layout_rtensor is None and result_tensor is not None:
            if not isinstance(obj, dict):
                raise RuntimeError(
                    "When input does not contain RTensor, "
                    "we expect to extract layouts from a dict batch "
                    f"returned by InferenceEngine. Get obj: {obj}, "
                    f"input layouts: {layouts}."
                )
            attn_mask = obj.get("attention_mask", None)
            if attn_mask is None:
                raise RuntimeError("`attention_mask` is not found")
            assert node_addr is not None
            shard = TensorShardInfo(
                size=attn_mask.shape[0],
                seqlens=[int(am.sum()) for am in attn_mask],
                shard_id="",
                node_addr=node_addr,
            )

            layout_rtensor = RTensor(
                shards=[shard],
                data=torch.empty_like(attn_mask, device="meta"),
            )
        return layout_rtensor

    @staticmethod
    def remotize(obj: Any, layout: RTensor, node_addr: str) -> Any:
        """Convert tensors to RTensors in nested structures.

        Parameters
        ----------
        obj : Any
            Object potentially containing tensors
        layout : RTensor
            Layout for creating RTensors
        node_addr : str
            Node address for shard storage

        Returns
        -------
        Any
            Object with tensors converted to RTensors
        """
        if isinstance(obj, torch.Tensor):
            return RTensor.from_batched(
                obj.detach().cpu(), layout=layout, node_addr=node_addr
            )

        if isinstance(obj, dict):
            return {
                k: RTensor.remotize(obj=v, layout=layout, node_addr=node_addr)
                for k, v in obj.items()
            }

        if isinstance(obj, list):
            return [
                RTensor.remotize(obj=item, layout=layout, node_addr=node_addr)
                for item in obj
            ]

        if isinstance(obj, tuple):
            return tuple(
                RTensor.remotize(obj=item, layout=layout, node_addr=node_addr)
                for item in obj
            )

        return obj

    @staticmethod
    def localize(obj: Any) -> Any:
        """Convert RTensors to local tensors in nested structures.

        Inverse of remotize() - fetches remote data and converts to local tensors.

        Parameters
        ----------
        obj : Any
            Object potentially containing RTensors

        Returns
        -------
        Any
            Object with RTensors converted to local tensors
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
    ) -> tuple[list[Any], list[list[int]] | None]:
        """Split data for data parallel processing.

        Parameters
        ----------
        obj : Any
            Object to split
        dp_size : int
            Number of data parallel groups
        group_indices : list[list[int]] | None
            Pre-computed group assignments (computed if None)

        Returns
        -------
        tuple[list[Any], list[list[int]] | None]
            Split objects and group indices
        """
        if group_indices is None:
            layout_rtensor = _find_in_structure(obj, RTensor)
            if layout_rtensor is not None:
                seqlens = [sum(s.seqlens) for s in layout_rtensor.shards]
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
            return split_rtensors, group_indices

        if isinstance(obj, dict):
            # Split each value, return list of dicts
            split_values = {
                k: RTensor.data_parallel_dispatch(v, dp_size, group_indices)[0]
                for k, v in obj.items()
            }
            return [
                {k: split_values[k][i] for k in obj.keys()} for i in range(dp_size)
            ], group_indices

        if isinstance(obj, list):
            # Split each element
            split_elements = [
                RTensor.data_parallel_dispatch(elem, dp_size, group_indices)[0]
                for elem in obj
            ]
            return [
                [split_elements[j][i] for j in range(len(obj))] for i in range(dp_size)
            ], group_indices

        if isinstance(obj, tuple):
            # Split each element
            split_elements = [
                RTensor.data_parallel_dispatch(elem, dp_size, group_indices)[0]
                for elem in obj
            ]
            return [
                tuple(split_elements[j][i] for j in range(len(obj)))
                for i in range(dp_size)
            ], group_indices

        # Non-RTensor objects: replicate to all groups
        return [obj] * dp_size, group_indices

    @staticmethod
    def data_parallel_merge(
        results: list[Any], group_indices: list[list[int]] | None
    ) -> Any:
        """Merge results from data parallel processing.

        Parameters
        ----------
        results : list[Any]
            Results from each DP group
        group_indices : list[list[int]] | None
            Group assignments used during dispatch

        Returns
        -------
        Any
            Merged result with original ordering restored
        """
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
            assert group_indices is not None
            rtensors = flat2d([r.split() for r in results])
            indices = flat2d(group_indices)
            assert len(rtensors) == len(indices), (len(rtensors), len(indices))
            inv_indices = np.zeros(len(indices), dtype=np.int64)
            inv_indices[indices] = np.arange(len(indices))
            return RTensor.cat([rtensors[i] for i in inv_indices])

        if isinstance(first, dict):
            merged = {}
            for key in first.keys():
                values = [r[key] for r in results]
                merged[key] = RTensor.data_parallel_merge(
                    values, group_indices=group_indices
                )
            return merged

        if isinstance(first, list):
            merged = []
            for i in range(len(first)):
                elements = [r[i] for r in results]
                merged.append(
                    RTensor.data_parallel_merge(elements, group_indices=group_indices)
                )
            return merged

        if isinstance(first, tuple):
            merged = []
            for i in range(len(first)):
                elements = [r[i] for r in results]
                merged.append(
                    RTensor.data_parallel_merge(elements, group_indices=group_indices)
                )
            return tuple(merged)

        # Scalars: return first (assume synchronized)
        return first

    @staticmethod
    def collect_shards(obj: Any) -> dict[str, list[Any]]:
        """Collect shard IDs grouped by node address from nested structure.

        Parameters
        ----------
        obj : Any
            Object potentially containing RTensors

        Returns
        -------
        dict[str, list[Any]]
            Mapping of node_addr -> list of shard_ids
        """
        shards_by_node: dict[str, list[Any]] = {}

        def _collect(o: Any) -> None:
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

    @staticmethod
    async def clear_node(node_addr: str, shard_ids: list[Any]) -> None:
        """Clear shards from a node.

        Parameters
        ----------
        node_addr : str
            The node address
        shard_ids : list[Any]
            List of shard IDs to delete
        """
        await get_backend().delete(node_addr, shard_ids)

    @property
    def shape(self) -> torch.Size:
        """Shape of the data tensor."""
        return self.data.shape

    @property
    def dtype(self) -> torch.dtype:
        """Data type of the tensor."""
        return self.data.dtype

    @property
    def device(self) -> torch.device:
        """Device of the tensor."""
        return self.data.device

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self.data.ndim

    @classmethod
    def __torch_function__(
        cls,
        func: Any,
        _types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> RTensor:
        if kwargs is None:
            kwargs = {}

        if func is torch.cat:
            return RTensor.cat(*args, **kwargs)

        raise NotImplementedError(f"RTensor does not implement torch function {func}")


def _pad_cat_dim0(tensors: list[torch.Tensor]) -> torch.Tensor:
    # Get the maximum shape for dims 1 to N-1
    shape = [0 for _ in range(tensors[0].ndim - 1)]
    for t in tensors:
        if t.ndim != len(shape) + 1:
            raise ValueError(
                f"Shard dimension mismatch: expected {len(shape) + 1}, got {t.ndim}"
            )
        for i in range(1, t.ndim):
            shape[i - 1] = max(shape[i - 1], t.shape[i])

    # Pad tensors
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


def _find_in_structure(obj: Any, type_: type) -> Any | None:
    if isinstance(obj, type_):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            result = _find_in_structure(v, type_)
            if result is not None:
                return result
    if isinstance(obj, (tuple, list)):
        for item in obj:
            result = _find_in_structure(item, type_)
            if result is not None:
                return result
    return None


# =============================================================================
# Local Storage (used by HttpTensorBackend)
# =============================================================================

# Global tensor data storage for distributed batch
# Storage: shard_id -> Tensor
_storage: dict[str, torch.Tensor] = {}
_storage_lock = Lock()
_storage_stats: dict[str, int] = defaultdict(int)


def _store_local(shard_id: str, tensor: torch.Tensor) -> None:
    """Store a tensor shard in local storage (internal use)."""
    global _storage, _storage_lock, _storage_stats
    with _storage_lock:
        _storage[shard_id] = tensor
        _storage_stats[shard_id] = tensor.nbytes


def store(shard_id: str, tensor: torch.Tensor) -> None:
    """Store a tensor shard in global storage."""
    _store_local(shard_id, tensor)


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
    """Get current storage stats."""
    global _storage_stats, _storage_lock, _storage
    with _storage_lock:
        return dict(num_tensors=len(_storage), total_bytes=sum(_storage_stats.values()))
