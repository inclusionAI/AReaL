from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass
from threading import Lock
from typing import Any, Protocol

import aiohttp
import orjson
import ray
import torch

from areal.infra.utils.concurrent import run_async_task


class TensorBackend(Protocol):
    def fetch(self, shard: TensorShardInfo) -> torch.Tensor:
        """Fetch tensor for the given shard.

        Parameters
        ----------
        shard : TensorShardInfo
            Shard metadata to fetch

        Returns
        -------
        torch.Tensor
            Tensor corresponding to the shard
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
    def fetch(self, shard: TensorShardInfo) -> torch.Tensor:
        """Fetch shard via HTTP."""

        async def _fetch():
            async with aiohttp.ClientSession() as session:
                return await self._fetch_tensor(
                    session, shard.shard_id, shard.node_addr
                )

        return run_async_task(_fetch)

    async def _fetch_tensor(
        self, session: aiohttp.ClientSession, shard_id: str, node_addr: str
    ) -> torch.Tensor:
        # Avoid circular import
        from areal.infra.rpc.serialization import deserialize_value

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
    def fetch(self, shard: TensorShardInfo) -> torch.Tensor:
        """Fetch shard from Ray object store."""
        return ray.get(shard.shard_id)

    def store(self, tensor: torch.Tensor) -> ray.ObjectRef:
        """Store tensor in Ray object store, return ObjectRef."""
        return ray.put(tensor)

    async def delete(self, node_addr: str, shard_ids: list[Any]) -> None:
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
    shard: TensorShardInfo
    data: torch.Tensor

    def to_local(self) -> torch.Tensor:
        if not self.data.is_meta:
            return self.data
        self.data = get_backend().fetch(self.shard)
        return self.data

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
                shard=shard,
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
            # Store the tensor directly as a single shard (from_batched is removed)
            tensor = obj.detach().cpu()
            shard_id = get_backend().store(tensor)
            shard = TensorShardInfo(
                size=tensor.shape[0],
                seqlens=layout.shard.seqlens,
                shard_id=shard_id,
                node_addr=node_addr,
            )
            return RTensor(shard=shard, data=tensor.to("meta"))

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
                if o.shard.node_addr not in shards_by_node:
                    shards_by_node[o.shard.node_addr] = []
                shards_by_node[o.shard.node_addr].append(o.shard.shard_id)
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
