from __future__ import annotations

import asyncio
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
from areal.infra.utils.http import DEFAULT_REQUEST_TIMEOUT, get_default_connector
from areal.utils import logging

logger = logging.getLogger("HttpRTensor")


class RTensorBackend(Protocol):
    def fetch(self, shards: list[TensorShardInfo]) -> list[torch.Tensor]:
        """Fetch multiple tensors concurrently.

        Parameters
        ----------
        shards : list[TensorShardInfo]
            List of shard metadata to fetch

        Returns
        -------
        list[torch.Tensor]
            List of tensors in the same order as the input shards
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
    All storage operations are handled by RTensorBackend implementations.

    Attributes
    ----------
    shard_id : Any
        Unique identifier for the shard (str for HTTP, ray.ObjectRef for Ray)
    node_addr : str
        Network address where shard is stored (empty for Ray backend)
    """

    shard_id: Any
    node_addr: str


class HttpRTensorBackend:
    def _create_session(self) -> aiohttp.ClientSession:
        """Create a properly configured aiohttp session for large tensor transfers."""
        timeout = aiohttp.ClientTimeout(
            total=DEFAULT_REQUEST_TIMEOUT,
            sock_connect=DEFAULT_REQUEST_TIMEOUT,
            connect=DEFAULT_REQUEST_TIMEOUT,
        )
        return aiohttp.ClientSession(
            timeout=timeout,
            read_bufsize=10 * 1024 * 1024,  # 10MB buffer
            connector=get_default_connector(),
        )

    async def _fetch_tensor(
        self,
        session: aiohttp.ClientSession,
        shard_id: str,
        node_addr: str,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> torch.Tensor:
        # Avoid circular import
        from areal.infra.rpc.serialization import deserialize_value

        url = f"http://{node_addr}/data/{shard_id}"
        last_exception = None

        for attempt in range(max_retries):
            try:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        raise RuntimeError(
                            f"Failed to fetch shard from {url}: {resp.status}"
                        )
                    data_bytes = await resp.read()
                    serialized_data = orjson.loads(data_bytes)
                    return deserialize_value(serialized_data)
            except (TimeoutError, aiohttp.ClientError) as e:
                last_exception = e
                logger.warning(
                    "RTensor fetch from %s failed: %s: %s (attempt %d/%d)",
                    url,
                    e.__class__.__name__,
                    str(e),
                    attempt + 1,
                    max_retries,
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)

        raise RuntimeError(
            f"Failed to fetch shard from {url} after {max_retries} attempts. "
            f"Last error: {repr(last_exception)}"
        )

    def fetch(self, shards: list[TensorShardInfo]) -> list[torch.Tensor]:
        """Fetch multiple shards concurrently via HTTP using a single session."""
        if not shards:
            return []

        async def _fetch():
            async with self._create_session() as session:
                tasks = [
                    self._fetch_tensor(session, s.shard_id, s.node_addr) for s in shards
                ]
                return await asyncio.gather(*tasks)

        return run_async_task(_fetch)

    def store(self, tensor: torch.Tensor) -> str:
        """Store tensor in local storage, return UUID shard_id."""
        shard_id = str(uuid.uuid4())
        _store_local(shard_id, tensor)
        return shard_id

    async def delete(self, node_addr: str, shard_ids: list[str]) -> None:
        """Delete shards via HTTP DELETE request."""
        async with self._create_session() as session:
            async with session.delete(
                f"http://{node_addr}/data/clear", json={"shard_ids": shard_ids}
            ) as resp:
                if resp.status == 200:
                    await resp.json()


class RayRTensorBackend:
    def fetch(self, shards: list[TensorShardInfo]) -> list[torch.Tensor]:
        """Fetch multiple shards from Ray object store."""
        if not shards:
            return []
        return ray.get([s.shard_id for s in shards])

    def store(self, tensor: torch.Tensor) -> ray.ObjectRef:
        """Store tensor in Ray object store, return ObjectRef."""
        return ray.put(tensor)

    async def delete(self, node_addr: str, shard_ids: list[Any]) -> None:
        """Free objects from Ray object store."""
        ray.internal.free(shard_ids)


_backend: RTensorBackend | None = None


def get_backend() -> RTensorBackend:
    global _backend
    if _backend is None:
        if ray.is_initialized():
            _backend = RayRTensorBackend()
        else:
            _backend = HttpRTensorBackend()
    return _backend


def set_backend(backend: RTensorBackend | None) -> None:
    global _backend
    _backend = backend


@dataclass
class RTensor:
    shard: TensorShardInfo
    data: torch.Tensor

    def to_local(self) -> torch.Tensor:
        if not self.data.is_meta:
            return self.data
        self.data = get_backend().fetch([self.shard])[0]
        return self.data

    @staticmethod
    def remotize(obj: Any, node_addr: str) -> Any:
        """Convert tensors to RTensors in nested structures.

        For dict objects that look like trajectory dicts (contain attention_mask),
        trailing padding is trimmed before storage to keep each RTensor compact.

        Parameters
        ----------
        obj : Any
            Object potentially containing tensors
        node_addr : str
            Node address for shard storage

        Returns
        -------
        Any
            Object with tensors converted to RTensors
        """
        if obj is None:
            return None

        if isinstance(obj, torch.Tensor):
            tensor = obj.detach().cpu()
            shard_id = get_backend().store(tensor)
            shard = TensorShardInfo(
                shard_id=shard_id,
                node_addr=node_addr,
            )
            return RTensor(shard=shard, data=tensor.to("meta"))

        if isinstance(obj, dict):
            # Compact trajectory dicts by trimming padding before storage.
            # split_and_unpad_tensor auto-derives trim lengths from attention_mask.
            attn_mask = obj.get("attention_mask")
            if isinstance(attn_mask, torch.Tensor) and attn_mask.ndim >= 2:
                from areal.utils.data import split_and_unpad_tensor

                compacted = split_and_unpad_tensor(
                    obj,
                    n_trajs=1,
                    traj_group_sizes=[attn_mask.shape[0]],
                )
                if compacted is not None:
                    obj = compacted[0]
            return {k: RTensor.remotize(v, node_addr=node_addr) for k, v in obj.items()}

        if isinstance(obj, list):
            return [RTensor.remotize(item, node_addr=node_addr) for item in obj]

        if isinstance(obj, tuple):
            return tuple(RTensor.remotize(item, node_addr=node_addr) for item in obj)

        return obj

    @staticmethod
    def localize(obj: Any) -> Any:
        """Convert RTensors to local tensors in nested structures.

        Inverse of remotize() - fetches remote data and converts to local tensors.
        All remote fetches are batched concurrently for performance.

        Parameters
        ----------
        obj : Any
            Object potentially containing RTensors

        Returns
        -------
        Any
            Object with RTensors converted to local tensors
        """
        # Pre-fetch all remote tensors concurrently
        rtensors: list[RTensor] = []
        RTensor._collect_all(obj, rtensors)
        meta_rtensors = [rt for rt in rtensors if rt.data.is_meta]
        if meta_rtensors:
            shards = [rt.shard for rt in meta_rtensors]
            results = get_backend().fetch(shards)
            for rt, tensor in zip(meta_rtensors, results):
                rt.data = tensor

        # Recursively replace RTensors with local tensors (all cache hits now)
        return RTensor._localize_recursive(obj)

    @staticmethod
    def _collect_all(obj: Any, result: list[RTensor]) -> None:
        """Collect all RTensor instances from a nested structure."""
        if isinstance(obj, RTensor):
            result.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                RTensor._collect_all(v, result)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                RTensor._collect_all(item, result)

    @staticmethod
    def _localize_recursive(obj: Any) -> Any:
        """Recursively replace RTensors with their local tensor data."""
        if isinstance(obj, RTensor):
            return obj.to_local()

        if isinstance(obj, dict):
            return {k: RTensor._localize_recursive(v) for k, v in obj.items()}

        if isinstance(obj, list):
            return [RTensor._localize_recursive(item) for item in obj]

        if isinstance(obj, tuple):
            return tuple(RTensor._localize_recursive(item) for item in obj)

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


# =============================================================================
# Local Storage (used by HttpRTensorBackend)
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
