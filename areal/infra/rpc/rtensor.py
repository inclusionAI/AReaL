# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import hashlib
import math
import platform
import uuid
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory as _SharedMemory
from threading import Lock
from typing import Any, Protocol, cast

import aiohttp
import orjson
import ray
import torch

from areal.infra.utils.concurrent import run_async_task
from areal.infra.utils.http import DEFAULT_REQUEST_TIMEOUT, get_default_connector
from areal.utils import logging

logger = logging.getLogger("HttpRTensor")

# =============================================================================
# SharedMemory IPC pool
# =============================================================================

_SHM_PREFIX = "rt_"

# Mapping torch dtype <-> uint8 enum
_DTYPE_TO_ENUM: dict[torch.dtype, int] = {
    torch.float16: 0,
    torch.float32: 1,
    torch.float64: 2,
    torch.int8: 3,
    torch.int16: 4,
    torch.int32: 5,
    torch.int64: 6,
    torch.bool: 7,
    torch.bfloat16: 8,
    torch.uint8: 9,
}
_ENUM_TO_DTYPE: dict[int, torch.dtype] = {v: k for k, v in _DTYPE_TO_ENUM.items()}

_DTYPE_ELEMENT_SIZE: dict[torch.dtype, int] = {
    dt: torch.tensor([], dtype=dt).element_size() for dt in _DTYPE_TO_ENUM
}

# macOS has a 31-char shm name limit; Linux allows 255.
_SHM_NAME_MAX_LEN = 31 if platform.system() == "Darwin" else 255


class RTensorShmPool:
    """Writer-owned SharedMemory pool for same-node RTensor IPC.

    This pool pre-allocates a single large SharedMemory segment and uses
    a bump allocator to sub-allocate regions for individual tensor shards.
    It is designed for the training-worker RPC server path where tensor
    lifetimes are bounded by training steps.

    Writer lifecycle (rpc_server only):
        pool = RTensorShmPool(...)
        pool.init_writer()           # creates the shm segment
        # ... step loop ...
        pool.allocate_and_write(...)  # per tensor
        pool.release(...)             # per tensor, at clear_batches
        pool.reset()                  # at step boundary
        pool.close()                  # at shutdown

    Reader lifecycle (any local process):
        pool.read_tensor(pool_name, offset, nbytes, dtype_enum, shape)
    """

    def __init__(
        self,
        job_token: str,
        role: str,
        worker_index: int,
        pool_size_bytes: int,
    ) -> None:
        self._job_token = job_token
        self._role = role
        self._worker_index = worker_index
        self._pool_size = pool_size_bytes
        self._lock = Lock()
        self._reader_lock = Lock()

        self._pool_name = self._make_pool_name(job_token, role, worker_index)
        self._shm: _SharedMemory | None = None
        self._enabled = True
        self._is_writer = False
        self._closing = False

        # bump allocator state (writer side)
        self._next_offset = 0
        self._occupied: dict[str, tuple[int, int, int, list[int]]] = {}
        self._in_flight = 0  # count of allocate_and_write() between reserve and publish

        # reader-side cache
        self._reader_pools: dict[str, _SharedMemory] = {}

    @staticmethod
    def _make_pool_name(
        job_token: str,
        role: str,
        worker_index: int,
        suffix: str = "",
    ) -> str:
        raw = f"{_SHM_PREFIX}{job_token}_{role}_{worker_index}{suffix}"
        if len(raw) + 1 > _SHM_NAME_MAX_LEN:
            digest = hashlib.md5(raw.encode()).hexdigest()
            raw = f"{_SHM_PREFIX}{digest[: _SHM_NAME_MAX_LEN - 4]}"
        return raw

    def init_writer(self) -> None:
        if not self._enabled:
            return

        for retry in range(3):
            try:
                self._shm = _SharedMemory(
                    name=self._pool_name,
                    create=True,
                    size=self._pool_size,
                )
                self._is_writer = True
                self._next_offset = 0
                return
            except FileExistsError:
                self._pool_name = self._make_pool_name(
                    self._job_token,
                    self._role,
                    self._worker_index,
                    suffix=f"_r{retry}",
                )
            except Exception as exc:
                logger.warning("Failed to initialize RTensor shm pool: %s", exc)
                self._enabled = False
                self._shm = None
                return

        logger.warning(
            "Failed to create RTensor shm pool after retries; fallback to HTTP"
        )
        self._enabled = False

    def allocate_and_write(self, shard_id: str, tensor: torch.Tensor) -> bool:
        """Try to write *tensor* into the pool.

        Returns ``True`` on success, ``False`` if the tensor should fall back
        to the HTTP path (pool disabled, unsupported dtype, or insufficient
        space).

        Thread safety
        -------------
        The method uses a **reserve -> write -> publish** protocol to ensure
        that ``get_meta()`` never exposes a shard whose data has not been
        fully written yet:

        1. **Reserve** (under ``_lock``): bump ``_next_offset`` to claim the
           region, but do NOT register the shard in ``_occupied``.
        2. **Write** (lock-free): copy tensor bytes into the reserved region.
           No other thread can discover this region via ``get_meta()`` because
           it is not yet in ``_occupied``.
        3. **Publish** (under ``_lock``): insert the shard into ``_occupied``,
           making it visible to ``get_meta()`` callers.
        """
        shm = self._shm
        if not self._enabled or shm is None or self._closing:
            return False
        if tensor.dtype not in _DTYPE_TO_ENUM:
            return False

        t = tensor.contiguous()
        try:
            data_view = t.numpy().ravel()
        except TypeError:
            data_view = t.view(torch.uint8).numpy().ravel()

        nbytes = data_view.nbytes
        dtype_enum = _DTYPE_TO_ENUM[t.dtype]
        shape = list(t.shape)

        # Step 1: reserve space (under lock)
        with self._lock:
            if self._closing:
                return False
            aligned = (self._next_offset + 63) & ~63
            if aligned + nbytes > self._pool_size:
                return False
            self._next_offset = aligned + nbytes
            self._in_flight += 1

        # Step 2: write data (lock-free; region is invisible to readers)
        try:
            buf = cast(memoryview, shm.buf)
            buf[aligned : aligned + nbytes] = data_view
        finally:
            # Step 3: publish metadata (under lock; readers can now discover it)
            with self._lock:
                self._in_flight -= 1
                self._occupied[shard_id] = (aligned, nbytes, dtype_enum, shape)

        return True

    def get_meta(self, shard_id: str) -> tuple[str, int, int, int, list[int]] | None:
        with self._lock:
            entry = self._occupied.get(shard_id)
        if entry is None:
            return None
        offset, nbytes, dtype_enum, shape = entry
        return (self._pool_name, offset, nbytes, dtype_enum, shape)

    def release(self, shard_id: str) -> None:
        with self._lock:
            self._occupied.pop(shard_id, None)

    def try_reset(self) -> bool:
        """Reset the bump pointer if the pool is fully drained.

        Unlike ``reset()``, this method does **not** assert — it returns
        ``False`` when the pool still has live or in-flight tensors.
        Intended to be called automatically after each ``release()``.
        """
        with self._lock:
            if self._occupied or self._in_flight != 0:
                return False
            self._next_offset = 0
            return True

    def reset(self) -> None:
        """Reset the bump pointer so the pool can be reused for the next step.

        Thread-safety contract (MUST be upheld by the caller)
        -----------------------------------------------------
        ``reset()`` reclaims the **entire** pool buffer by resetting
        ``_next_offset`` to 0.  After this call, subsequent
        ``allocate_and_write()`` invocations may overwrite any previously
        occupied region.

        To avoid a TOCTOU race where a reader's ``torch.frombuffer()`` ->
        ``.clone()`` window overlaps with a writer reusing the same region,
        the caller **MUST** guarantee **both** of the following before
        invoking ``reset()``:

        1. All ``release()`` calls for the current step have completed
           (i.e. ``_occupied`` is empty).  In practice this means
           ``await clear_batches()`` must have returned successfully.
        2. All reader processes have finished their ``read_tensor()`` calls
           for shards belonging to the current step.  In the training-worker
           scenario this is guaranteed because ``localize()`` completes
           before the trainer advances to ``clear_batches()``.

        If either condition is violated, readers may clone corrupted data
        (silent data corruption).
        """
        with self._lock:
            assert not self._occupied, (
                f"Cannot reset pool: {len(self._occupied)} live tensors remain"
            )
            assert self._in_flight == 0, (
                f"Cannot reset pool: {self._in_flight} in-flight allocations"
            )
            self._next_offset = 0

    def _attach_reader(self, pool_name: str) -> memoryview:
        with self._reader_lock:
            shm = self._reader_pools.get(pool_name)
            if shm is None:
                shm = _SharedMemory(name=pool_name, create=False)
                self._reader_pools[pool_name] = shm
            return cast(memoryview, shm.buf)

    def read_tensor(
        self,
        pool_name: str,
        offset: int,
        nbytes: int,
        dtype_enum: int,
        shape: list[int],
    ) -> torch.Tensor:
        buf = self._attach_reader(pool_name)
        if offset < 0 or offset + nbytes > len(buf):
            raise ValueError(
                f"Pool read out of bounds: offset={offset}, nbytes={nbytes}, pool_size={len(buf)}"
            )

        dtype = _ENUM_TO_DTYPE.get(dtype_enum)
        if dtype is None:
            raise ValueError(f"Unknown dtype enum {dtype_enum} in pool metadata")

        shape_tuple = tuple(shape)
        expected = int(math.prod(shape_tuple)) * _DTYPE_ELEMENT_SIZE[dtype]
        if expected != nbytes:
            raise ValueError(
                f"Pool meta mismatch: expected {expected} bytes from shape/dtype, got {nbytes}"
            )

        raw = torch.frombuffer(buf, dtype=torch.uint8, count=nbytes, offset=offset)
        return raw.view(dtype).reshape(shape_tuple).clone()

    def close(self) -> None:
        with self._lock:
            self._closing = True
            self._enabled = False

        with self._reader_lock:
            for shm in self._reader_pools.values():
                try:
                    shm.close()
                except Exception:
                    pass
            self._reader_pools.clear()

        if self._shm is not None:
            try:
                self._shm.close()
                if self._is_writer:
                    self._shm.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass
            finally:
                self._shm = None
                self._is_writer = False


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
    """Metadata for a single shard of an RTensor."""

    shard_id: Any
    node_addr: str

    # shm pool metadata (only set when the writer stored this tensor in pool)
    pool_name: str | None = None
    pool_offset: int | None = None
    pool_nbytes: int | None = None
    pool_dtype: int | None = None
    pool_shape: list[int] | None = None

    @property
    def has_pool_meta(self) -> bool:
        return (
            self.pool_name is not None
            and self.pool_offset is not None
            and self.pool_nbytes is not None
            and self.pool_dtype is not None
            and self.pool_shape is not None
        )


class HttpRTensorBackend:
    def __init__(
        self,
        max_shards_per_request: int = 32,
        shm_pool: RTensorShmPool | None = None,
    ) -> None:
        if max_shards_per_request <= 0:
            raise ValueError("max_shards_per_request must be positive")
        self.max_shards_per_request = max_shards_per_request
        self._shm_pool = shm_pool
        self._reader_pool: RTensorShmPool | None = None

    def _get_pool_for_reading(self) -> RTensorShmPool:
        if self._shm_pool is not None:
            return self._shm_pool
        if self._reader_pool is None:
            self._reader_pool = RTensorShmPool(
                job_token="",
                role="",
                worker_index=0,
                pool_size_bytes=0,
            )
            self._reader_pool._enabled = False
        return self._reader_pool

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
        from areal.utils.network import format_hostport, split_hostport

        try:
            host, port = split_hostport(node_addr)
            base = format_hostport(host, port)
        except ValueError:
            base = node_addr
        url = f"http://{base}/data/{shard_id}"
        last_exception = None

        for attempt in range(max_retries):
            try:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        error_body = (await resp.text()).strip()
                        detail = f" body={error_body}" if error_body else ""
                        raise RuntimeError(
                            f"Failed to fetch shard from {url}: {resp.status}{detail}"
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

    async def _fetch_shard_group(
        self,
        session: aiohttp.ClientSession,
        node_addr: str,
        grouped: list[tuple[int, TensorShardInfo]],
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> list[torch.Tensor]:
        from areal.infra.rpc.serialization import deserialize_value

        shard_ids = [shard.shard_id for _, shard in grouped]
        url = f"http://{node_addr}/data/batch"
        last_exception = None

        for attempt in range(max_retries):
            try:
                async with session.post(url, json={"shard_ids": shard_ids}) as resp:
                    if resp.status != 200:
                        error_body = (await resp.text()).strip()
                        detail = f" body={error_body}" if error_body else ""
                        raise RuntimeError(
                            f"Failed to fetch shard batch from {url}: {resp.status}{detail}"
                        )

                    data_bytes = await resp.read()
                    serialized_data = orjson.loads(data_bytes)
                    tensors = cast(
                        list[torch.Tensor], deserialize_value(serialized_data)
                    )
                    if len(tensors) != len(grouped):
                        raise RuntimeError(
                            f"Batch fetch from {url} returned {len(tensors)} shards for {len(grouped)} requested"
                        )
                    return tensors
            except (TimeoutError, aiohttp.ClientError) as e:
                last_exception = e
                logger.warning(
                    "RTensor batch fetch from %s failed: %s: %s (attempt %d/%d)",
                    url,
                    e.__class__.__name__,
                    str(e),
                    attempt + 1,
                    max_retries,
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)

        raise RuntimeError(
            f"Failed to fetch shard batch from {url} after {max_retries} attempts. "
            f"Last error: {repr(last_exception)}"
        )

    def fetch(self, shards: list[TensorShardInfo]) -> list[torch.Tensor]:
        """Fetch multiple shards concurrently."""
        if not shards:
            return []

        from areal.utils.network import is_local_addr

        indexed_shards = list(enumerate(shards))
        pool_hits: list[tuple[int, TensorShardInfo]] = []
        remote_by_node: dict[str, list[tuple[int, TensorShardInfo]]] = defaultdict(list)
        results: list[torch.Tensor | None] = [None] * len(shards)

        for index, shard in indexed_shards:
            if (
                shard.has_pool_meta
                and shard.node_addr
                and is_local_addr(shard.node_addr)
            ):
                pool_hits.append((index, shard))
            else:
                remote_by_node[shard.node_addr].append((index, shard))

        if pool_hits:
            pool = self._get_pool_for_reading()
            for idx, shard in pool_hits:
                try:
                    tensor = pool.read_tensor(
                        pool_name=cast(str, shard.pool_name),
                        offset=cast(int, shard.pool_offset),
                        nbytes=cast(int, shard.pool_nbytes),
                        dtype_enum=cast(int, shard.pool_dtype),
                        shape=cast(list[int], shard.pool_shape),
                    )
                    results[idx] = tensor
                except (FileNotFoundError, ValueError, OSError) as exc:
                    logger.debug(
                        "Pool read failed for shard %s, falling back to HTTP: %s",
                        shard.shard_id,
                        exc,
                    )
                    remote_by_node[shard.node_addr].append((idx, shard))

        if remote_by_node:

            async def _fetch_remote() -> None:
                async with self._create_session() as session:

                    async def _fetch_node(
                        node_addr: str, grouped: list[tuple[int, TensorShardInfo]]
                    ) -> None:
                        for start in range(
                            0, len(grouped), self.max_shards_per_request
                        ):
                            chunk = grouped[start : start + self.max_shards_per_request]
                            tensors = await self._fetch_shard_group(
                                session, node_addr, chunk
                            )
                            for (original_index, _), tensor in zip(
                                chunk, tensors, strict=True
                            ):
                                results[original_index] = tensor

                    await asyncio.gather(
                        *[
                            _fetch_node(node_addr, grouped)
                            for node_addr, grouped in remote_by_node.items()
                        ]
                    )

            run_async_task(_fetch_remote)

        return cast(list[torch.Tensor], results)

    def store(self, tensor: torch.Tensor) -> str:
        """Store tensor in local storage, return UUID shard_id."""
        shard_id = str(uuid.uuid4())
        _store_local(shard_id, tensor)
        if self._shm_pool is not None:
            self._shm_pool.allocate_and_write(shard_id, tensor)
        return shard_id

    def get_pool_meta(
        self, shard_id: str
    ) -> tuple[str, int, int, int, list[int]] | None:
        if self._shm_pool is None:
            return None
        return self._shm_pool.get_meta(shard_id)

    async def delete(self, node_addr: str, shard_ids: list[str]) -> None:
        """Delete shards via HTTP DELETE request."""
        from areal.utils.network import format_hostport, split_hostport

        try:
            host, port = split_hostport(node_addr)
            base = format_hostport(host, port)
        except ValueError:
            base = node_addr
        async with self._create_session() as session:
            async with session.delete(
                f"http://{base}/data/clear", json={"shard_ids": shard_ids}
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


# =============================================================================
# Client-side Fetch Buffer
# =============================================================================
# Caches fetched tensors by shard_id so that repeated fetch() calls for the
# same shard (e.g. when the same rollout_batch is sent to multiple engine
# calls across RPC boundaries) avoid redundant network transfers.
# Entries are evicted by clear_node() when clear_batches() runs at the end
# of each train step.

_fetch_buffer: dict[Any, torch.Tensor] = {}
_fetch_buffer_lock = Lock()


@dataclass
class RTensor:
    shard: TensorShardInfo
    data: torch.Tensor

    def to_local(self) -> torch.Tensor:
        """Fetch the tensor data, returning a cached version when available.

        .. warning::
            The returned tensor may be a **shared reference** held in the
            internal fetch buffer.  Callers **must not** modify it in-place
            (e.g. ``tensor.fill_()``); doing so would silently corrupt data
            seen by other consumers of the same shard.
        """
        if not self.data.is_meta:
            return self.data
        # Check client-side fetch buffer before making a network request.
        with _fetch_buffer_lock:
            cached = _fetch_buffer.get(self.shard.shard_id)
            if cached is not None:
                self.data = cached
                return self.data
        # Buffer miss: fetch from backend and populate buffer.
        self.data = get_backend().fetch([self.shard])[0]
        with _fetch_buffer_lock:
            # Double-check: another thread may have populated the buffer
            # while we were fetching.  Prefer the existing entry to avoid
            # duplicating memory.
            existing = _fetch_buffer.get(self.shard.shard_id)
            if existing is not None:
                self.data = existing
            else:
                _fetch_buffer[self.shard.shard_id] = self.data
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
            backend = get_backend()
            shard_id = backend.store(tensor)
            get_pool_meta = getattr(cast(Any, backend), "get_pool_meta", None)
            pool_meta: tuple[str, int, int, int, list[int]] | None
            if callable(get_pool_meta):
                pool_meta = cast(
                    tuple[str, int, int, int, list[int]] | None,
                    get_pool_meta(shard_id),
                )
            else:
                pool_meta = None

            if pool_meta is None:
                shard = TensorShardInfo(
                    shard_id=shard_id,
                    node_addr=node_addr,
                )
            else:
                shard = TensorShardInfo(
                    shard_id=shard_id,
                    node_addr=node_addr,
                    pool_name=pool_meta[0],
                    pool_offset=pool_meta[1],
                    pool_nbytes=pool_meta[2],
                    pool_dtype=pool_meta[3],
                    pool_shape=pool_meta[4],
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
            # Resolve as many as possible from the client-side fetch buffer.
            to_fetch: list[RTensor] = []
            with _fetch_buffer_lock:
                for rt in meta_rtensors:
                    cached = _fetch_buffer.get(rt.shard.shard_id)
                    if cached is not None:
                        rt.data = cached
                    else:
                        to_fetch.append(rt)

            # Batch-fetch only the misses from the backend.
            if to_fetch:
                shards = [rt.shard for rt in to_fetch]
                results = get_backend().fetch(shards)
                with _fetch_buffer_lock:
                    for rt, tensor in zip(to_fetch, results, strict=True):
                        rt.data = tensor
                        _fetch_buffer[rt.shard.shard_id] = tensor

        # Recursively replace RTensors with local tensors (all buffer hits now)
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
        """Clear shards from a node and evict them from the fetch buffer.

        Parameters
        ----------
        node_addr : str
            The node address
        shard_ids : list[Any]
            List of shard IDs to delete
        """
        with _fetch_buffer_lock:
            for sid in shard_ids:
                _fetch_buffer.pop(sid, None)
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
    """Store a tensor shard in local storage."""
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
            backend = _backend
            if (
                isinstance(backend, HttpRTensorBackend)
                and backend._shm_pool is not None
            ):
                backend._shm_pool.release(shard_id)
                backend._shm_pool.try_reset()
            return 1
        return 0


def storage_stats() -> dict[str, int]:
    """Get current storage stats."""
    global _storage_stats, _storage_lock, _storage
    with _storage_lock:
        return dict(
            num_tensors=len(_storage),
            total_bytes=sum(_storage_stats.values()),
        )
