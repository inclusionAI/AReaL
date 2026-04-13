"""Partitioned HTTP dispatcher for one 5D-parallel worker group.

Replicates the dispatch semantics of
:class:`~areal.infra.controller.train_controller.TrainController`:

- Detect tensor vs scalar inputs.
- Partition tensor inputs across DP groups via
  ``balanced_greedy_partition``.
- Fan out to all workers (DP heads receive their data slice; non-DP-head
  workers receive an empty signal so they can participate in NCCL
  collectives via intra-group broadcast).
- Collect results from DP heads and merge them back into the original
  trajectory order.
- Pad the batch to a multiple of ``dp_size * group_size`` when not
  evenly divisible (eval-padding behaviour from PR 1109).

Usage::

    result = await dispatcher.dispatch("/train_batch").post(body)
    version = await dispatcher.dispatch("/get_version").get()
    all_stats = await dispatcher.broadcast("/export_stats").get()
    responses = await dispatcher.broadcast("/set_version").post(body)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import aiohttp
import orjson

from areal.experimental.training_service.data_proxy.topology import WorkerTopology
from areal.infra.controller.train_controller import (
    _dispatch_tensors,
    _is_tensor_like,
    _merge_tensors,
    _pad_eval_batch,
)
from areal.infra.rpc.serialization import deserialize_value, serialize_value
from areal.utils import logging

logger = logging.getLogger("TrainDataProxy")


@dataclass
class _WorkerResponse:
    """Internal container for a validated worker HTTP response."""

    addr: str
    status: int
    content: bytes


class Dispatcher:
    """Partitioned HTTP dispatcher for one 5D-parallel worker group."""

    def __init__(
        self,
        topology: WorkerTopology,
        request_timeout: float = 600.0,
        *,
        _session: Any | None = None,
    ):
        self._topology = topology
        self._request_timeout = request_timeout
        self._session = _session or aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=request_timeout),
        )

    async def close(self) -> None:
        """Close the underlying HTTP session."""
        await self._session.close()

    # ------------------------------------------------------------------
    # Fluent API
    # ------------------------------------------------------------------

    def dispatch(self, path: str) -> DispatchRequest:
        """Return a dispatch builder for *path*.

        Dispatch operations route tensors via DP-aware partitioning
        and return a single merged result.  Scalar payloads are sent to
        all DP heads and the first response is returned.
        """
        return DispatchRequest(self, path)

    def broadcast(self, path: str) -> BroadcastRequest:
        """Return a broadcast builder for *path*.

        Broadcast operations send the same request to every worker
        and return all responses.
        """
        return BroadcastRequest(self, path)

    # ------------------------------------------------------------------
    # Internal HTTP helpers
    # ------------------------------------------------------------------

    async def _do_get(self, addr: str, path: str) -> _WorkerResponse:
        async with self._session.get(f"{addr}{path}") as resp:
            content = await resp.read()
            return _WorkerResponse(addr=addr, status=resp.status, content=content)

    async def _do_post(self, addr: str, path: str, body: bytes) -> _WorkerResponse:
        async with self._session.post(
            f"{addr}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
        ) as resp:
            content = await resp.read()
            return _WorkerResponse(addr=addr, status=resp.status, content=content)

    async def _gather_validated(
        self,
        tasks: list,
        addrs: list[str],
    ) -> list[_WorkerResponse]:
        """Run *tasks* concurrently and validate every response."""
        raw = await asyncio.gather(*tasks, return_exceptions=True)
        validated: list[_WorkerResponse] = []
        for i, result in enumerate(raw):
            if isinstance(result, BaseException):
                raise RuntimeError(f"Worker {addrs[i]} failed: {result}")
            _raise_for_worker(result)
            validated.append(result)
        return validated


class DispatchRequest:
    """Builder for DP-aware dispatch operations.

    Tensor payloads are partitioned across DP groups.  Scalar payloads
    are sent to all DP heads and the first response is returned.

    Obtain via :meth:`Dispatcher.dispatch`.
    """

    __slots__ = ("_dispatcher", "_path")

    def __init__(self, dispatcher: Dispatcher, path: str) -> None:
        self._dispatcher = dispatcher
        self._path = path

    async def get(self) -> bytes:
        """GET from all DP heads, return the first response."""
        d = self._dispatcher
        dp_head_addrs = [d._topology.workers[i].addr for i in d._topology.dp_heads]
        tasks = [d._do_get(addr, self._path) for addr in dp_head_addrs]
        responses = await d._gather_validated(tasks, dp_head_addrs)
        return responses[0].content

    async def post(self, body: bytes) -> bytes:
        """POST with tensor-aware dispatch.

        If the payload contains partitionable tensor batches, it is split
        across DP groups, padded if necessary, and the per-shard results
        are merged in original trajectory order.

        Otherwise the body is forwarded as-is to all DP heads and the
        first response is returned (scalar dispatch).
        """
        data = orjson.loads(body)
        raw_args = deserialize_value(data.get("args", []))
        raw_kwargs = deserialize_value(data.get("kwargs", {}))

        group_size: int = 1
        if isinstance(raw_kwargs, dict):
            group_size = raw_kwargs.pop("group_size", 1)

        if (
            _is_tensor_like(raw_args) or _is_tensor_like(raw_kwargs)
        ) and _contains_partitionable_tensor_batch(raw_args, raw_kwargs):
            return await self._tensor_dispatch(raw_args, raw_kwargs, group_size)
        return await self._scalar_post(body)

    async def _scalar_post(self, body: bytes) -> bytes:
        """POST to all DP heads, return the first response."""
        d = self._dispatcher
        dp_head_addrs = [d._topology.workers[i].addr for i in d._topology.dp_heads]
        tasks = [d._do_post(addr, self._path, body) for addr in dp_head_addrs]
        responses = await d._gather_validated(tasks, dp_head_addrs)
        return responses[0].content

    # ------------------------------------------------------------------
    # Tensor dispatch (partitioned fan-out + merge)
    # ------------------------------------------------------------------

    async def _tensor_dispatch(
        self,
        raw_args: list[Any],
        raw_kwargs: dict[str, Any],
        group_size: int,
    ) -> bytes:
        d = self._dispatcher
        dp_size = d._topology.dp_size

        # Pad when batch is not evenly divisible (PR 1109 eval-padding).
        args_tuple = _pad_eval_batch(tuple(raw_args), dp_size, group_size)
        raw_args = list(args_tuple)

        dp_args, dp_kwargs, group_indices = self._partition_inputs(
            raw_args, raw_kwargs, group_size
        )

        dp_head_results = await self._fan_out(dp_args, dp_kwargs)

        merged = _merge_tensors(dp_head_results, group_indices)

        return orjson.dumps({"status": "success", "result": serialize_value(merged)})

    # ------------------------------------------------------------------
    # Partitioning (mirrors TrainController._partition_inputs)
    # ------------------------------------------------------------------

    def _partition_inputs(
        self,
        args: list[Any],
        kwargs: dict[str, Any],
        group_size: int,
    ) -> tuple[list[list[Any]], dict[str, list[Any]], list[list[int]]]:
        dp_size = self._dispatcher._topology.dp_size
        group_indices: list[list[int]] | None = None

        def _split(item: Any) -> list[Any]:
            nonlocal group_indices
            if _is_tensor_like(item):
                if group_indices is None:
                    splits, group_indices = _dispatch_tensors(
                        item, dp_size, group_size=group_size
                    )
                    return splits
                return [[item[i] for i in idxs] for idxs in group_indices]
            return [item] * dp_size

        dp_args = [_split(a) for a in args]
        dp_kwargs = {k: _split(v) for k, v in kwargs.items()}

        if group_indices is None:
            raise RuntimeError(
                "dispatch_compute called with tensor detection but no "
                "tensor-like arg was found during partitioning"
            )

        return dp_args, dp_kwargs, group_indices

    # ------------------------------------------------------------------
    # Fan-out to workers (DP heads get partition, others get empty)
    # ------------------------------------------------------------------

    async def _fan_out(
        self,
        dp_args: list[list[Any]],
        dp_kwargs: dict[str, list[Any]],
    ) -> list[Any]:
        d = self._dispatcher
        dp_head_set = set(d._topology.dp_heads)

        payloads: list[bytes] = []
        dp_idx = 0
        for i in range(len(d._topology.workers)):
            if i in dp_head_set:
                worker_args = [splits[dp_idx] for splits in dp_args]
                worker_kwargs = {k: splits[dp_idx] for k, splits in dp_kwargs.items()}
                dp_idx += 1
            else:
                worker_args = []
                worker_kwargs = {}

            payloads.append(
                orjson.dumps(
                    {
                        "args": serialize_value(worker_args),
                        "kwargs": serialize_value(worker_kwargs),
                    }
                )
            )

        addrs = [w.addr for w in d._topology.workers]
        tasks = [
            d._do_post(addrs[i], self._path, payloads[i])
            for i in range(len(d._topology.workers))
        ]
        responses = await d._gather_validated(tasks, addrs)

        dp_head_results: list[Any] = []
        for i in d._topology.dp_heads:
            result_data = orjson.loads(responses[i].content)
            result = deserialize_value(result_data.get("result"))
            dp_head_results.append(result)

        return dp_head_results


class BroadcastRequest:
    """Builder for broadcast operations across all workers.

    Every worker receives the same request and all responses are
    returned.

    Obtain via :meth:`Dispatcher.broadcast`.
    """

    __slots__ = ("_dispatcher", "_path")

    def __init__(self, dispatcher: Dispatcher, path: str) -> None:
        self._dispatcher = dispatcher
        self._path = path

    async def get(self) -> list[bytes]:
        """GET from every worker, return all responses."""
        d = self._dispatcher
        addrs = [w.addr for w in d._topology.workers]
        tasks = [d._do_get(addr, self._path) for addr in addrs]
        responses = await d._gather_validated(tasks, addrs)
        return [r.content for r in responses]

    async def post(self, body: bytes) -> list[bytes]:
        """POST the same body to every worker, return all responses."""
        d = self._dispatcher
        addrs = [w.addr for w in d._topology.workers]
        tasks = [d._do_post(addr, self._path, body) for addr in addrs]
        responses = await d._gather_validated(tasks, addrs)
        return [r.content for r in responses]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _raise_for_worker(resp: _WorkerResponse) -> None:
    if resp.status >= 400:
        text = resp.content.decode("utf-8", errors="replace")
        raise RuntimeError(f"Worker {resp.addr} returned {resp.status}: {text}")


def _contains_partitionable_tensor_batch(
    args: list[Any], kwargs: dict[str, Any]
) -> bool:
    """Return True when payload matches list-of-items partition contract.

    The current partitioner (``_dispatch_tensors``) operates on list-like batches of
    per-item dict payloads. Some endpoints (e.g. ``forward_batch`` with packed
    tensor dicts) send tensor-containing dicts directly; those should use scalar
    fan-out instead of list partitioning.
    """

    def _is_partitionable(v: Any) -> bool:
        return isinstance(v, list) and len(v) > 0 and _is_tensor_like(v)

    return any(_is_partitionable(v) for v in args) or any(
        _is_partitionable(v) for v in kwargs.values()
    )
