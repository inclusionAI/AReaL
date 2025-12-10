"""HTTP client for distributed batch memory retrieval."""

import asyncio
from collections import defaultdict
from typing import Any

import aiohttp
import orjson
import torch

from areal.controller.batch_metadata import BatchMetadata, ShardId, ShardMetadata
from areal.scheduler.rpc.serialization import deserialize_value, serialize_value
from areal.utils import logging

logger = logging.getLogger("BatchClient")

# Default connection limit for batch data fetching
DEFAULT_CONNECTION_LIMIT = 1000


class BatchDataClient:
    """HTTP client for fetching distributed batch data."""

    def __init__(
        self,
        timeout: float = 300.0,
        connection_limit: int = DEFAULT_CONNECTION_LIMIT,
        connect_timeout: float | None = None,
        read_timeout: float | None = None,
        retries: int = 2,
        backoff_factor: float = 0.5,
    ):
        # Split timeout so we can surface slow connects vs slow reads.
        self.timeout = aiohttp.ClientTimeout(
            total=timeout, connect=connect_timeout, sock_read=read_timeout
        )
        self.connection_limit = connection_limit
        self.retries = retries
        self.backoff_factor = backoff_factor

    async def fetch_shard(
        self, session: aiohttp.ClientSession, shard: ShardMetadata
    ) -> dict[str, Any]:
        """Fetch a shard from a node."""
        url = f"http://{shard.node_addr}/data/{shard.shard_id}"
        params = {}

        last_exc: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                async with session.get(
                    url, params=params, timeout=self.timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise RuntimeError(
                            f"Failed to fetch shard {shard.shard_id} from {shard.node_addr}: "
                            f"HTTP {response.status} - {error_text}"
                        )

                    data_bytes = await response.read()
                    serialized_data = orjson.loads(data_bytes)
                    data = deserialize_value(serialized_data)

                    assert isinstance(data, torch.Tensor)
                    result = {shard.shard_id.key: data}

                    logger.info(
                        f"Fetched shard {shard.shard_id} from {shard.node_addr} "
                        f"({len(data_bytes)} bytes)"
                    )
                    return result

            except asyncio.TimeoutError:
                last_exc = RuntimeError(
                    f"Timeout fetching shard {shard.shard_id} from {shard.node_addr}"
                )
            except Exception as e:
                last_exc = RuntimeError(
                    f"Error fetching shard {shard.shard_id} from {shard.node_addr}: {e}"
                )

            if attempt < self.retries:
                delay = self.backoff_factor * (2**attempt)
                logger.warning(
                    f"Retrying shard {shard.shard_id} from {shard.node_addr} after attempt "
                    f"{attempt + 1}/{self.retries + 1} failed: {last_exc}; sleep {delay}s"
                )
                await asyncio.sleep(delay)

        assert last_exc is not None
        raise last_exc

    async def fetch_shards(self, metadata: BatchMetadata) -> list[dict[str, Any]]:
        """Fetch all shards for a batch.

        Shards with the same task_id are grouped together into a single dict.
        Returns a list of dicts, where each dict contains all data for one task_id.
        """
        if not metadata.shards:
            return []

        connector = aiohttp.TCPConnector(limit=self.connection_limit)
        async with aiohttp.ClientSession(
            timeout=self.timeout, connector=connector
        ) as session:
            logger.info(
                f"Fetching {len(metadata.shards)} shards for batch {metadata.batch_id}"
            )
            tasks = [self.fetch_shard(session, shard) for shard in metadata.shards]
            shard_results = await asyncio.gather(*tasks, return_exceptions=True)

            task_data_map: dict[str, dict[str, Any]] = defaultdict(dict)
            failures: list[str] = []

            for shard, shard_result in zip(metadata.shards, shard_results):
                if isinstance(shard_result, Exception):
                    failures.append(
                        f"{shard.shard_id}@{shard.node_addr}: {shard_result}"
                    )
                    continue

                task_id = shard.shard_id.task_id
                task_data_map[task_id].update(shard_result)

            if failures:
                raise RuntimeError(
                    "Failed to fetch shards: " + "; ".join(sorted(failures))
                )

            return list(task_data_map.values())

    async def store_shard(
        self,
        session: aiohttp.ClientSession,
        node_addr: str,
        shard_id: ShardId,
        data: torch.Tensor,
    ) -> None:
        """Store a shard on a node."""
        url = f"http://{node_addr}/data/{shard_id}"

        serialized_data = serialize_value(data)
        data_bytes = orjson.dumps(serialized_data)

        try:
            async with session.put(
                url,
                data=data_bytes,
                headers={"Content-Type": "application/octet-stream"},
                timeout=self.timeout,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(
                        f"Failed to store shard {shard_id} to {node_addr}: "
                        f"HTTP {response.status} - {error_text}"
                    )

                logger.debug(
                    f"Stored shard {shard_id} to {node_addr} ({len(data_bytes)} bytes)"
                )

        except asyncio.TimeoutError as e:
            raise RuntimeError(
                f"Timeout storing shard {shard_id} to {node_addr}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Error storing shard {shard_id} to {node_addr}: {e}"
            ) from e

    async def clear_batches(
        self, node_addrs: set[str], shard_ids: list[ShardId]
    ) -> None:
        """Clear specific shards on multiple nodes."""
        connector = aiohttp.TCPConnector(limit=self.connection_limit)
        async with aiohttp.ClientSession(
            timeout=self.timeout, connector=connector
        ) as session:
            tasks = [
                self._clear_node(session, node_addr, shard_ids)
                for node_addr in node_addrs
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _clear_node(
        self, session: aiohttp.ClientSession, node_addr: str, shard_ids: list[ShardId]
    ) -> None:
        """Clear specific shards on a single node."""
        url = f"http://{node_addr}/data/clear"

        shard_id_strings = [str(shard_id) for shard_id in shard_ids]

        try:
            async with session.delete(
                url, json={"shard_ids": shard_id_strings}, timeout=self.timeout
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.warning(
                        f"Failed to clear data on {node_addr}: "
                        f"HTTP {response.status} - {error_text}"
                    )
                else:
                    result = await response.json()
                    logger.debug(
                        f"Cleared {result.get('cleared_count', 0)} shards on {node_addr}"
                    )

        except Exception as e:
            logger.warning(f"Error clearing data on {node_addr}: {e}")
