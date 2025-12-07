"""HTTP client for distributed batch memory retrieval.

This module implements an HTTP client that fetches data shards from multiple
nodes concurrently and assembles them into a complete dataset.
"""

import asyncio
import io
import pickle
from typing import Any

import aiohttp
import torch

from areal.controller.batch_metadata import BatchMetadata, ShardMetadata
from areal.utils import logging

logger = logging.getLogger("BatchClient")


class BatchDataClient:
    """HTTP client for fetching distributed batch data.

    This client fetches data shards from multiple nodes concurrently and
    assembles them into a complete dataset.
    """

    def __init__(self, timeout: float = 300.0):
        """Initialize the batch data client.

        Parameters
        ----------
        timeout : float, optional
            Timeout for HTTP requests in seconds, by default 300.0
        """
        self.timeout = aiohttp.ClientTimeout(total=timeout)

    async def fetch_shard(
        self, session: aiohttp.ClientSession, shard: ShardMetadata
    ) -> dict[str, Any]:
        """Fetch a logical shard (sub-range) from a physical shard.

        This fetches only the logical shard identified by ``shard.offset`` and
        ``shard.batch_size`` from the remote node, reducing data transfer.

        Parameters
        ----------
        session : aiohttp.ClientSession
            HTTP session to use
        shard : ShardMetadata
            Metadata describing the logical shard to fetch

        Returns
        -------
        dict[str, Any]
            Sliced shard data matching the logical sub-range
        """
        # Build URL with query parameters for logical shard slicing
        # Only include parameters if we need to slice (offset > 0 or batch_size is specified)
        url = f"http://{shard.node_addr}/data/{shard.shard_id}"
        params = {}
        if shard.offset > 0:
            params["offset"] = shard.offset
        if shard.batch_size > 0:
            params["batch_size"] = shard.batch_size

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
                buffer = io.BytesIO(data_bytes)
                data = pickle.load(buffer)

                logger.debug(
                    f"Fetched logical shard {shard.shard_id} from {shard.node_addr} "
                    f"(offset={shard.offset}, batch_size={shard.batch_size}, "
                    f"{len(data_bytes)} bytes)"
                )
                return data

        except asyncio.TimeoutError as e:
            raise RuntimeError(
                f"Timeout fetching shard {shard.shard_id} from {shard.node_addr}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Error fetching shard {shard.shard_id} from {shard.node_addr}: {e}"
            ) from e

    async def fetch_shards(
        self, metadata: BatchMetadata
    ) -> list[dict[str, torch.Tensor | Any]]:
        """Fetch all shards for a batch and return raw shard data.

        This method only fetches data from remote nodes without any merging or
        processing. Data operations should be handled by the caller.

        Parameters
        ----------
        metadata : BatchMetadata
            Metadata describing the batch to fetch

        Returns
        -------
        list[dict[str, torch.Tensor | Any]]
            List of raw shard data dictionaries, one per shard
        """
        if not metadata.shards:
            return []

        session = aiohttp.ClientSession()
        try:
            logger.info(
                f"Fetching {len(metadata.shards)} shards for batch {metadata.batch_id}"
            )
            tasks = [self.fetch_shard(session, shard) for shard in metadata.shards]
            shard_data_list = await asyncio.gather(*tasks)
            return shard_data_list
        finally:
            if not session.closed:
                await session.close()

    async def store_shard(
        self,
        session: aiohttp.ClientSession,
        node_addr: str,
        shard_id: str,
        global_step: int,
        data: dict[str, torch.Tensor | Any],
    ) -> None:
        """Store a shard on a node.

        Parameters
        ----------
        session : aiohttp.ClientSession
            HTTP session to use
        node_addr : str
            Network address (host:port) of the target node
        shard_id : str
            Unique identifier for this shard
        global_step : int
            Global training step
        data : dict[str, torch.Tensor | Any]
            Data to store
        """
        url = f"http://{node_addr}/data/{shard_id}?global_step={global_step}"

        # Serialize data
        buffer = io.BytesIO()
        pickle.dump(data, buffer)
        data_bytes = buffer.getvalue()

        try:
            async with session.put(
                url, data=data_bytes, timeout=self.timeout
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

    async def clear_old_data(self, node_addrs: set[str], global_step: int) -> None:
        """Clear old data on multiple nodes.

        Parameters
        ----------
        node_addrs : set[str]
            Set of node addresses (host:port) to clear
        global_step : int
            Clear all data with step < global_step
        """
        session = aiohttp.ClientSession()
        try:
            tasks = [
                self._clear_node(session, node_addr, global_step)
                for node_addr in node_addrs
            ]
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            # Always close the session when done
            if not session.closed:
                await session.close()

    async def _clear_node(
        self, session: aiohttp.ClientSession, node_addr: str, global_step: int
    ) -> None:
        """Clear old data on a single node.

        Parameters
        ----------
        session : aiohttp.ClientSession
            HTTP session to use
        node_addr : str
            Network address (host:port) of the target node
        global_step : int
            Clear all data with step < global_step
        """
        url = f"http://{node_addr}/data/clear?global_step={global_step}"

        try:
            async with session.delete(url, timeout=self.timeout) as response:
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
