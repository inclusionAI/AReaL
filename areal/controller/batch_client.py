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
from areal.utils.data import concat_padded_tensors

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
        """Fetch the full physical shard from a node (without slicing).

        This fetches the entire shard identified by ``shard.shard_id`` from the
        remote node. Logical slicing based on ``offset`` and ``batch_size`` is
        handled in :meth:`_fetch_shard`.

        Parameters
        ----------
        session : aiohttp.ClientSession
            HTTP session to use
        shard : ShardMetadata
            Metadata describing the shard to fetch

        Returns
        -------
        dict[str, Any]
            Full shard data
        """
        url = f"http://{shard.node_addr}/data/{shard.shard_id}"

        try:
            async with session.get(url, timeout=self.timeout) as response:
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
                    f"Fetched shard {shard.shard_id} from {shard.node_addr} "
                    f"({len(data_bytes)} bytes)"
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

    async def fetch_batch(
        self, metadata: BatchMetadata
    ) -> dict[str, torch.Tensor | Any]:
        """Fetch all shards for a batch and assemble into complete dataset.

        Parameters
        ----------
        metadata : BatchMetadata
            Metadata describing the batch to fetch

        Returns
        -------
        dict[str, torch.Tensor | Any]
            Complete dataset assembled from all shards
        """
        if not metadata.shards:
            return {}

        session = aiohttp.ClientSession()
        try:
            # Fetch all shards concurrently (with logical slicing)
            logger.info(
                f"Fetching {len(metadata.shards)} shards for batch {metadata.batch_id}"
            )
            for i, shard in enumerate(metadata.shards):
                logger.info(
                    f"Shard {i + 1}/{len(metadata.shards)}: {shard} (node={shard.node_id}, "
                    f"addr={shard.node_addr}, shard_id={shard.shard_id}, "
                    f"offset={shard.offset}, batch_size={shard.batch_size}, "
                    f"fields={list(shard.fields.keys())})"
                )
            tasks = [self._fetch_shard(session, shard) for shard in metadata.shards]
            shard_data_list = await asyncio.gather(*tasks)

            # Concatenate shard data
            logger.debug(
                f"Assembling {len(shard_data_list)} shards for batch {metadata.batch_id}"
            )

            # Check if all shards have the same keys
            if shard_data_list:
                all_keys = set()
                for shard_data in shard_data_list:
                    all_keys.update(shard_data.keys())

                # Check if all shards have the same keys
                same_keys = all(
                    set(shard_data.keys()) == all_keys for shard_data in shard_data_list
                )

                if same_keys and "attention_mask" in all_keys:
                    # All shards have the same keys and have attention_mask
                    # Use the original concat_padded_tensors
                    dataset = concat_padded_tensors(shard_data_list)
                else:
                    # Different keys across shards, merge manually
                    logger.warning(
                        f"Shards have different keys, merging manually. "
                        f"All keys: {sorted(all_keys)}"
                    )
                    dataset = self._merge_shards_with_different_keys(
                        shard_data_list, all_keys
                    )
            else:
                dataset = {}

            return dataset
        finally:
            # Always close the session when done
            if not session.closed:
                await session.close()

    def _merge_shards_with_different_keys(
        self, shard_data_list: list[dict[str, torch.Tensor | Any]], all_keys: set[str]
    ) -> dict[str, torch.Tensor | Any]:
        """Merge shards that may have different keys.

        Parameters
        ----------
        shard_data_list : list[dict[str, torch.Tensor | Any]]
            List of shard data dictionaries
        all_keys : set[str]
            Set of all keys across all shards

        Returns
        -------
        dict[str, torch.Tensor | Any]
            Merged dataset
        """
        result = {}

        for key in sorted(all_keys):
            # Collect all values for this key from shards that have it
            values_to_concat = []

            for shard_data in shard_data_list:
                if key in shard_data:
                    values_to_concat.append(shard_data[key])

            if not values_to_concat:
                continue

            # Determine the type of value
            first_value = values_to_concat[0]

            if isinstance(first_value, torch.Tensor):
                # Check if tensors need padding (multi-dimensional with varying lengths)
                if first_value.ndim > 1:
                    # Assume dim=1 is the sequence dimension, check if padding is needed
                    max_length = max(tensor.shape[1] for tensor in values_to_concat)
                    need_padding = any(
                        tensor.shape[1] < max_length for tensor in values_to_concat
                    )

                    if need_padding:
                        # Pad tensors to max_length before concatenating
                        padded_tensors = []
                        for tensor in values_to_concat:
                            if tensor.shape[1] < max_length:
                                # Pad along sequence dimension (dim=1)
                                pad_width = max_length - tensor.shape[1]
                                # Determine pad value based on key
                                if key == "attention_mask":
                                    pad_value = 0
                                else:
                                    pad_value = 0.0

                                # Create padding for dim=1 (sequence dimension)
                                # torch.nn.functional.pad format: (pad_left, pad_right) from last to first dim
                                n_dim = tensor.ndim
                                pad_mode = (0,) * (2 * (n_dim - 2)) + (
                                    0,
                                    pad_width,
                                )  # Pad right side of dim=1
                                padded_tensor = torch.nn.functional.pad(
                                    tensor, pad_mode, value=pad_value
                                )
                                padded_tensors.append(padded_tensor)
                            else:
                                padded_tensors.append(tensor)
                        result[key] = torch.cat(padded_tensors, dim=0)
                    else:
                        # All tensors have same shape, directly concat
                        result[key] = torch.cat(values_to_concat, dim=0)
                else:
                    # 1D tensor, directly concat
                    result[key] = torch.cat(values_to_concat, dim=0)
            elif isinstance(first_value, list):
                # Concatenate lists
                merged_list = []
                for v in values_to_concat:
                    merged_list.extend(v)
                result[key] = merged_list
            else:
                # For scalar values, just keep the first one
                # (or raise an error if this doesn't make sense for your use case)
                result[key] = first_value
                logger.warning(
                    f"Key '{key}' has scalar value, keeping first value: {first_value}"
                )

        return result

    async def _fetch_shard(
        self, session: aiohttp.ClientSession, shard: ShardMetadata
    ) -> dict[str, torch.Tensor | Any]:
        """Fetch a logical shard (sub-range) from a physical shard.

        This method uses :class:`ShardMetadata`'s ``offset`` and ``batch_size``
        to slice the full physical shard along the batch dimension.

        Parameters
        ----------
        session : aiohttp.ClientSession
            HTTP session to use
        shard : ShardMetadata
            Metadata describing the logical shard to fetch

        Returns
        -------
        dict[str, torch.Tensor | Any]
            Sliced shard data matching the logical sub-range
        """
        full_data = await self.fetch_shard(session, shard)

        # If offset is zero and batch_size matches, fast-path: no slicing needed.
        # Otherwise, slice along batch dimension.
        offset = shard.offset
        length = shard.batch_size

        sliced: dict[str, torch.Tensor | Any] = {}
        for key, value in full_data.items():
            if isinstance(value, torch.Tensor):
                sliced[key] = value[offset : offset + length]
            elif isinstance(value, list):
                sliced[key] = value[offset : offset + length]
            else:
                # Scalar / non-batched value, keep as-is
                sliced[key] = value

        return sliced

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
