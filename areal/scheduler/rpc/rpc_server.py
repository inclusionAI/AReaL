import argparse
import asyncio
import logging as stdlib_logging
import os
import socket
import traceback
import uuid
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from queue import Queue
from threading import Lock, Thread
from typing import Any

import orjson
import torch
from flask import Flask, Response, jsonify, request
from torch import Tensor

from areal.api.cli_args import BaseExperimentConfig
from areal.api.engine_api import InferenceEngine, TrainEngine
from areal.controller.batch import DistributedBatchMemory
from areal.controller.batch_metadata import (
    BatchMetadata,
    ShardId,
    ShardMetadata,
    TensorMetadata,
)
from areal.platforms import current_platform
from areal.scheduler.rpc.serialization import deserialize_value, serialize_value
from areal.utils import logging, name_resolve, seeding
from areal.utils.data import (
    broadcast_tensor_container,
    tensor_container_to,
)
from areal.utils.dynamic_import import import_from_string

logger = logging.getLogger("SyncRPCServer")

# Global engine instance - must be TrainEngine or InferenceEngine
_engine: TrainEngine | InferenceEngine | None = None

# Global batch data storage for distributed batch memory
# Storage: shard_id -> dict[str, Tensor]
_batch_storage: dict[ShardId, Tensor] = {}
_batch_storage_lock = Lock()
_batch_storage_stats: dict[ShardId, int] = defaultdict(int)

# NCCL worker thread for executing non-/data/ endpoints in a single thread
# This ensures NCCL compatibility while allowing /data/ requests to be processed concurrently
_nccl_worker_thread: Thread | None = None
_nccl_work_queue: Queue[tuple[Callable, tuple, dict, Future]] | None = None
_nccl_worker_lock = Lock()

# Server address (set at startup)
_server_host: str = "0.0.0.0"
_server_port: int = 8000

# Create Flask app
app = Flask(__name__)


def _init_nccl_worker():
    """Initialize the NCCL worker thread for executing non-/data/ endpoints."""
    global _nccl_worker_thread, _nccl_work_queue

    with _nccl_worker_lock:
        if _nccl_worker_thread is not None and _nccl_worker_thread.is_alive():
            return  # Already initialized

        _nccl_work_queue = Queue()

        def nccl_worker():
            """Worker thread that executes non-/data/ endpoints sequentially."""
            logger.info("NCCL worker thread started")
            while True:
                try:
                    work_item = _nccl_work_queue.get()
                    if work_item is None:  # Shutdown signal
                        logger.info("NCCL worker thread shutting down")
                        break

                    func, args, kwargs, future = work_item
                    try:
                        result = func(*args, **kwargs)
                        future.set_result(result)
                    except Exception as e:
                        future.set_exception(e)
                    finally:
                        _nccl_work_queue.task_done()
                except Exception as e:
                    logger.error(f"Error in NCCL worker thread: {e}")
                    if work_item and len(work_item) > 3:
                        work_item[3].set_exception(e)

        _nccl_worker_thread = Thread(target=nccl_worker, daemon=True, name="NCCLWorker")
        _nccl_worker_thread.start()
        logger.info("NCCL worker thread initialized")


def _submit_to_nccl_worker(func: Callable, *args, **kwargs) -> Any:
    """Submit a function to the NCCL worker thread for execution.

    This ensures all non-/data/ endpoints (which may involve NCCL operations)
    run in the same thread, maintaining NCCL compatibility while allowing
    /data/ requests to be processed concurrently in other threads.
    """
    global _nccl_work_queue

    _init_nccl_worker()

    future = Future()
    _nccl_work_queue.put((func, args, kwargs, future))
    return future.result()  # Block until result is available


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint to verify server is alive."""
    global _engine
    return jsonify({"status": "healthy", "engine_initialized": _engine is not None})


@app.route("/configure", methods=["POST"])
def configure():
    """Configure worker with experiment config.

    This endpoint is routed to the NCCL worker thread.
    """
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"detail": "Invalid JSON in request body"}), 400

        config = data.get("config")
        if config is None:
            return jsonify({"detail": "Missing 'config' field in request"}), 400

        role = data.get("role")
        if role is None:
            return jsonify({"detail": "Missing 'role' field in request"}), 400

        rank = data.get("rank")
        if rank is None:
            return jsonify({"detail": "Missing 'rank' field in request"}), 400

        config = deserialize_value(config)
        config: BaseExperimentConfig

        def execute_configure():
            name_resolve.reconfigure(config.cluster.name_resolve)
            seeding.set_random_seed(config.seed, key=f"{role}{rank}")
            return {
                "status": "success",
                "message": "Worker configured successful.",
                "result": None,
            }

        result = _submit_to_nccl_worker(execute_configure)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Unexpected error in configure: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route("/set_env", methods=["POST"])
def set_env():
    """Set environment variables for the worker process.

    This endpoint is routed to the NCCL worker thread.
    """
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Invalid JSON in request body"}), 400

        env_payload = data.get("env")
        if env_payload is None:
            return jsonify({"error": "Missing 'env' field in request"}), 400
        if not isinstance(env_payload, dict):
            return jsonify({"error": "'env' must be a dictionary"}), 400

        for key in env_payload.keys():
            if not isinstance(key, str):
                return (
                    jsonify(
                        {
                            "error": (
                                f"Environment variable name must be str, got {type(key)}"
                            )
                        }
                    ),
                    400,
                )

        def execute_set_env():
            for key, value in env_payload.items():
                os.environ[key] = str(value)
                logger.info(f"Set {key}={value}")
            return {"status": "success"}

        result = _submit_to_nccl_worker(execute_set_env)
        return jsonify(result)

    except Exception as e:
        logger.error(f"Unexpected error in set_env: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route("/create_engine", methods=["POST"])
def create_engine():
    """
    Create and initialize a TrainEngine or InferenceEngine instance on this worker.

    This endpoint is routed to the NCCL worker thread.

    Expected JSON payload:
    {
        "engine": "areal.engine.ppo.actor.FSDPPPOActor",  # Import path
        "init_args": [...],  # Positional arguments
        "init_kwargs": {
            "config": ...,  # Engine config
        }
    }

    Distributed training environment variables (RANK, WORLD_SIZE, MASTER_ADDR,
    MASTER_PORT, LOCAL_RANK, etc.) should be configured via the `/set_env`
    endpoint before invoking this endpoint.
    """
    global _engine

    try:
        # Parse request in main thread (has Flask request context)
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Invalid JSON in request body"}), 400

        engine_path = data.get("engine")
        # Deserialize init_args and init_kwargs (may contain tensors or dataclasses)
        init_args = deserialize_value(data.get("init_args", []))
        init_kwargs = deserialize_value(data.get("init_kwargs", {}))

        if not engine_path:
            return jsonify({"error": "Missing 'engine' field in request"}), 400

        # Dynamic import (can be done in main thread)
        try:
            engine_class = import_from_string(engine_path)

            # Validate that the class is a TrainEngine or InferenceEngine
            if not issubclass(engine_class, TrainEngine) and not issubclass(
                engine_class, InferenceEngine
            ):
                raise TypeError(
                    f"Engine class must be a subclass of TrainEngine or InferenceEngine, "
                    f"got {engine_class}.."
                )
        except (ValueError, ImportError, AttributeError) as e:
            logger.error(f"Failed to import engine '{engine_path}': {e}")
            return (
                jsonify(
                    {"error": f"Failed to import engine '{engine_path}': {str(e)}"}
                ),
                400,
            )
        except TypeError as e:
            logger.error(f"Invalid engine type: {e}")
            return jsonify({"error": str(e)}), 400

        # Instantiate engine in NCCL worker thread (may involve NCCL initialization)
        def create_engine_in_nccl_thread():
            """Create engine in NCCL worker thread."""
            try:
                engine = engine_class(*init_args, **init_kwargs)
                logger.info(f"Engine '{engine_path}' instantiated successfully")
                return engine
            except Exception as e:
                logger.error(
                    f"Failed to instantiate engine: {e}\n{traceback.format_exc()}"
                )
                raise

        try:
            _engine = _submit_to_nccl_worker(create_engine_in_nccl_thread)
            return jsonify(
                {
                    "status": "success",
                    "message": f"Engine '{engine_path}' created and initialized",
                    "result": None,
                }
            )
        except Exception as e:
            return jsonify({"error": f"Failed to instantiate engine: {str(e)}"}), 500

    except Exception as e:
        logger.error(
            f"Unexpected error in create_engine: {e}\n{traceback.format_exc()}"
        )
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route("/call", methods=["POST"])
def call_engine_method():
    """
    Call a method on the engine instance.

    This endpoint is routed to the NCCL worker thread to ensure
    all NCCL operations run in the same thread, preventing conflicts.

    Expected JSON payload:
    {
        "method": "train_batch",
        "args": [...],
        "kwargs": {...}
    }
    """
    global _engine

    if _engine is None:
        return (
            jsonify({"error": "Engine not initialized. Call /create_engine first."}),
            503,
        )

    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Invalid JSON in request body"}), 400

        method_name = data.get("method")
        args = data.get("args", [])
        kwargs = data.get("kwargs", {})

        if not method_name:
            return jsonify({"error": "Missing 'method' field in request"}), 400

        args = deserialize_value(args)
        kwargs = deserialize_value(kwargs)
        should_broadcast = kwargs.pop("should_broadcast", True)
        should_return_distributed_batch = kwargs.pop("return_distributed_batch", False)
        result_key = kwargs.pop("result_key", None)
        task_id = kwargs.pop("task_id", None)

        # Extract input batch metadata before resolving
        input_metadata_list = []
        input_metadata_list.extend(_extract_input_batch_metadata(args))
        input_metadata_list.extend(_extract_input_batch_metadata(kwargs))
        assert len(input_metadata_list) <= 1, (
            "Only one input batch metadata is supported"
        )
        input_batch_metadata = input_metadata_list[0] if input_metadata_list else None

        try:
            logger.info(
                f"Resolving batch metadata for method '{method_name}', args: {args}, kwargs: {kwargs}"
            )
            args = _resolve_batch_metadata(args)
            kwargs = _resolve_batch_metadata(kwargs)
        except Exception as e:
            logger.error(
                f"Resolving batch metadata for method '{method_name}' failed: {e}\n{traceback.format_exc()}"
            )
            return (
                jsonify(
                    {"error": f"Metadata resolution '{method_name}' failed: {str(e)}"}
                ),
                500,
            )

        def execute_in_nccl_thread():
            try:
                if should_broadcast and isinstance(_engine, TrainEngine):
                    logger.info(
                        f"Broadcasting data for TrainEngine method: {method_name}"
                    )

                    args_bcast = tensor_container_to(
                        args, current_platform.current_device()
                    )
                    args_bcast = broadcast_tensor_container(
                        args_bcast,
                        src_rank=_engine.current_data_parallel_head(),
                        group=_engine.context_and_model_parallel_group,
                    )
                    kwargs_bcast = tensor_container_to(
                        kwargs, current_platform.current_device()
                    )
                    kwargs_bcast = broadcast_tensor_container(
                        kwargs_bcast,
                        src_rank=_engine.current_data_parallel_head(),
                        group=_engine.context_and_model_parallel_group,
                    )
                    logger.info("Broadcasting data done.")
                else:
                    args_bcast = args
                    kwargs_bcast = kwargs

                logger.info(f"Calling engine method: {method_name}")
                method = getattr(_engine, method_name)
                result = method(*args_bcast, **kwargs_bcast)

                # Handle update weights future
                if isinstance(result, Future):
                    logger.info("Waiting for update weights future")
                    result = result.result()
                    logger.info("Update weights future done")

                if should_return_distributed_batch:
                    result = _handle_distributed_batch_return(
                        result,
                        result_key,
                        task_id=task_id,
                        input_batch_metadata=input_batch_metadata,
                    )
                    logger.debug("Handling distributed batch memory return")

                return result
            except AttributeError as e:
                logger.error(f"Method '{method_name}' not found on engine: {e}")
                raise ValueError(f"Engine does not have method '{method_name}'")
            except Exception as e:
                logger.error(
                    f"Engine method '{method_name}' failed: {e}\n{traceback.format_exc()}"
                )
                raise

        # Submit to NCCL worker thread
        try:
            result = _submit_to_nccl_worker(execute_in_nccl_thread)
        except Exception as e:
            error_msg = str(e)
            if "Engine does not have method" in error_msg:
                return (
                    jsonify({"error": error_msg}),
                    400,
                )
            return (
                jsonify(
                    {"error": f"Engine method '{method_name}' failed: {error_msg}"}
                ),
                500,
            )

        serialized_result = serialize_value(result)
        return jsonify({"status": "success", "result": serialized_result})

    except Exception as e:
        logger.error(f"Unexpected error in call: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


# ==================== Batch Reletad Functions ====================
def _ensure_tensor_on_cpu(value: Any) -> Any:
    """Recursively ensure all tensors in a value are on CPU."""
    if isinstance(value, torch.Tensor):
        # Move to CPU if on GPU, detach to avoid gradient tracking
        if value.is_cuda:
            return value.detach().cpu()
        return value.detach() if value.requires_grad else value
    elif isinstance(value, dict):
        return {k: _ensure_tensor_on_cpu(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        converted = [_ensure_tensor_on_cpu(item) for item in value]
        return type(value)(converted)
    else:
        return value


def _is_batch_metadata(data: Any) -> bool:
    """Check if data is a DistributedBatchMemory metadata wrapper."""
    return isinstance(data, dict) and data.get("__distributed_batch_metadata__")


def _get_batch_metadata(data: Any) -> BatchMetadata | None:
    """Extract BatchMetadata from a DistributedBatchMemory metadata wrapper."""
    if _is_batch_metadata(data):
        return data.get("metadata")
    return None


def _extract_input_batch_metadata(data: Any) -> list[BatchMetadata]:
    """Extract all DistributedBatchMemory metadata from input data."""
    metadata_list = []

    if isinstance(data, dict):
        metadata = _get_batch_metadata(data)
        if metadata is not None:
            metadata_list.append(metadata)
        else:
            # Recursively check dict values
            for v in data.values():
                metadata_list.extend(_extract_input_batch_metadata(v))
    elif isinstance(data, (list, tuple)):
        # Recursively check list/tuple elements
        for item in data:
            metadata_list.extend(_extract_input_batch_metadata(item))

    return metadata_list


async def _aresolve_batch_metadata(data: Any) -> Any:
    """Resolve DistributedBatchMemory metadata to actual data.

    Recursively traverses data structures (same traversal logic as _extract_input_batch_metadata)
    and replaces DistributedBatchMemory metadata with actual data fetched from remote nodes.
    """
    if isinstance(data, dict):
        metadata = _get_batch_metadata(data)
        if metadata is not None:
            batch = DistributedBatchMemory.from_metadata(metadata)
            return await batch.aget_data()
        else:
            # Recursively resolve dict values
            resolved_items = await asyncio.gather(
                *[_aresolve_batch_metadata(v) for v in data.values()]
            )
            return {k: v for k, v in zip(data.keys(), resolved_items)}
    elif isinstance(data, (list, tuple)):
        # Recursively resolve list/tuple elements
        resolved_items = await asyncio.gather(
            *[_aresolve_batch_metadata(item) for item in data]
        )
        return type(data)(resolved_items)
    else:
        return data


def _resolve_batch_metadata(data: Any) -> Any:
    """Resolve DistributedBatch metadata to actual data."""

    def run_in_thread():
        """Run async resolution in a new thread with a new event loop."""
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            return new_loop.run_until_complete(_aresolve_batch_metadata(data))
        except Exception as e:
            logger.error(
                f"Error resolving batch metadata in thread: {e}\n{traceback.format_exc()}"
            )
            raise
        finally:
            new_loop.close()

    with ThreadPoolExecutor() as executor:
        future = executor.submit(run_in_thread)
        return future.result()


def _create_matched_batch_metadata(
    data_to_store: dict[str, Tensor],
    input_batch_metadata: BatchMetadata,
    node_id: str,
    node_addr: str,
) -> DistributedBatchMemory:
    """Create batch metadata matching input structure with tensor splitting.

    This function creates shards matching the input metadata structure:
    1. Groups input shards by task_id
    2. For each task_id, creates output shards for each key in data_to_store
    3. Splits tensors along dimension 0 if sizes don't match

    Parameters
    ----------
    data_to_store : dict[str, Tensor]
        Result data to split and store (keys become shard_id.key)
    input_batch_metadata : BatchMetadata
        Input batch metadata to match
    node_id : str
        Current node identifier
    node_addr : str
        Current node address
    task_id : str | None
        Optional task ID prefix (not used, kept for compatibility)

    Returns
    -------
    DistributedBatchMemory
        Batch with metadata matching input structure
    """
    global _batch_storage, _batch_storage_lock, _batch_storage_stats

    # Group input shards by task_id and calculate offsets
    # Process shards in order to maintain correct offset calculation
    task_id_to_shards: dict[str, list[ShardMetadata]] = {}
    task_id_to_size: dict[str, int] = {}
    task_id_offsets: dict[str, int] = {}

    current_offset = 0
    for shard in input_batch_metadata.shards:
        task_id = shard.shard_id.task_id
        if task_id not in task_id_to_shards:
            task_id_to_shards[task_id] = []
            task_id_offsets[task_id] = current_offset
            task_id_to_size[task_id] = 0

        task_id_to_shards[task_id].append(shard)
        shard_size = shard.tensor_metadata.shape[0]
        # Only count the first shard of each task_id for offset calculation
        # (assuming all shards with same task_id have same batch_size)
        if len(task_id_to_shards[task_id]) == 1:
            task_id_to_size[task_id] = shard_size
            current_offset += shard_size

    # Get result keys (these will become the new shard_id.key values)
    result_keys = list(data_to_store.keys())
    if not result_keys:
        raise ValueError("data_to_store must contain at least one tensor")

    # Create output shards
    output_shards = []

    for task_id, input_shards in task_id_to_shards.items():
        # Calculate total batch size for this task_id group
        # Use the size from the first shard (all shards with same task_id should have same size)
        total_size = task_id_to_size[task_id]
        offset = task_id_offsets[task_id]

        # For each result key, create a shard with the same task_id
        for result_key in result_keys:
            tensor = data_to_store[result_key]

            # Slice the tensor for this task_id group
            sliced_tensor = tensor[offset : offset + total_size].clone().cpu()

            # Create shard_id with same task_id but new key
            shard_id = ShardId(task_id=task_id, key=result_key)

            # Store shard data (single tensor, not dict)
            with _batch_storage_lock:
                _batch_storage[shard_id] = sliced_tensor
                serialized_data = serialize_value(sliced_tensor)
                data_bytes = orjson.dumps(serialized_data)
                _batch_storage_stats[shard_id] = len(data_bytes)

            logger.debug(
                f"Stored shard {shard_id} (size={len(data_bytes)} bytes, "
                f"batch_size={total_size}, node_addr={node_addr})"
            )

            # Create tensor metadata
            tensor_metadata = TensorMetadata(
                shape=tuple(sliced_tensor.shape),
                dtype=str(sliced_tensor.dtype),
                device=str(sliced_tensor.device),
            )

            # Create shard metadata
            output_shard = ShardMetadata(
                node_id=node_id,
                node_addr=node_addr,
                shard_id=shard_id,
                tensor_metadata=tensor_metadata,
            )
            output_shards.append(output_shard)

    # Create batch metadata with matched structure
    batch_metadata = BatchMetadata(
        batch_id=str(uuid.uuid4()),
        shards=output_shards,
    )

    batch = DistributedBatchMemory.from_metadata(batch_metadata)
    logger.info(
        f"Created DistributedBatchMemory matching input: {batch_metadata.batch_id}, "
        f"num_shards={len(output_shards)}, shard_ids={[s.shard_id for s in output_shards]}"
    )

    return batch


def _handle_distributed_batch_return(
    result: Any,
    result_key: str | None,
    task_id: str | None = None,
    input_batch_metadata: BatchMetadata | None = None,
) -> Any:
    """Handle distributed batch memory return.

    When the return value is one of the following types, automatically write to
    local `_batch_storage` and return `DistributedBatchMemory` (or its list)
    metadata:

    - ``torch.Tensor``
    - ``dict[str, torch.Tensor]``
    - ``list[dict[str, torch.Tensor]]``
    """
    global _batch_storage, _batch_storage_lock, _batch_storage_stats

    # Handle list: recursively process each element
    if isinstance(result, list):
        return [
            _handle_distributed_batch_return(
                r,
                result_key,
                task_id=task_id,
                input_batch_metadata=input_batch_metadata,
            )
            for r in result
        ]

    # Check if result is Tensor or dict[str, Tensor]
    data_to_store = None
    if isinstance(result, torch.Tensor):
        if result_key is None:
            result_key = "default_key"
        data_to_store = {result_key: result}
    elif isinstance(result, dict) and all(
        isinstance(v, torch.Tensor) for v in result.values()
    ):
        data_to_store = result

    if data_to_store is None:
        return result

    # Get node info
    node_id = os.environ.get("HOSTNAME", "unknown")
    rank = int(os.environ.get("RANK", "0"))
    node_id = f"{node_id}_rank{rank}"

    # Get node address
    global _server_host, _server_port
    node_addr = f"{_server_host}:{_server_port}"

    # If input_batch_metadata is not provided, create a fake one
    # to reuse _create_matched_batch_metadata logic
    if input_batch_metadata is None or not input_batch_metadata.shards:
        task_id = task_id or str(uuid.uuid4())

        # Find the first tensor to determine batch size
        first_tensor = next(
            (v for v in data_to_store.values() if isinstance(v, torch.Tensor)), None
        )
        if first_tensor is None:
            return result

        batch_size = first_tensor.shape[0] if len(first_tensor.shape) > 0 else 1

        # Create a single fake shard with dummy key
        # This will be used to create one task_id group
        fake_shard_id = ShardId(task_id=task_id, key="__dummy_input__")
        fake_tensor_metadata = TensorMetadata(
            shape=(batch_size,),
            dtype=str(first_tensor.dtype),
            device=str(first_tensor.device),
        )
        fake_shard = ShardMetadata(
            node_id=node_id,
            node_addr=node_addr,
            shard_id=fake_shard_id,
            tensor_metadata=fake_tensor_metadata,
        )

        input_batch_metadata = BatchMetadata(
            batch_id=str(uuid.uuid4()),
            shards=[fake_shard],
        )

    return _create_matched_batch_metadata(
        data_to_store,
        input_batch_metadata,
        node_id,
        node_addr,
    )


# ==================== Batch Data Storage Endpoints ====================
@app.route("/data/<shard_id>", methods=["PUT"])
def store_batch_data(shard_id: str):
    """Store batch data shard."""
    global _batch_storage, _batch_storage_lock, _batch_storage_stats

    try:
        # Convert string shard_id to ShardId
        shard_id_obj = ShardId.from_string(shard_id)

        data_bytes = request.get_data()
        # Deserialize from orjson, then deserialize_value to restore tensors
        serialized_data = orjson.loads(data_bytes)
        data = deserialize_value(serialized_data)

        cpu_data = _ensure_tensor_on_cpu(data)

        with _batch_storage_lock:
            _batch_storage[shard_id_obj] = cpu_data
            _batch_storage_stats[shard_id_obj] = len(data_bytes)

        logger.debug(
            f"Stored batch shard {shard_id_obj} (size={len(data_bytes)} bytes)"
        )
        # Echo back the original shard_id string to avoid adding default keys
        return jsonify({"status": "ok", "shard_id": shard_id})

    except Exception as e:
        logger.error(f"Error storing batch shard {shard_id_obj}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/data/<shard_id>", methods=["GET"])
def retrieve_batch_data(shard_id: str):
    """Retrieve batch data shard."""
    global _batch_storage, _batch_storage_lock

    # Convert string shard_id to ShardId
    shard_id_obj = ShardId.from_string(shard_id)

    logger.debug(f"Received data get request for shard {shard_id_obj}")
    try:
        with _batch_storage_lock:
            if shard_id_obj not in _batch_storage:
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": f"Shard {shard_id_obj} not found",
                        }
                    ),
                    404,
                )

            data = _batch_storage[shard_id_obj]

        # Serialize using serialize_value to handle tensors, then encode with orjson
        serialized_data = serialize_value(data)
        data_bytes = orjson.dumps(serialized_data)

        logger.info(
            f"Retrieved batch shard {shard_id_obj} (size={len(data_bytes)} bytes)"
        )
        return Response(data_bytes, mimetype="application/octet-stream")

    except Exception as e:
        logger.error(f"Error retrieving batch shard {shard_id_obj}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/data/clear", methods=["DELETE"])
def clear_batch_data():
    """Clear specified batch data shards.

    Expected JSON payload:
    {
        "shard_ids": ["id1", "id2", ...]
    }
    """
    global _batch_storage, _batch_storage_lock, _batch_storage_stats

    try:
        data = request.get_json(silent=True) or {}
        shard_ids = data.get("shard_ids", [])
        if not isinstance(shard_ids, list):
            return (
                jsonify({"status": "error", "message": "'shard_ids' must be a list"}),
                400,
            )
        # Convert string shard_ids to ShardId objects
        shard_id_objs = []
        for sid in shard_ids:
            if isinstance(sid, str):
                shard_id_objs.append(ShardId.from_string(sid))

        if not shard_id_objs:
            return jsonify({"status": "ok", "cleared_count": 0})

        with _batch_storage_lock:
            cleared_count = 0
            for shard_id_obj in shard_id_objs:
                if shard_id_obj in _batch_storage:
                    del _batch_storage[shard_id_obj]
                    _batch_storage_stats.pop(shard_id_obj, None)
                    cleared_count += 1

        logger.info(
            f"Cleared {cleared_count} batch shards: {[str(sid) for sid in shard_id_objs]}"
        )
        return jsonify({"status": "ok", "cleared_count": cleared_count})

    except Exception as e:
        logger.error(f"Error clearing batch data: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/data/stats", methods=["GET"])
def batch_data_stats():
    """Get batch data storage statistics."""
    global _batch_storage, _batch_storage_lock, _batch_storage_stats

    try:
        with _batch_storage_lock:
            total_shards = len(_batch_storage)
            total_size = sum(_batch_storage_stats.values())

        return jsonify(
            {
                "status": "ok",
                "total_shards": total_shards,
                "total_size_bytes": total_size,
            }
        )
    except Exception as e:
        logger.error(f"Error getting batch data stats: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# ==================== Cleanup ====================


def cleanup_engine():
    """Clean up engine on shutdown."""
    global _engine
    if _engine is not None:
        try:
            _engine.destroy()
            logger.info("Engine destroyed successfully")
        except Exception as e:
            logger.error(f"Error destroying engine: {e}")
        _engine = None


def cleanup_batch_storage():
    """Clean up batch storage on shutdown."""
    global _batch_storage, _batch_storage_lock, _batch_storage_stats
    with _batch_storage_lock:
        _batch_storage.clear()
        _batch_storage_stats.clear()
    logger.info("Batch storage cleared")


def cleanup_nccl_worker():
    """Clean up NCCL worker thread."""
    global _nccl_worker_thread, _nccl_work_queue

    with _nccl_worker_lock:
        if _nccl_work_queue is not None:
            # Send shutdown signal
            _nccl_work_queue.put(None)
            _nccl_work_queue = None

        if _nccl_worker_thread is not None:
            _nccl_worker_thread.join(timeout=5.0)
            if _nccl_worker_thread.is_alive():
                logger.warning("NCCL worker thread did not shut down gracefully")
            _nccl_worker_thread = None
            logger.info("NCCL worker thread cleaned up")


def main():
    """Main entry point for the sync RPC server."""
    parser = argparse.ArgumentParser(
        description="AReaL Sync RPC Server for TrainEngine/InferenceEngine"
    )
    parser.add_argument("--port", type=int, required=True, help="Port to serve on")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--werkzeug-log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level for Werkzeug (Flask's WSGI server). Default: WARNING",
    )

    args, _ = parser.parse_known_args()

    # Configure Werkzeug logging
    werkzeug_logger = stdlib_logging.getLogger("werkzeug")
    werkzeug_logger.setLevel(getattr(stdlib_logging, args.werkzeug_log_level))

    # Set global server address variables
    global _server_host, _server_port
    _server_host = args.host
    if _server_host == "0.0.0.0":
        _server_host = socket.gethostbyname(socket.gethostname())
    _server_port = args.port

    logger.info(f"Starting sync RPC server on {args.host}:{args.port}")
    logger.info(f"Werkzeug log level: {args.werkzeug_log_level}")

    # Run Flask app with multi-threaded mode
    # /data/ endpoints are processed in request threads (concurrent)
    # /call and other non-/data/ endpoints are routed to NCCL worker thread
    # This ensures NCCL compatibility while allowing /data/ requests to be processed concurrently
    try:
        app.run(
            host=args.host,
            port=args.port,
            threaded=True,  # Multi-threaded for concurrent /data/ request handling
            processes=1,  # Single process
            debug=False,
            use_reloader=False,
        )
    except KeyboardInterrupt:
        logger.info("Shutting down sync RPC server")
    finally:
        cleanup_nccl_worker()
        cleanup_engine()
        cleanup_batch_storage()


if __name__ == "__main__":
    main()
