import argparse
import asyncio
import io
import os
import pickle
import socket
import traceback
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from queue import Queue
from threading import Lock, Thread
from typing import Any

from flask import Flask, Response, jsonify, request

from areal.api.cli_args import BaseExperimentConfig
from areal.api.engine_api import InferenceEngine, TrainEngine
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
# Storage: shard_id -> (global_step, data)
_batch_storage: dict[str, tuple[int, dict]] = {}
_batch_storage_lock = Lock()
_batch_storage_stats: dict[str, int] = defaultdict(int)

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
        # Parse request in main thread (has Flask request context)
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

        # Execute in NCCL worker thread
        def execute_configure():
            """Execute configure in NCCL worker thread."""
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
        # Parse request in main thread (has Flask request context)
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Invalid JSON in request body"}), 400

        env_payload = data.get("env")
        if env_payload is None:
            return jsonify({"error": "Missing 'env' field in request"}), 400
        if not isinstance(env_payload, dict):
            return jsonify({"error": "'env' must be a dictionary"}), 400

        # Validate keys
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

        # Execute in NCCL worker thread
        def execute_set_env():
            """Execute set_env in NCCL worker thread."""
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
        # Parse request in main thread (has Flask request context)
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Invalid JSON in request body"}), 400

        method_name = data.get("method")
        args = data.get("args", [])
        kwargs = data.get("kwargs", {})

        if not method_name:
            return jsonify({"error": "Missing 'method' field in request"}), 400

        logger.info(f"22 args: {method_name}")
        # Deserialize args and kwargs (convert SerializedTensor dicts to tensors)
        args = deserialize_value(args)
        kwargs = deserialize_value(kwargs)
        should_bcast = kwargs.pop("_should_bcast", True)
        should_return_distributed_batch = kwargs.pop(
            "_should_return_distributed_batch", False
        )
        distributed_batch_target_key = kwargs.pop("_distributed_batch_target_key", None)

        # Resolve batch metadata (may involve async operations)
        try:
            args = _resolve_batch_metadata(args)
            kwargs = _resolve_batch_metadata(kwargs)
            if method_name == "compute_advantages":
                logger.info(f"compute_advantages args: {args}")
                logger.info(f"compute_advantages kwargs: {kwargs}")
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

        # Execute in NCCL worker thread to ensure thread safety
        def execute_in_nccl_thread():
            """Execute the method call in NCCL worker thread."""
            try:
                if should_bcast and isinstance(_engine, TrainEngine):
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

                # Call method directly
                logger.info(f"Calling engine method: {method_name}")
                method = getattr(_engine, method_name)
                result = method(*args_bcast, **kwargs_bcast)

                # HACK: handle update weights future
                if isinstance(result, Future):
                    logger.info("Waiting for update weights future")
                    result = result.result()
                    logger.info("Update weights future done")

                # Handle distributed batch memory return
                if should_return_distributed_batch:
                    result = _handle_distributed_batch_return(
                        result,
                        distributed_batch_target_key,
                        _engine,
                    )
                    logger.info(f"Handling distributed batch memory return: {result}")

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

        # Serialize result (convert tensors to SerializedTensor dicts)
        serialized_result = serialize_value(result)

        return jsonify({"status": "success", "result": serialized_result})

    except Exception as e:
        logger.error(f"Unexpected error in call: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


async def _aresolve_batch_metadata(data: Any) -> Any:
    """Async version of _resolve_batch_metadata.

    Resolve DistributedBatch metadata to actual data using async data fetching.
    """
    if isinstance(data, dict) and data.get("__distributed_batch_metadata__"):
        # This is a metadata reference, fetch actual data
        from areal.controller.batch import DistributedBatchMemory

        logger.info(f"11 Resolving batch metadata: {data}")
        metadata = data.get("metadata")
        logger.info(f"11 metadata: {metadata}")
        if metadata is not None:
            batch = DistributedBatchMemory.from_metadata(metadata)
            # Worker fetches its own data using async method
            logger.info(f"11 batch: {batch}")
            xxx = await batch.aget_data()
            for key, value in xxx.items():
                logger.info(
                    f"11 key: {key}, shape: {value.shape}, dtype: {value.dtype}"
                )
            return xxx
        return {}
    elif isinstance(data, (list, tuple)):
        # Recursively resolve list/tuple elements
        resolved_items = await asyncio.gather(
            *[_aresolve_batch_metadata(item) for item in data]
        )
        return type(data)(resolved_items)
    elif isinstance(data, dict):
        # Recursively resolve dict values
        resolved_items = await asyncio.gather(
            *[_aresolve_batch_metadata(v) for v in data.values()]
        )
        return {k: v for k, v in zip(data.keys(), resolved_items)}
    else:
        return data


def _resolve_batch_metadata(data: Any) -> Any:
    """Resolve DistributedBatch metadata to actual data.

    If data contains metadata markers, fetch the actual data from batch storage.
    Otherwise, return data as-is.

    This function always uses a thread pool with a new event loop to run async code,
    avoiding event loop conflicts in any context.
    """

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


def _handle_distributed_batch_return(
    result: Any,
    distributed_batch_target_key: str | None,
    engine: TrainEngine | InferenceEngine,
) -> Any:
    """Handle distributed batch memory return.

    当返回值为以下类型之一时, 自动写入本地 `_batch_storage` 并返回
    `DistributedBatchMemory`(或其列表)的 **metadata**:

    - ``torch.Tensor``
    - ``dict[str, torch.Tensor]``
    - ``list[dict[str, torch.Tensor]]``

    其他类型保持原样返回。

    Parameters
    ----------
    result : Any
        The result from engine method
    distributed_batch_target_key : str | None
        Key to use when converting Tensor to dict
    engine : TrainEngine | InferenceEngine
        Engine instance (to get version/node info)

    Returns
    -------
    Any
        DistributedBatchMemory metadata or original result
    """
    global _batch_storage, _batch_storage_lock, _batch_storage_stats

    import torch

    # 1) 如果是 list, 逐元素递归处理, 支持 list[dict[str, Tensor]] 等场景
    if isinstance(result, list):
        return [
            _handle_distributed_batch_return(r, distributed_batch_target_key, engine)
            for r in result
        ]

    # 2) 单个 Tensor 或 dict[str, Tensor]
    # Check if result is dict[str, Tensor] or Tensor
    should_store = False
    data_to_store = None
    if isinstance(result, torch.Tensor):
        # Convert Tensor to dict
        if distributed_batch_target_key is None:
            distributed_batch_target_key = "data"
        data_to_store = {distributed_batch_target_key: result}
        should_store = True
    elif isinstance(result, dict) and any(
        isinstance(v, torch.Tensor) for v in result.values()
    ):
        # Already a dict with tensors
        data_to_store = result
        should_store = True

    if not should_store or data_to_store is None:
        # Not a tensor/dict result, return as-is
        return result
    # Get global_step from engine
    try:
        global_step = engine.get_version()
    except Exception:
        global_step = 0
        logger.warning("Failed to get version from engine, using global_step=0")

    # Get node info
    node_id = os.environ.get("HOSTNAME", "unknown")
    rank = int(os.environ.get("RANK", "0"))
    node_id = f"{node_id}_rank{rank}"

    # Get node address: use global server host and port (set at startup)
    global _server_host, _server_port
    node_addr = f"{_server_host}:{_server_port}"

    # Generate shard ID
    import uuid

    shard_id = str(uuid.uuid4())

    # Store data in local batch storage
    with _batch_storage_lock:
        _batch_storage[shard_id] = (global_step, data_to_store)
        # Estimate size
        data_bytes = pickle.dumps(data_to_store)
        _batch_storage_stats[shard_id] = len(data_bytes)
    logger.info(
        f"Stored result as shard {shard_id} (step={global_step}, size={len(data_bytes)} bytes, node_addr={node_addr})"
    )

    # Create metadata
    from areal.controller.batch import DistributedBatchMemory
    from areal.controller.batch_metadata import (
        BatchMetadata,
        ScalarMetadata,
        ShardMetadata,
        TensorMetadata,
    )

    # Infer batch size and fields
    first_value = next(iter(data_to_store.values()))
    if isinstance(first_value, torch.Tensor):
        batch_size = first_value.shape[0]
    elif isinstance(first_value, list):
        batch_size = len(first_value)
    else:
        batch_size = 1

    # Create field metadata
    fields = {}
    for key, value in data_to_store.items():
        if isinstance(value, torch.Tensor):
            fields[key] = TensorMetadata(
                shape=tuple(value.shape),
                dtype=str(value.dtype),
                device=str(value.device),
            )
        elif isinstance(value, list):
            fields[key] = ScalarMetadata(value_type="list", length=len(value))
        else:
            fields[key] = ScalarMetadata(value_type=type(value).__name__, length=1)

    # Create shard metadata
    shard = ShardMetadata(
        node_id=node_id,
        node_addr=node_addr,
        shard_id=shard_id,
        batch_size=batch_size,
        offset=0,
        fields=fields,
    )

    # Create batch metadata
    batch_metadata = BatchMetadata(
        batch_id=str(uuid.uuid4()),
        global_step=global_step,
        total_batch_size=batch_size,
        shards=[shard],
    )

    # Create DistributedBatchMemory with metadata
    batch = DistributedBatchMemory.from_metadata(batch_metadata)

    logger.debug(
        f"Created DistributedBatchMemory with metadata: {batch_metadata.batch_id}, "
        f"batch_size={batch_size}"
    )

    # Return the batch (will be serialized with metadata)
    return batch


# ==================== Batch Data Storage Endpoints ====================
@app.route("/data/<shard_id>", methods=["PUT"])
def store_batch_data(shard_id: str):
    """Store batch data shard.

    URL: /data/{shard_id}?global_step={step}
    Body: Pickled dict[str, torch.Tensor | Any]
    """
    global _batch_storage, _batch_storage_lock, _batch_storage_stats

    try:
        global_step = int(request.args.get("global_step", 0))

        # Read and deserialize data
        data_bytes = request.get_data()
        buffer = io.BytesIO(data_bytes)
        data = pickle.load(buffer)

        # Store with lock
        with _batch_storage_lock:
            _batch_storage[shard_id] = (global_step, data)
            _batch_storage_stats[shard_id] = len(data_bytes)

        logger.debug(
            f"Stored batch shard {shard_id} (step={global_step}, size={len(data_bytes)} bytes)"
        )
        return jsonify({"status": "ok", "shard_id": shard_id})

    except Exception as e:
        logger.error(f"Error storing batch shard {shard_id}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/data/<shard_id>", methods=["GET"])
def retrieve_batch_data(shard_id: str):
    """Retrieve batch data shard.

    URL: /data/{shard_id}
    Response: Pickled dict[str, torch.Tensor | Any]
    """
    global _batch_storage, _batch_storage_lock

    logger.info(f"Received data get request, batch shard {shard_id}")
    try:
        with _batch_storage_lock:
            if shard_id not in _batch_storage:
                return (
                    jsonify(
                        {"status": "error", "message": f"Shard {shard_id} not found"}
                    ),
                    404,
                )

            global_step, data = _batch_storage[shard_id]

        # Serialize data
        buffer = io.BytesIO()
        pickle.dump(data, buffer)
        data_bytes = buffer.getvalue()

        logger.info(
            f"Retrieved batch shard {shard_id} (step={global_step}, size={len(data_bytes)} bytes)"
        )
        return Response(data_bytes, mimetype="application/octet-stream")

    except Exception as e:
        logger.error(f"Error retrieving batch shard {shard_id}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/data/clear", methods=["DELETE"])
def clear_batch_data():
    """Clear old batch data.

    URL: /data/clear?global_step={step}
    Clears all shards with global_step < step
    """
    global _batch_storage, _batch_storage_lock, _batch_storage_stats

    try:
        global_step = int(request.args.get("global_step", 0))

        with _batch_storage_lock:
            shards_to_remove = [
                shard_id
                for shard_id, (step, _) in _batch_storage.items()
                if step < global_step
            ]

            for shard_id in shards_to_remove:
                del _batch_storage[shard_id]
                if shard_id in _batch_storage_stats:
                    del _batch_storage_stats[shard_id]

        logger.info(
            f"Cleared {len(shards_to_remove)} batch shards with step < {global_step}"
        )
        return jsonify({"status": "ok", "cleared_count": len(shards_to_remove)})

    except Exception as e:
        logger.error(f"Error clearing batch data: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/data/stats", methods=["GET"])
def batch_data_stats():
    """Get batch data storage statistics.

    URL: /data/stats
    """
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
    import logging as stdlib_logging

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
