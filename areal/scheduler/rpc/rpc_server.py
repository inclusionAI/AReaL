import argparse
import logging as stdlib_logging
import os
import traceback
from collections.abc import Callable
from concurrent.futures import Future
from queue import Queue
from threading import Lock, Thread
from typing import Any

import orjson
from flask import Flask, Response, jsonify, request
from werkzeug.serving import make_server

from areal.api.cli_args import BaseExperimentConfig, NameResolveConfig
from areal.api.engine_api import InferenceEngine, TrainEngine
from areal.platforms import current_platform
from areal.scheduler.rpc import rtensor
from areal.scheduler.rpc.rtensor import RTensor
from areal.scheduler.rpc.serialization import (
    deserialize_value,
    serialize_value,
)
from areal.utils import logging, name_resolve, names, perf_tracer, seeding
from areal.utils.data import (
    broadcast_tensor_container,
    tensor_container_to,
)
from areal.utils.dynamic_import import import_from_string
from areal.utils.network import gethostip

logger = logging.getLogger("SyncRPCServer")

# Global engine instance - must be TrainEngine or InferenceEngine
_engine: TrainEngine | InferenceEngine | None = None

_role: str | None = None

# Engine thread for executing all engine-related endpoints serially
# This ensures NCCL compatibility by running engine operations in a single thread,
# while allowing /data/ endpoints to be processed concurrently
_engine_thread: Thread | None = None
_engine_work_queue: Queue[tuple[Callable, tuple, dict, Future]] | None = None
_engine_thread_lock = Lock()

# Server address (set at startup)
_server_host: str = "0.0.0.0"
_server_port: int = 8000


# Create Flask app
app = Flask(__name__)


def _inject_special_args(method: str, args, kwargs):
    if method != "connect_engine":
        return args, kwargs

    engine_path = kwargs.pop("engine", None)
    if engine_path is None:
        raise ValueError("connect_engine requires 'engine' to be a class path string")

    EngineCls = import_from_string(engine_path)

    # MockInferenceEngine takes no args → safe
    engine = EngineCls()

    kwargs["engine"] = engine
    return args, kwargs


def _init_engine_thread():
    global _engine_thread, _engine_work_queue

    with _engine_thread_lock:
        if _engine_thread is not None and _engine_thread.is_alive():
            return  # Already initialized

        _engine_work_queue = Queue()

        def engine_worker():
            logger.info("Engine thread started")
            while True:
                try:
                    work_item = _engine_work_queue.get()
                    if work_item is None:  # Shutdown signal
                        logger.info("Engine thread shutting down")
                        break

                    func, args, kwargs, future = work_item
                    try:
                        result = func(*args, **kwargs)
                        future.set_result(result)
                    except Exception as e:
                        future.set_exception(e)
                    finally:
                        _engine_work_queue.task_done()
                except Exception as e:
                    logger.error(f"Error in engine thread: {e}")
                    if work_item and len(work_item) > 3:
                        work_item[3].set_exception(e)

        _engine_thread = Thread(target=engine_worker, daemon=True, name="EngineWorker")
        _engine_thread.start()
        logger.info("Engine thread initialized")


def _submit_to_engine_thread(func: Callable, *args, **kwargs) -> Any:
    global _engine_work_queue

    _init_engine_thread()

    future = Future()
    _engine_work_queue.put((func, args, kwargs, future))
    return future.result()  # Block until result is available


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint to verify server is alive."""
    global _engine
    return jsonify({"status": "healthy", "engine_initialized": _engine is not None})


@app.route("/configure", methods=["POST"])
def configure():
    """Configure worker with experiment config.

    This endpoint is routed to the engine thread for serial execution.
    """
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"detail": "Invalid JSON in request body"}), 400

        config = data.get("config")
        if config is None:
            return jsonify({"detail": "Missing 'config' field in request"}), 400

        rank = data.get("rank")
        if rank is None:
            return jsonify({"detail": "Missing 'rank' field in request"}), 400

        config = deserialize_value(config)
        config: BaseExperimentConfig

        def execute_configure():
            global _role
            seeding.set_random_seed(config.seed, key=f"{_role}{rank}")
            return {
                "status": "success",
                "message": "Worker configured successful.",
                "result": None,
            }

        result = _submit_to_engine_thread(execute_configure)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Unexpected error in configure: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route("/set_env", methods=["POST"])
def set_env():
    """Set environment variables for the worker process.

    This endpoint is routed to the engine thread for serial execution.
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

        result = _submit_to_engine_thread(execute_set_env)
        return jsonify(result)

    except Exception as e:
        logger.error(f"Unexpected error in set_env: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route("/create_engine", methods=["POST"])
def create_engine():
    """
    Create and initialize a TrainEngine or InferenceEngine instance on this worker.

    This endpoint is routed to the engine thread for serial execution.

    Expected JSON payload:
    {
        "engine": "areal.engine.ppo.actor.FSDPPPOActor",  # Import path
        "init_args": [...],  # Positional arguments
        "init_kwargs": {
            "config": ...,  # Engine config
        }
    }
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

        # Instantiate engine in engine thread (may involve NCCL initialization)
        def create_engine_in_engine_thread():
            """Create engine in engine thread."""
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
            _engine = _submit_to_engine_thread(create_engine_in_engine_thread)
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

    This endpoint is routed to the engine thread to ensure all engine operations
    run serially in the same thread, preventing NCCL conflicts.

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
        raw_args = data.get("args", [])
        raw_kwargs = data.get("kwargs", {})
        raw_args, raw_kwargs = _inject_special_args(method_name, raw_args, raw_kwargs)

        if not method_name:
            return jsonify({"error": "Missing 'method' field in request"}), 400

        # Deserialize data
        raw_args = deserialize_value(raw_args)
        raw_kwargs = deserialize_value(raw_kwargs)

        # Fetch remote tensors if any
        args = RTensor.localize(raw_args)
        kwargs = RTensor.localize(raw_kwargs)

        # FIXME: remove should_broadcast param
        should_broadcast = kwargs.pop("should_broadcast", True)

        def execute_in_engine_thread():
            try:
                if should_broadcast and isinstance(_engine, TrainEngine):
                    logger.debug(
                        f"Broadcasting data for TrainEngine method: {method_name}"
                    )

                    nonlocal raw_args, raw_kwargs
                    raw_args = broadcast_tensor_container(
                        raw_args,
                        src_rank=_engine.current_data_parallel_head(),
                        group=_engine.context_and_model_parallel_group,
                    )
                    raw_kwargs = broadcast_tensor_container(
                        raw_kwargs,
                        src_rank=_engine.current_data_parallel_head(),
                        group=_engine.context_and_model_parallel_group,
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
                    logger.debug("Broadcasting data done.")
                else:
                    args_bcast = args
                    kwargs_bcast = kwargs

                logger.debug(f"Calling engine method: {method_name}")

                # Determine trace category based on method name
                category = "misc"  # Default category
                method_lower = method_name.lower()
                if any(keyword in method_lower for keyword in ["submit", "wait"]):
                    category = "scheduler"
                elif any(
                    keyword in method_lower
                    for keyword in ["update_weights", "broadcast"]
                ):
                    category = "comm"
                elif any(keyword in method_lower for keyword in ["save", "load"]):
                    category = "io"
                elif any(
                    keyword in method_lower
                    for keyword in [
                        "train",
                        "eval",
                        "forward",
                        "compute",
                        "step",
                        "update",
                        "optimizer",
                        "zero_grad",
                        "lr_scheduler",
                    ]
                ):
                    category = "compute"

                # Wrap engine method call with perf_tracer
                with perf_tracer.trace_scope(
                    f"rpc.{method_name}",
                    category=category,
                    args={"method": method_name},
                ):
                    method = getattr(_engine, method_name)
                    result = method(*args_bcast, **kwargs_bcast)

                    # Handle update weights future
                    if isinstance(result, Future):
                        logger.debug("Waiting for update weights future")
                        result = result.result()
                        logger.debug("Update weights future done")

                return result
            except AttributeError as e:
                logger.error(f"Method '{method_name}' not found on engine: {e}")
                raise ValueError(f"Engine does not have method '{method_name}'")
            except Exception as e:
                logger.error(
                    f"Engine method '{method_name}' failed: {e}\n{traceback.format_exc()}"
                )
                raise

        # Submit to engine thread
        try:
            result = _submit_to_engine_thread(execute_in_engine_thread)
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

        # Convert all tensors to RTensors and store the tensor locally
        layout = RTensor.extract_layout(
            result,
            layouts=dict(args=raw_args, kwargs=raw_kwargs),
            node_addr=f"{_server_host}:{_server_port}",
        )
        if layout is not None:
            result = RTensor.remotize(
                result,
                layout,
                node_addr=f"{_server_host}:{_server_port}",
            )
        serialized_result = serialize_value(result)
        return jsonify({"status": "success", "result": serialized_result})

    except Exception as e:
        logger.error(f"Unexpected error in call: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


# ==================== Batch Data Storage Endpoints ====================
@app.route("/data/<shard_id>", methods=["PUT"])
def store_batch_data(shard_id: str):
    """Store batch data shard."""

    try:
        data_bytes = request.get_data()

        # Deserialize to get tensor (already on CPU)
        serialized_data = orjson.loads(data_bytes)
        data = deserialize_value(serialized_data)

        rtensor.store(shard_id, data)

        logger.debug(f"Stored batch shard {shard_id} (size={len(data_bytes)} bytes)")
        return jsonify({"status": "ok", "shard_id": shard_id})

    except Exception as e:
        logger.error(f"Error storing batch shard {shard_id}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/data/<shard_id>", methods=["GET"])
def retrieve_batch_data(shard_id: str):
    """Retrieve batch data shard."""

    logger.debug(f"Received data get request for shard {shard_id}")
    try:
        try:
            data = rtensor.fetch(shard_id)
        except KeyError:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": f"Shard {shard_id} not found",
                    }
                ),
                404,
            )

        serialized_data = serialize_value(data)
        data_bytes = orjson.dumps(serialized_data)

        logger.debug(f"Retrieved batch shard {shard_id} (size={len(data_bytes)} bytes)")
        return Response(data_bytes, mimetype="application/octet-stream")

    except Exception as e:
        logger.error(f"Error retrieving batch shard {shard_id}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/data/clear", methods=["DELETE"])
def clear_batch_data():
    """Clear specified batch data shards.

    Expected JSON payload:
    {
        "shard_ids": ["id1", "id2", ...]
    }
    """
    try:
        data = request.get_json(silent=True) or {}
        shard_ids = data.get("shard_ids", [])
        if not isinstance(shard_ids, list):
            return (
                jsonify({"status": "error", "message": "'shard_ids' must be a list"}),
                400,
            )

        cleared_count = sum(rtensor.remove(sid) for sid in shard_ids)
        stats = dict(cleared_count=cleared_count, **rtensor.storage_stats())
        logger.info(f"Cleared {cleared_count} batch shards. Stats: {stats}")
        stats.update({"status": "ok"})
        return jsonify(stats)

    except Exception as e:
        logger.error(f"Error clearing batch data: {e}")
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


def cleanup_engine_thread():
    """Clean up engine thread on shutdown."""
    global _engine_thread, _engine_work_queue

    with _engine_thread_lock:
        if _engine_work_queue is not None:
            # Send shutdown signal
            _engine_work_queue.put(None)
            _engine_work_queue = None

        if _engine_thread is not None:
            _engine_thread.join(timeout=5.0)
            if _engine_thread.is_alive():
                logger.warning("Engine thread did not shut down gracefully")
            _engine_thread = None
            logger.info("Engine thread cleaned up")


def main():
    """Main entry point for the sync RPC server."""
    parser = argparse.ArgumentParser(
        description="AReaL Sync RPC Server for TrainEngine/InferenceEngine"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Port to serve on (default: 0 = auto-assign)",
    )
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
    # name_resolve config
    parser.add_argument("--experiment-name", type=str, required=True)
    parser.add_argument("--trial-name", type=str, required=True)
    parser.add_argument("--role", type=str, required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--name-resolve-type", type=str, default="nfs")
    parser.add_argument(
        "--nfs-record-root", type=str, default="/tmp/areal/name_resolve"
    )
    parser.add_argument("--etcd3-addr", type=str, default="localhost:2379")

    args, _ = parser.parse_known_args()

    # Configure Werkzeug logging
    werkzeug_logger = stdlib_logging.getLogger("werkzeug")
    werkzeug_logger.setLevel(getattr(stdlib_logging, args.werkzeug_log_level))

    # Set global server address variables
    global _server_host, _server_port, _role
    _server_host = args.host
    if _server_host == "0.0.0.0":
        _server_host = gethostip()
    _role = args.role

    # Get worker identity
    worker_id = f"{args.role}/{args.worker_index}"

    # Make a flask server
    server = make_server(args.host, args.port, app, threaded=True)
    _server_port = server.socket.getsockname()[1]

    name_resolve.reconfigure(
        NameResolveConfig(
            type=args.name_resolve_type,
            nfs_record_root=args.nfs_record_root,
            etcd3_addr=args.etcd3_addr,
        )
    )
    key = names.worker_discovery(
        args.experiment_name, args.trial_name, args.role, args.worker_index
    )
    name_resolve.add(key, f"{_server_host}:{_server_port}", replace=True)

    logger.info(
        f"Starting sync RPC server on {_server_host}:{_server_port} for worker {worker_id}"
    )
    logger.info(f"Werkzeug log level: {args.werkzeug_log_level}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down sync RPC server")
    finally:
        perf_tracer.save(force=True)
        cleanup_engine_thread()
        cleanup_engine()
        server.shutdown()


if __name__ == "__main__":
    main()
