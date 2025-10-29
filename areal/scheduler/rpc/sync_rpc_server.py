"""Single-threaded Flask-based RPC server for distributed TrainEngine workers.

This server runs on worker nodes to expose TrainEngine methods via HTTP/JSON RPC.
It uses a single-threaded WSGI server to avoid threading conflicts with PyTorch
distributed communication (NCCL).

Key differences from async_rpc_server:
- Single-threaded: Uses Flask with threaded=False for NCCL compatibility
- TrainEngine only: Only accepts TrainEngine subclasses
- No /run_workflow: Workflow execution is handled by async_rpc_server
"""

import argparse
import importlib
import traceback

from flask import Flask, jsonify, request

from areal.api.engine_api import TrainEngine
from areal.scheduler.rpc.serialization import deserialize_value, serialize_value
from areal.utils import logging, stats_tracker

logger = logging.getLogger("SyncRPCServer")

# Global engine instance - must be TrainEngine
_engine: TrainEngine | None = None


app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint to verify server is alive."""
    return jsonify({"status": "healthy", "engine_initialized": _engine is not None})


@app.route("/create_engine", methods=["POST"])
def create_engine():
    """
    Create and initialize a TrainEngine instance on this worker.

    Expected JSON payload:
    {
        "engine": "areal.engine.ppo.actor.FSDPPPOActor",  # Import path
        "init_args": [...],  # Positional arguments
        "init_kwargs": {...}  # Keyword arguments
    }
    """
    global _engine

    try:
        data = request.get_json()
        engine_path = data.get("engine")
        # Deserialize init_args and init_kwargs (may contain tensors or dataclasses)
        init_args = deserialize_value(data.get("init_args", []))
        init_kwargs = deserialize_value(data.get("init_kwargs", {}))

        if not engine_path:
            return jsonify({"error": "Missing 'engine' field in request"}), 400

        # Dynamic import
        try:
            module_path, class_name = engine_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            engine_class = getattr(module, class_name)

            # Validate that the class is a TrainEngine
            if not issubclass(engine_class, TrainEngine):
                raise TypeError(
                    f"Engine class must be a subclass of TrainEngine, "
                    f"got {engine_class}. Use async_rpc_server for InferenceEngine."
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

        # Instantiate engine
        try:
            _engine = engine_class(*init_args, **init_kwargs)
            logger.info(f"Engine '{engine_path}' instantiated successfully")
            return jsonify(
                {
                    "status": "success",
                    "message": f"Engine '{engine_path}' created and initialized",
                    "result": None,
                }
            )
        except Exception as e:
            logger.error(f"Failed to instantiate engine: {e}\n{traceback.format_exc()}")
            return (
                jsonify({"error": f"Failed to instantiate engine: {str(e)}"}),
                500,
            )

    except Exception as e:
        logger.error(
            f"Unexpected error in create_engine: {e}\n{traceback.format_exc()}"
        )
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route("/call", methods=["POST"])
def call_engine_method():
    """
    Call a method on the TrainEngine instance.

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
        method_name = data.get("method")
        args = data.get("args", [])
        kwargs = data.get("kwargs", {})

        if not method_name:
            return jsonify({"error": "Missing 'method' field in request"}), 400

        # Deserialize args and kwargs (convert SerializedTensor dicts to tensors)
        args = deserialize_value(args)
        kwargs = deserialize_value(kwargs)

        try:
            should_bcast = kwargs.pop("_should_bcast", True)
            if should_bcast:
                logger.info(f"Broadcasting data for TrainEngine method: {method_name}")
                from areal.utils.data import broadcast_tensor_container

                args = broadcast_tensor_container(
                    args,
                    src_rank=_engine.current_data_parallel_head(),
                    group=_engine.context_and_model_parallel_group,
                )
                kwargs = broadcast_tensor_container(
                    kwargs,
                    src_rank=_engine.current_data_parallel_head(),
                    group=_engine.context_and_model_parallel_group,
                )
                logger.info("Broadcasting data done.")
        except Exception as e:
            logger.error(
                f"Broadcasting data for method '{method_name}' failed: {e}\n{traceback.format_exc()}"
            )
            return (
                jsonify({"error": f"Data broadcast '{method_name}' failed: {str(e)}"}),
                500,
            )

        # Call method directly
        logger.info(f"Calling engine method: {method_name}")
        try:
            # Get the method - will raise AttributeError if it doesn't exist
            method = getattr(_engine, method_name)
            result = method(*args, **kwargs)

            # Serialize result (convert tensors to SerializedTensor dicts)
            serialized_result = serialize_value(result)
            return jsonify({"status": "success", "result": serialized_result})

        except AttributeError as e:
            logger.error(f"Method '{method_name}' not found on engine: {e}")
            return (
                jsonify({"error": f"Engine does not have method '{method_name}'"}),
                400,
            )
        except Exception as e:
            logger.error(
                f"Engine method '{method_name}' failed: {e}\n{traceback.format_exc()}"
            )
            return (
                jsonify({"error": f"Engine method '{method_name}' failed: {str(e)}"}),
                500,
            )

    except Exception as e:
        logger.error(f"Unexpected error in call: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route("/export_stats", methods=["POST"])
def export_stats():
    """Export training statistics from stats_tracker."""
    try:
        global _engine
        if _engine is None:
            return (
                jsonify({"error": "Engine not initialized"}),
                503,
            )

        # TrainEngine: reduce stats across data_parallel_group
        result = stats_tracker.export(reduce_group=_engine.data_parallel_group)
        return jsonify({"status": "success", "result": result})

    except Exception as e:
        logger.error(f"Unexpected error in export_stats: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


def main():
    """Main entry point for the sync RPC server."""
    parser = argparse.ArgumentParser(
        description="AReaL Sync RPC Server for TrainEngine"
    )
    parser.add_argument("--port", type=int, required=True, help="Port to serve on")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )

    args, _ = parser.parse_known_args()

    logger.info(f"Starting sync RPC server on {args.host}:{args.port}")

    # Run Flask with single-threaded WSGI server
    # threaded=False ensures no thread pool (required for NCCL compatibility)
    # processes=1 ensures single process (no forking)
    app.run(
        host=args.host,
        port=args.port,
        threaded=False,
        processes=1,
        debug=False,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
