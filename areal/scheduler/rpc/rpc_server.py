import argparse
import traceback
from concurrent.futures import Future

from flask import Flask, jsonify, request

from areal.api.cli_args import BaseExperimentConfig
from areal.api.engine_api import InferenceEngine, TrainEngine
from areal.platforms import current_platform
from areal.scheduler.rpc.serialization import deserialize_value, serialize_value
from areal.utils import logging, name_resolve, seeding, stats_tracker
from areal.utils.data import (
    broadcast_tensor_container,
    tensor_container_to,
)
from areal.utils.dynamic_import import import_from_string

logger = logging.getLogger("SyncRPCServer")

# Global engine instance - must be TrainEngine or InferenceEngine
_engine: TrainEngine | InferenceEngine | None = None

# Create Flask app
app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint to verify server is alive."""
    global _engine
    return jsonify({"status": "healthy", "engine_initialized": _engine is not None})


@app.route("/configure", methods=["POST"])
def configure():
    """Configure worker with experiment config."""
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

        name_resolve.reconfigure(config.cluster.name_resolve)
        seeding.set_random_seed(config.seed, key=f"{role}{rank}")

        return jsonify(
            {
                "status": "success",
                "message": "Worker configured successful.",
                "result": None,
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error in configure: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route("/create_engine", methods=["POST"])
def create_engine():
    """
    Create and initialize a TrainEngine or InferenceEngine instance on this worker.

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
        if data is None:
            return jsonify({"error": "Invalid JSON in request body"}), 400

        engine_path = data.get("engine")
        # Deserialize init_args and init_kwargs (may contain tensors or dataclasses)
        init_args = deserialize_value(data.get("init_args", []))
        init_kwargs = deserialize_value(data.get("init_kwargs", {}))

        if not engine_path:
            return jsonify({"error": "Missing 'engine' field in request"}), 400

        # Dynamic import
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

        # Deserialize args and kwargs (convert SerializedTensor dicts to tensors)
        args = deserialize_value(args)
        kwargs = deserialize_value(kwargs)

        try:
            should_bcast = kwargs.pop("_should_bcast", True)
            if should_bcast and isinstance(_engine, TrainEngine):
                logger.info(f"Broadcasting data for TrainEngine method: {method_name}")

                args = tensor_container_to(args, current_platform.current_device())
                args = broadcast_tensor_container(
                    args,
                    src_rank=_engine.current_data_parallel_head(),
                    group=_engine.context_and_model_parallel_group,
                )
                kwargs = tensor_container_to(kwargs, current_platform.current_device())
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

            # HACK: handle update weights future
            if isinstance(result, Future):
                logger.info("Waiting for update weights future")
                result = result.result()
                logger.info("Update weights future done")

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
            return jsonify({"error": "Engine not initialized"}), 503

        # TrainEngine: reduce stats across data_parallel_group
        if not isinstance(_engine, TrainEngine):
            return (
                jsonify({"error": "/export_stats is only available for TrainEngine"}),
                400,
            )
        result = stats_tracker.export(reduce_group=_engine.data_parallel_group)
        return jsonify({"status": "success", "result": result})

    except Exception as e:
        logger.error(f"Unexpected error in export_stats: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


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

    logger.info(f"Starting sync RPC server on {args.host}:{args.port}")
    logger.info(f"Werkzeug log level: {args.werkzeug_log_level}")

    # Run Flask app with single-threaded synchronous mode
    # threaded=False ensures NCCL compatibility
    try:
        app.run(
            host=args.host,
            port=args.port,
            threaded=False,  # Single-threaded synchronous execution
            processes=1,  # Single process
            debug=False,
            use_reloader=False,
        )
    except KeyboardInterrupt:
        logger.info("Shutting down sync RPC server")
    finally:
        cleanup_engine()


if __name__ == "__main__":
    main()
