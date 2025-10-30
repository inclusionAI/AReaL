import argparse
import importlib
import json
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from areal.api.cli_args import BaseExperimentConfig
from areal.api.engine_api import TrainEngine
from areal.platforms import current_platform
from areal.scheduler.rpc.serialization import deserialize_value, serialize_value
from areal.utils import logging, name_resolve, seeding, stats_tracker

logger = logging.getLogger("SyncRPCServer")

# Global engine instance - must be TrainEngine
_engine: TrainEngine | None = None


class SyncRPCHandler(BaseHTTPRequestHandler):
    """HTTP request handler for sync RPC server endpoints."""

    def log_message(self, format: str, *args: Any) -> None:
        """Override to use our logger instead of stderr."""
        logger.debug(f"{self.address_string()} - {format % args}")

    def _send_json_response(self, data: dict, status_code: int = 200) -> None:
        """Send JSON response with appropriate headers."""
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def _read_json_body(self) -> dict | None:
        """Read and parse JSON request body."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length == 0:
                return {}
            body = self.rfile.read(content_length)
            return json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse JSON body: {e}")
            self._send_json_response(
                {"error": f"Invalid JSON in request body: {str(e)}"}, 400
            )
            return None

    def do_GET(self) -> None:
        """Handle GET requests."""
        if self.path == "/health":
            self._handle_health_check()
        else:
            self._send_json_response({"error": f"Not found: {self.path}"}, 404)

    def do_POST(self) -> None:
        """Handle POST requests."""
        if self.path == "/create_engine":
            self._handle_create_engine()
        elif self.path == "/call":
            self._handle_call_engine_method()
        elif self.path == "/export_stats":
            self._handle_export_stats()
        elif self.path == "/configure":
            self._handle_configure()
        else:
            self._send_json_response({"error": f"Not found: {self.path}"}, 404)

    def _handle_health_check(self) -> None:
        """Health check endpoint to verify server is alive."""
        global _engine
        self._send_json_response(
            {"status": "healthy", "engine_initialized": _engine is not None}
        )

    def _handle_configure(self) -> None:
        try:
            data = self._read_json_body()
            if data is None:
                return

            config = data.get("config")
            if not config:
                raise self._send_json_response(
                    {"error": "Missing 'config' field in request"}, 400
                )
            role = data.get("role")
            if not role:
                raise self._send_json_response(
                    {"error": "Missing 'role' field in request"}, 400
                )
            rank = data.get("rank")
            if not rank:
                raise self._send_json_response(
                    {"error": "Missing 'rank' field in request"}, 400
                )

            config = deserialize_value(config)
            config: BaseExperimentConfig

            name_resolve.reconfigure(config.cluster.name_resolve)

            seeding.set_random_seed(config.seed, key=f"{role}{rank}")

        except Exception as e:
            logger.error(
                f"Unexpected error in configure: {e}\n{traceback.format_exc()}"
            )
            self._send_json_response({"error": f"Internal server error: {str(e)}"}, 500)

    def _handle_create_engine(self) -> None:
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
            data = self._read_json_body()
            if data is None:
                return

            engine_path = data.get("engine")
            # Deserialize init_args and init_kwargs (may contain tensors or dataclasses)
            init_args = deserialize_value(data.get("init_args", []))
            init_kwargs = deserialize_value(data.get("init_kwargs", {}))

            if not engine_path:
                self._send_json_response(
                    {"error": "Missing 'engine' field in request"}, 400
                )
                return

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
                self._send_json_response(
                    {"error": f"Failed to import engine '{engine_path}': {str(e)}"},
                    400,
                )
                return
            except TypeError as e:
                logger.error(f"Invalid engine type: {e}")
                self._send_json_response({"error": str(e)}, 400)
                return

            # Instantiate engine
            try:
                _engine = engine_class(*init_args, **init_kwargs)
                logger.info(f"Engine '{engine_path}' instantiated successfully")
                self._send_json_response(
                    {
                        "status": "success",
                        "message": f"Engine '{engine_path}' created and initialized",
                        "result": None,
                    }
                )
            except Exception as e:
                logger.error(
                    f"Failed to instantiate engine: {e}\n{traceback.format_exc()}"
                )
                self._send_json_response(
                    {"error": f"Failed to instantiate engine: {str(e)}"}, 500
                )

        except Exception as e:
            logger.error(
                f"Unexpected error in create_engine: {e}\n{traceback.format_exc()}"
            )
            self._send_json_response({"error": f"Internal server error: {str(e)}"}, 500)

    def _handle_call_engine_method(self) -> None:
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
            self._send_json_response(
                {"error": "Engine not initialized. Call /create_engine first."}, 503
            )
            return

        try:
            data = self._read_json_body()
            if data is None:
                return

            method_name = data.get("method")
            args = data.get("args", [])
            kwargs = data.get("kwargs", {})

            if not method_name:
                self._send_json_response(
                    {"error": "Missing 'method' field in request"}, 400
                )
                return

            # Deserialize args and kwargs (convert SerializedTensor dicts to tensors)
            args = deserialize_value(args)
            kwargs = deserialize_value(kwargs)

            try:
                should_bcast = kwargs.pop("_should_bcast", True)
                if should_bcast:
                    logger.info(
                        f"Broadcasting data for TrainEngine method: {method_name}"
                    )
                    from areal.utils.data import (
                        broadcast_tensor_container,
                        tensor_container_to,
                    )

                    # TODO: to device here
                    args = tensor_container_to(args, current_platform.current_device())
                    args = broadcast_tensor_container(
                        args,
                        src_rank=_engine.current_data_parallel_head(),
                        group=_engine.context_and_model_parallel_group,
                    )
                    kwargs = tensor_container_to(
                        kwargs, current_platform.current_device()
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
                self._send_json_response(
                    {"error": f"Data broadcast '{method_name}' failed: {str(e)}"}, 500
                )
                return

            # Call method directly
            logger.info(f"Calling engine method: {method_name}")
            try:
                # Get the method - will raise AttributeError if it doesn't exist
                method = getattr(_engine, method_name)
                result = method(*args, **kwargs)

                # Serialize result (convert tensors to SerializedTensor dicts)
                serialized_result = serialize_value(result)
                self._send_json_response(
                    {"status": "success", "result": serialized_result}
                )

            except AttributeError as e:
                logger.error(f"Method '{method_name}' not found on engine: {e}")
                self._send_json_response(
                    {"error": f"Engine does not have method '{method_name}'"}, 400
                )
            except Exception as e:
                logger.error(
                    f"Engine method '{method_name}' failed: {e}\n{traceback.format_exc()}"
                )
                self._send_json_response(
                    {"error": f"Engine method '{method_name}' failed: {str(e)}"}, 500
                )

        except Exception as e:
            logger.error(f"Unexpected error in call: {e}\n{traceback.format_exc()}")
            self._send_json_response({"error": f"Internal server error: {str(e)}"}, 500)

    def _handle_export_stats(self) -> None:
        """Export training statistics from stats_tracker."""
        try:
            global _engine
            if _engine is None:
                self._send_json_response({"error": "Engine not initialized"}, 503)
                return

            # TrainEngine: reduce stats across data_parallel_group
            result = stats_tracker.export(reduce_group=_engine.data_parallel_group)
            self._send_json_response({"status": "success", "result": result})

        except Exception as e:
            logger.error(
                f"Unexpected error in export_stats: {e}\n{traceback.format_exc()}"
            )
            self._send_json_response({"error": f"Internal server error: {str(e)}"}, 500)


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

    # Create and run single-threaded HTTP server
    # HTTPServer is single-threaded by default (processes one request at a time)
    # This ensures NCCL compatibility
    server = HTTPServer((args.host, args.port), SyncRPCHandler)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down sync RPC server")
        server.shutdown()


if __name__ == "__main__":
    main()
