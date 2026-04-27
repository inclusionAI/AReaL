"""Engine Blueprint: engine lifecycle and method invocation.

Provides a Flask Blueprint that manages engine threads, engine creation,
and engine method calls.  Registers hooks on :class:`GuardState` for
``/configure``, ``/health``, and cleanup.

Routes:

- ``POST /set_env``       — set environment variables in the engine thread
- ``POST /create_engine`` — instantiate a TrainEngine or InferenceEngine
- ``POST /call``          — invoke a method on a named engine instance

The engine thread guarantees serial execution of all engine operations,
which is required for NCCL compatibility.
"""

from __future__ import annotations

import os
import traceback
from collections.abc import Callable
from concurrent.futures import Future
from queue import Queue
from threading import Lock, Thread
from typing import Any

from flask import Blueprint, jsonify, request

from areal.api import InferenceEngine, TrainEngine
from areal.infra.platforms import current_platform
from areal.infra.rpc.guard.app import GuardState, get_state
from areal.infra.rpc.rtensor import RTensor
from areal.infra.rpc.serialization import deserialize_value, serialize_value
from areal.utils import logging, perf_tracer, seeding
from areal.utils.data import broadcast_tensor_container, tensor_container_to
from areal.utils.dynamic_import import import_from_string

logger = logging.getLogger("EngineBP")

engine_bp = Blueprint("engine", __name__)

# ---------------------------------------------------------------------------
# Engine-specific module-level state
# ---------------------------------------------------------------------------

# Global engine instances — keyed by engine_name (e.g., "actor/0", "ref/0")
_engines: dict[str, TrainEngine | InferenceEngine] = {}

# Engine thread for executing all engine-related operations serially.
# This ensures NCCL compatibility by running engine operations in a single
# thread, while allowing /data/ endpoints to be processed concurrently.
_engine_thread: Thread | None = None
_engine_work_queue: Queue | None = None
_engine_thread_lock = Lock()


# ---------------------------------------------------------------------------
# Engine thread management
# ---------------------------------------------------------------------------


def _init_engine_thread() -> None:
    """Lazily initialize the engine worker thread."""
    global _engine_thread, _engine_work_queue

    with _engine_thread_lock:
        if _engine_thread is not None:
            if _engine_thread.is_alive():
                return  # Already initialized
            else:
                raise RuntimeError("Engine thread is dead.")

        _engine_work_queue = Queue()

        def engine_worker():
            logger.info("Engine thread started")
            while True:
                try:
                    work_item = _engine_work_queue.get()
                    if work_item is None:  # Shutdown signal
                        logger.info("Engine thread shutting down")
                        break

                    func, args, kwargs, future, func_name = work_item
                    try:
                        result = func(*args, **kwargs)
                        future.set_result(result)
                    except Exception as e:
                        future.set_exception(e)
                    finally:
                        _engine_work_queue.task_done()
                except Exception as e:
                    logger.error(
                        f"Error in engine thread when "
                        f"running {func_name}: {e}\n{traceback.format_exc()}"
                    )
                    if work_item and len(work_item) > 3:
                        work_item[3].set_exception(e)

        _engine_thread = Thread(target=engine_worker, daemon=True, name="EngineWorker")
        _engine_thread.start()
        logger.info("Engine thread initialized")


def _submit_to_engine_thread(
    func_name: str, func: Callable, *args: Any, **kwargs: Any
) -> Any:
    global _engine_work_queue

    _init_engine_thread()

    _is_init_op = func_name in (
        "configure",
        "create_engine",
        "call_create_process_group",
        "call_initialize",
        "call_connect_engine",
    )
    if _is_init_op:
        logger.info(f"[DiagInit] _submit_to_engine_thread: submitting '{func_name}'...")

    future: Future = Future()
    _engine_work_queue.put((func, args, kwargs, future, func_name))
    result = future.result()

    if _is_init_op:
        logger.info(f"[DiagInit] _submit_to_engine_thread: '{func_name}' completed")

    return result


# ---------------------------------------------------------------------------
# Hook registration
# ---------------------------------------------------------------------------


def register_engine_hooks(state: GuardState) -> None:
    """Register engine-specific hooks on the :class:`GuardState`.

    Must be called after creating the Flask app and before starting
    the server.  Registers:

    - health hook → adds ``engine_count`` and ``engines`` to /health
    - configure hook → sets random seeds in the engine thread
    - cleanup hooks → destroy engines and shut down engine thread
    """
    state.register_health_hook(_engine_health_hook)
    state.register_configure_hook(_engine_configure_hook)
    state.register_cleanup_hook(cleanup_engine_thread)
    state.register_cleanup_hook(cleanup_engines)


def _engine_health_hook() -> dict[str, Any]:
    """Contribute engine info to the /health response."""
    return {"engine_count": len(_engines), "engines": list(_engines.keys())}


def _engine_configure_hook(data: dict) -> dict:
    import time as _time

    _t0 = _time.time()
    config_data = data.get("config")
    if config_data is None:
        raise ValueError("Missing 'config' field in request")

    rank = data.get("rank")
    if rank is None:
        raise ValueError("Missing 'rank' field in request")

    logger.info(f"[DiagInit] _engine_configure_hook ENTERED (rank={rank})")

    logger.info(f"[DiagInit] _engine_configure_hook rank={rank}: deserializing config...")
    _t1 = _time.time()
    config = deserialize_value(config_data)
    logger.info(
        f"[DiagInit] _engine_configure_hook rank={rank}: config deserialized in "
        f"{_time.time() - _t1:.2f}s"
    )

    state = get_state()
    role = state.role

    def execute_configure():
        logger.info(f"[DiagInit] execute_configure rank={rank}: setting random seed...")
        seeding.set_random_seed(config.seed, key=f"{role}{rank}")
        logger.info(f"[DiagInit] execute_configure rank={rank}: seed set successfully")
        return {
            "status": "success",
            "message": "Worker configured successful.",
            "result": None,
        }

    logger.info(f"[DiagInit] _engine_configure_hook rank={rank}: submitting to engine thread...")
    result = _submit_to_engine_thread("configure", execute_configure)
    logger.info(
        f"[DiagInit] _engine_configure_hook rank={rank}: COMPLETED in "
        f"{_time.time() - _t0:.2f}s"
    )
    return result


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def cleanup_engines() -> None:
    """Destroy all engine instances."""
    global _engines
    if _engines:
        for engine_name, engine in list(_engines.items()):
            try:
                engine.destroy()
                logger.info(f"Engine '{engine_name}' destroyed successfully")
            except Exception as e:
                logger.error(f"Error destroying engine '{engine_name}': {e}")
        _engines.clear()


def cleanup_engine_thread() -> None:
    """Shut down the engine worker thread."""
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


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------


@engine_bp.route("/set_env", methods=["POST"])
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
                                "Environment variable name must be str, "
                                f"got {type(key)}"
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

        result = _submit_to_engine_thread("set_env", execute_set_env)
        return jsonify(result)

    except Exception as e:
        logger.error(f"Unexpected error in set_env: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@engine_bp.route("/create_engine", methods=["POST"])
def create_engine():
    global _engines

    import time as _time

    _t0 = _time.time()

    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Invalid JSON in request body"}), 400

        engine = data.get("engine")
        engine_name = data.get("engine_name")
        init_args = deserialize_value(data.get("init_args", []))
        init_kwargs = deserialize_value(data.get("init_kwargs", {}))

        if not engine:
            return (
                jsonify({"error": "Missing 'engine' field in request"}),
                400,
            )

        if not engine_name:
            return (
                jsonify({"error": "Missing 'engine_name' field in request"}),
                400,
            )

        if engine_name in _engines:
            return (
                jsonify(
                    {
                        "error": f"Engine '{engine_name}' already exists. "
                        "Use a different name or delete the existing "
                        "engine first."
                    }
                ),
                400,
            )

        logger.info(
            f"[DiagInit] /create_engine ENTERED: engine={engine}, "
            f"engine_name={engine_name}"
        )

        try:
            logger.info(f"[DiagInit] /create_engine {engine_name}: importing engine class...")
            _t1 = _time.time()
            engine_class = import_from_string(engine)

            if not issubclass(engine_class, TrainEngine) and not issubclass(
                engine_class, InferenceEngine
            ):
                raise TypeError(
                    "Engine class must be a subclass of TrainEngine or "
                    f"InferenceEngine, got {engine_class}.."
                )
            logger.info(
                f"[DiagInit] /create_engine {engine_name}: engine class imported in "
                f"{_time.time() - _t1:.2f}s"
            )
        except (ValueError, ImportError, AttributeError) as e:
            logger.error(f"Failed to import engine '{engine}': {e}")
            return (
                jsonify({"error": (f"Failed to import engine '{engine}': {str(e)}")}),
                400,
            )
        except TypeError as e:
            logger.error(f"Invalid engine type: {e}")
            return jsonify({"error": str(e)}), 400

        def create_engine_in_engine_thread():
            try:
                logger.info(
                    f"[DiagInit] /create_engine {engine_name}: instantiating "
                    f"{engine} in engine thread..."
                )
                _t2 = _time.time()
                engine_obj = engine_class(*init_args, **init_kwargs)
                logger.info(
                    f"[DiagInit] /create_engine {engine_name}: instantiated in "
                    f"{_time.time() - _t2:.2f}s"
                )
                return engine_obj
            except Exception as e:
                logger.error(
                    f"Failed to instantiate engine: {e}\n{traceback.format_exc()}"
                )
                raise

        try:
            logger.info(
                f"[DiagInit] /create_engine {engine_name}: submitting to engine thread..."
            )
            engine_obj = _submit_to_engine_thread(
                "create_engine", create_engine_in_engine_thread
            )
            _engines[engine_name] = engine_obj
            logger.info(
                f"[DiagInit] /create_engine {engine_name}: COMPLETED in "
                f"{_time.time() - _t0:.2f}s total"
            )
            return jsonify(
                {
                    "status": "success",
                    "message": (f"Engine '{engine_name}' created and initialized"),
                    "engine_name": engine_name,
                    "result": None,
                }
            )
        except Exception as e:
            return (
                jsonify({"error": f"Failed to instantiate engine: {str(e)}"}),
                500,
            )

    except Exception as e:
        logger.error(
            f"Unexpected error in create_engine: {e}\n{traceback.format_exc()}"
        )
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@engine_bp.route("/call", methods=["POST"])
def call_engine_method():
    global _engines

    import time as _time

    _t0 = _time.time()

    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Invalid JSON in request body"}), 400

        method_name = data.get("method")
        engine_name = data.get("engine_name")
        raw_args = data.get("args", [])
        raw_kwargs = data.get("kwargs", {})

        if not method_name:
            return (
                jsonify({"error": "Missing 'method' field in request"}),
                400,
            )

        if not engine_name:
            return (
                jsonify({"error": "Missing 'engine_name' field in request"}),
                400,
            )

        if engine_name not in _engines:
            return (
                jsonify(
                    {
                        "error": f"Engine '{engine_name}' not found. "
                        f"Available engines: {list(_engines.keys())}"
                    }
                ),
                404,
            )

        engine = _engines[engine_name]

        raw_args = deserialize_value(raw_args)
        raw_kwargs = deserialize_value(raw_kwargs)
        args = RTensor.localize(raw_args)
        kwargs = RTensor.localize(raw_kwargs)

        _is_init_method = method_name in (
            "create_process_group",
            "initialize",
            "connect_engine",
        )
        if _is_init_method:
            logger.info(
                f"[DiagInit] /call ENTERED: method={method_name}, "
                f"engine={engine_name}"
            )

        def execute_in_engine_thread():
            try:
                if isinstance(engine, TrainEngine) and engine.initialized:
                    logger.debug(
                        f"Broadcasting data for TrainEngine method: {method_name}"
                    )

                    nonlocal raw_args, raw_kwargs
                    raw_args = broadcast_tensor_container(
                        tensor_container_to(
                            raw_args, current_platform.current_device()
                        ),
                        src_rank=engine.current_data_parallel_head(),
                        group=engine.context_and_model_parallel_group,
                    )
                    raw_kwargs = broadcast_tensor_container(
                        tensor_container_to(
                            raw_kwargs, current_platform.current_device()
                        ),
                        src_rank=engine.current_data_parallel_head(),
                        group=engine.context_and_model_parallel_group,
                    )

                    args_bcast = broadcast_tensor_container(
                        tensor_container_to(
                            args, current_platform.current_device()
                        ),
                        src_rank=engine.current_data_parallel_head(),
                        group=engine.context_and_model_parallel_group,
                    )
                    kwargs_bcast = broadcast_tensor_container(
                        tensor_container_to(
                            kwargs, current_platform.current_device()
                        ),
                        src_rank=engine.current_data_parallel_head(),
                        group=engine.context_and_model_parallel_group,
                    )
                    logger.debug("Broadcasting data done.")
                else:
                    args_bcast = args
                    kwargs_bcast = kwargs

                if _is_init_method:
                    logger.info(
                        f"[DiagInit] /call {engine_name}.{method_name}: "
                        f"executing in engine thread..."
                    )
                    _et0 = _time.time()

                logger.debug(f"Calling engine '{engine_name}' method: {method_name}")

                category = "misc"
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

                with perf_tracer.trace_scope(
                    f"rpc.{method_name}",
                    category=category,
                    args={"method": method_name, "engine": engine_name},
                ):
                    method = getattr(engine, method_name)
                    result = method(*args_bcast, **kwargs_bcast)

                    if isinstance(result, Future):
                        logger.debug("Waiting for update weights future")
                        result = result.result()
                        logger.debug("Update weights future done")

                if _is_init_method:
                    logger.info(
                        f"[DiagInit] /call {engine_name}.{method_name}: "
                        f"COMPLETED in {_time.time() - _et0:.2f}s"
                    )

                return result
            except AttributeError as e:
                logger.error(f"Method '{method_name}' not found on engine: {e}")
                raise ValueError(f"Engine does not have method '{method_name}'")
            except Exception as e:
                logger.error(
                    f"Engine method '{method_name}' failed: "
                    f"{e}\n{traceback.format_exc()}"
                )
                raise

        try:
            result = _submit_to_engine_thread(
                f"call_{method_name}", execute_in_engine_thread
            )
        except Exception as e:
            error_msg = str(e)
            if "Engine does not have method" in error_msg:
                return (
                    jsonify({"error": error_msg}),
                    400,
                )
            return (
                jsonify(
                    {"error": (f"Engine method '{method_name}' failed: {error_msg}")}
                ),
                500,
            )

        if _is_init_method:
            logger.info(
                f"[DiagInit] /call {engine_name}.{method_name}: total "
                f"{_time.time() - _t0:.2f}s (including RPC overhead)"
            )

        state = get_state()
        result = RTensor.remotize(result, node_addr=state.node_addr)
        serialized_result = serialize_value(result)
        return jsonify({"status": "success", "result": serialized_result})

    except Exception as e:
        logger.error(f"Unexpected error in call: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500
