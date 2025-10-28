"""Modern FastAPI-based RPC server for engine workers.

This server runs on worker nodes to expose engine methods via HTTP/JSON RPC.
It uses safe JSON serialization instead of cloudpickle.
"""

import argparse
import importlib
import traceback
from contextlib import asynccontextmanager

import orjson
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import ORJSONResponse

from areal.api.engine_api import InferenceEngine, TrainEngine
from areal.utils import logging

logger = logging.getLogger("RPCServer")

# Global engine instance - must be TrainEngine or InferenceEngine
_engine: TrainEngine | InferenceEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    logger.info("RPC server starting up...")
    yield
    # Shutdown
    global _engine
    logger.info("Shutting down RPC server...")
    if _engine is not None:
        try:
            # Call destroy method if available
            if hasattr(_engine, "destroy"):
                _engine.destroy()
                logger.info("Engine destroyed successfully")
        except Exception as e:
            logger.error(f"Error destroying engine: {e}")
    _engine = None


app = FastAPI(
    title="AReaL Worker RPC Server",
    description="FastAPI-based RPC server for remote engine operations",
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Health check endpoint to verify server is alive."""
    return {"status": "healthy", "engine_initialized": _engine is not None}


@app.post("/create_engine")
async def create_engine(request: Request):
    """
    Create and initialize an engine instance on this worker.

    Expected JSON payload:
    {
        "engine": "areal.engine.ppo.actor.FSDPPPOActor",  # Import path
        "init_args": [...],  # Positional arguments
        "init_kwargs": {...}  # Keyword arguments
    }
    """
    global _engine

    try:
        body = await request.body()
        data = orjson.loads(body)

        engine_path = data.get("engine")
        init_args = data.get("init_args", [])
        init_kwargs = data.get("init_kwargs", {})

        if not engine_path:
            raise HTTPException(
                status_code=400, detail="Missing 'engine' field in request"
            )

        # Dynamic import
        try:
            module_path, class_name = engine_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            engine_class = getattr(module, class_name)

            # Validate that the class is a TrainEngine or InferenceEngine
            if not (
                issubclass(engine_class, TrainEngine)
                or issubclass(engine_class, InferenceEngine)
            ):
                raise TypeError(
                    f"Engine class must be a subclass of TrainEngine or InferenceEngine, "
                    f"got {engine_class}"
                )
        except (ValueError, ImportError, AttributeError) as e:
            logger.error(f"Failed to import engine '{engine_path}': {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to import engine '{engine_path}': {str(e)}",
            )
        except TypeError as e:
            logger.error(f"Invalid engine type: {e}")
            raise HTTPException(
                status_code=400,
                detail=str(e),
            )

        # Instantiate engine
        try:
            _engine = engine_class(*init_args, **init_kwargs)
            logger.info(f"Engine '{engine_path}' instantiated successfully")
        except Exception as e:
            logger.error(f"Failed to instantiate engine: {e}\n{traceback.format_exc()}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to instantiate engine: {str(e)}",
            )

        # Initialize engine if it has initialize method
        try:
            result = _engine.initialize(*init_args, **init_kwargs)
            logger.info(f"Engine initialized with result: {result}")
            return {
                "status": "success",
                "message": f"Engine '{engine_path}' created and initialized",
                "result": result,
            }
        except Exception as e:
            logger.error(f"Failed to initialize engine: {e}\n{traceback.format_exc()}")
            raise HTTPException(
                status_code=500, detail=f"Failed to initialize engine: {str(e)}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Unexpected error in create_engine: {e}\n{traceback.format_exc()}"
        )
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/call")
async def call_engine_method(request: Request):
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
        raise HTTPException(
            status_code=503,
            detail="Engine not initialized. Call /create_engine first.",
        )

    try:
        body = await request.body()
        data = orjson.loads(body)

        method_name = data.get("method")
        args = data.get("args", [])
        kwargs = data.get("kwargs", {})

        if not method_name:
            raise HTTPException(
                status_code=400, detail="Missing 'method' field in request"
            )

        # Call method directly (no need for hasattr/getattr with typed engine)
        logger.info(f"Calling engine method: {method_name}")
        try:
            # Get the method - will raise AttributeError if it doesn't exist
            method = getattr(_engine, method_name)
            result = method(*args, **kwargs)

            # Serialize result
            # Note: This assumes the result is JSON-serializable
            # For complex types (tensors, etc.), you may need custom serialization
            return {"status": "success", "result": result}

        except AttributeError as e:
            logger.error(f"Method '{method_name}' not found on engine: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Engine does not have method '{method_name}'",
            )
        except Exception as e:
            logger.error(
                f"Engine method '{method_name}' failed: {e}\n{traceback.format_exc()}"
            )
            raise HTTPException(
                status_code=500,
                detail=f"Engine method '{method_name}' failed: {str(e)}",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in call: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


def main():
    """Main entry point for the RPC server."""
    parser = argparse.ArgumentParser(description="AReaL Worker RPC Server")
    parser.add_argument("--port", type=int, required=True, help="Port to serve on")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )

    args, unknown = parser.parse_known_args()
    port = args.port

    logger.info(f"Starting RPC server on {args.host}:{port}")

    # Run uvicorn server with a single worker (required for GPU workloads)
    uvicorn.run(
        app,
        host=args.host,
        port=port,
        workers=1,  # Single worker required for GPU memory management
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    main()
