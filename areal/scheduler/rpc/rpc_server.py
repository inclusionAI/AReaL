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
from areal.scheduler.rpc.serialization import deserialize_value, serialize_value
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
app._expected_trajectory_keys = None


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
        # Deserialize init_args and init_kwargs (may contain tensors or dataclasses)
        init_args = deserialize_value(data.get("init_args", []))
        init_kwargs = deserialize_value(data.get("init_kwargs", {}))

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
            return {
                "status": "success",
                "message": f"Engine '{engine_path}' created and initialized",
                "result": None,
            }
        except Exception as e:
            logger.error(f"Failed to instantiate engine: {e}\n{traceback.format_exc()}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to instantiate engine: {str(e)}",
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

        # Deserialize args and kwargs (convert SerializedTensor dicts to tensors)
        args = deserialize_value(args)
        kwargs = deserialize_value(kwargs)

        try:
            if isinstance(_engine, TrainEngine):
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
            raise HTTPException(
                status_code=500,
                detail=f"Data bcast '{method_name}' failed: {str(e)}",
            )

        # Call method directly (no need for hasattr/getattr with typed engine)
        logger.info(f"Calling engine method: {method_name}")
        try:
            # Get the method - will raise AttributeError if it doesn't exist
            method = getattr(_engine, method_name)
            result = method(*args, **kwargs)

            # Serialize result (convert tensors to SerializedTensor dicts)
            serialized_result = serialize_value(result)
            return {"status": "success", "result": serialized_result}

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


@app.post("/run_workflow")
async def run_workflow(request: Request):
    """
    Run a workflow's arun_episode method directly without using the engine.

    Expected JSON payload:
    {
        "workflow": "areal.api.workflow_api.RolloutWorkflow",  # Import path
        "workflow_kwargs": {...},  # Keyword arguments for workflow instantiation
        "data": {...}  # Data to pass to arun_episode
    }
    """
    try:
        body = await request.body()
        data = orjson.loads(body)

        workflow_path = data.get("workflow")
        workflow_kwargs = data.get("workflow_kwargs")
        episode_data = data.get("data")
        should_accept_path = data.get("should_accept_path", None)
        check_trajectory_format = data.get("check_trajectory_format")

        if not workflow_path:
            raise HTTPException(
                status_code=400, detail="Missing 'workflow' field in request"
            )

        if episode_data is None:
            raise HTTPException(
                status_code=400, detail="Missing 'data' field in request"
            )

        # Deserialize episode_data (may contain tensors)
        episode_data = deserialize_value(episode_data)

        # Dynamic import workflow
        try:
            module_path, class_name = workflow_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            workflow_class = getattr(module, class_name)
            logger.info(f"Imported workflow class: {workflow_path}")
        except (ValueError, ImportError, AttributeError) as e:
            logger.error(f"Failed to import workflow '{workflow_path}': {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to import workflow '{workflow_path}': {str(e)}",
            )
        # Instantiate workflow
        try:
            workflow = workflow_class(**workflow_kwargs)
            logger.info(f"Workflow '{workflow_path}' instantiated successfully")
        except Exception as e:
            logger.error(
                f"Failed to instantiate workflow: {e}\n{traceback.format_exc()}"
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to instantiate workflow: {str(e)}",
            )

        should_accept = None
        if should_accept_path is not None:
            # Dynamic import filtering function
            try:
                module_path, fn_name = should_accept_path.rsplit(".", 1)
                module = importlib.import_module(module_path)
                should_accept = getattr(module, fn_name)
                logger.info(f"Imported filtering function: {should_accept_path}")
            except (ValueError, ImportError, AttributeError) as e:
                logger.error(
                    f"Failed to import filtering function '{should_accept_path}': {e}"
                )
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to import filtering function '{should_accept_path}': {str(e)}",
                )

        # Run episode
        try:
            global _engine
            traj = await workflow.arun_episode(_engine, episode_data)

            global app
            if check_trajectory_format and traj is not None:
                from areal.core.workflow_executor import (
                    check_trajectory_format as check_fn,
                )

                check_fn(
                    traj,
                    expected_keys=app._expected_trajectory_keys,
                    logger=logger,
                )
                # Track expected keys for consistency checking
                if isinstance(traj, dict) and "input_ids" in traj:
                    if app._expected_trajectory_keys is None:
                        app._expected_trajectory_keys = set(traj.keys())
                        logger.info(
                            f"Trajectory format check: tracking keys "
                            f"{app._expected_trajectory_keys}"
                        )

            from areal.experimental.openai.types import InteractionWithTokenLogpReward
            from areal.utils.data import concat_padded_tensors

            # Convert InteractionWithTokenLogpReward to tensor dict if needed
            if isinstance(traj, dict) and all(
                isinstance(v, InteractionWithTokenLogpReward) for v in traj.values()
            ):
                traj = concat_padded_tensors(
                    [v.to_tensor_dict() for v in traj.values()]
                )

            assert traj is None or isinstance(traj, dict), traj

            # Apply should_accept filtering
            accept_this = traj is not None and (
                should_accept is None or should_accept(traj)
            )

            # Serialize trajectory result (convert tensors to SerializedTensor dicts)
            if accept_this:
                serialized_traj = serialize_value(traj)
                return {"status": "success", "result": serialized_traj}
            else:
                return {"status": "success", "result": None}
        except Exception as e:
            logger.error(f"Workflow arun_episode failed: {e}\n{traceback.format_exc()}")
            raise HTTPException(
                status_code=500,
                detail=f"Workflow arun_episode failed: {str(e)}",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in run_workflow: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


def main():
    """Main entry point for the RPC server."""
    parser = argparse.ArgumentParser(description="AReaL Worker RPC Server")
    parser.add_argument("--port", type=int, required=True, help="Port to serve on")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )

    args, _ = parser.parse_known_args()
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
