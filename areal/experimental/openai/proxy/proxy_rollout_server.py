from __future__ import annotations

import argparse
import inspect
import os
import threading
import time
from typing import TYPE_CHECKING, Any

import uvicorn
from anthropic.types.message import Message
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
    AnthropicAdapter,
)
from litellm.types.utils import ModelResponse as LitellmModelResponse
from pydantic import BaseModel

from openai.types.chat import ChatCompletion
from openai.types.chat.completion_create_params import CompletionCreateParams
from openai.types.responses import Response
from openai.types.responses.response_create_params import ResponseCreateParams

from areal.api.cli_args import NameResolveConfig, OpenAIProxyConfig
from areal.experimental.openai.client import ArealOpenAI
from areal.scheduler.rpc.serialization import deserialize_value, serialize_value
from areal.utils import name_resolve, names, seeding
from areal.utils.dynamic_import import import_from_string
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.logging import getLogger
from areal.utils.network import find_free_ports, gethostip

from .server import (
    ANTHROPIC_MESSAGES_PATHNAME,
    CHAT_COMPLETIONS_PATHNAME,
    RESPONSES_PATHNAME,
    RL_END_SESSION_PATHNAME,
    RL_SET_REWARD_PATHNAME,
    RL_START_SESSION_PATHNAME,
    ExportTrajectoriesRequest,
    ExportTrajectoriesResponse,
    SessionData,
    SetRewardRequest,
    StartSessionRequest,
    StartSessionResponse,
    serialize_interactions,
)

if TYPE_CHECKING:
    from areal.api.engine_api import InferenceEngine


logger = getLogger("ProxyRolloutServer")


# =============================================================================
# Module-Level Globals (like rpc_server.py)
# =============================================================================


# Engine and client (created via /create_engine and /call with method "initialize")
_engine: InferenceEngine | None = None
_openai_client: ArealOpenAI | None = None

# Session management
_session_cache: dict[str, SessionData] = {}
_lock = threading.Lock()
_capacity = 0
_last_cleanup_time: float = 0
_session_timeout_seconds: int = 3600  # Default timeout (overridden by config)

# Server address (set at startup)
_server_host: str = "0.0.0.0"
_server_port: int = 8000

# Server config (needed for name_resolve registration)
_experiment_name: str | None = None
_trial_name: str | None = None
_name_resolve_type: str = "nfs"
_nfs_record_root: str = "/tmp/areal/name_resolve"
_etcd3_addr: str = "localhost:2379"

# Adapter to convert Anthropic request to OpenAI format
_adapter = AnthropicAdapter()

# =============================================================================
# Request Validation
# =============================================================================


async def validate_json_request(raw_request: Request):
    """Validate that the request content-type is application/json."""
    content_type = raw_request.headers.get("content-type", "").lower()
    media_type = content_type.split(";", maxsplit=1)[0]
    if media_type != "application/json":
        raise RequestValidationError(
            errors=[
                {
                    "loc": ["header", "content-type"],
                    "msg": "Unsupported Media Type: Only 'application/json' is allowed",
                    "type": "value_error",
                }
            ]
        )


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok", "initialized": _engine is not None}


def _setup_openai_client():
    global _openai_client, _session_timeout_seconds
    config = _engine.config
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    openai_cfg = config.openai or OpenAIProxyConfig()
    _openai_client = ArealOpenAI(
        engine=_engine,
        tokenizer=tokenizer,
        tool_call_parser=openai_cfg.tool_call_parser,
        reasoning_parser=openai_cfg.reasoning_parser,
        engine_max_tokens=openai_cfg.engine_max_tokens,
        chat_template_type=openai_cfg.chat_template_type,
    )
    # Set session timeout from config
    _session_timeout_seconds = openai_cfg.session_timeout_seconds


@app.post("/configure")
async def configure(raw_request: Request):
    data = await raw_request.json()
    config = deserialize_value(data.get("config"))
    rank = data.get("rank", 0)
    seeding.set_random_seed(config.seed, key=f"proxy{rank}")
    return {"status": "success"}


@app.post("/set_env")
async def set_env(raw_request: Request):
    data = await raw_request.json()
    for key, value in data.get("env", {}).items():
        os.environ[key] = str(value)
    return {"status": "success"}


@app.post("/create_engine")
async def create_engine(raw_request: Request):
    global _engine
    if _engine is not None:
        raise HTTPException(status_code=400, detail="Engine already exists")

    data = await raw_request.json()
    engine_class = import_from_string(data.get("engine"))
    init_kwargs = deserialize_value(data.get("init_kwargs", {}))
    _engine = engine_class(**init_kwargs)
    return {"status": "success"}


@app.post("/call")
async def call_engine_method(raw_request: Request):
    global _engine, _openai_client
    if _engine is None:
        raise HTTPException(status_code=400, detail="Engine not initialized")

    data = await raw_request.json()
    method_name = data.get("method")
    args = deserialize_value(data.get("args", []))
    kwargs = deserialize_value(data.get("kwargs", {}))

    method = getattr(_engine, method_name)
    result = method(*args, **kwargs)

    if method_name == "initialize":
        _setup_openai_client()

    return {"status": "success", "result": serialize_value(result)}


# =============================================================================
# Capacity Management
# =============================================================================


@app.post("/grant_capacity")
def grant_capacity():
    """Grant capacity for a new session."""
    global _capacity
    with _lock:
        _capacity += 1
        return {"capacity": _capacity}


# =============================================================================
# Session Management
# =============================================================================


def _cleanup_stale_sessions():
    """Remove stale sessions from the cache.

    Called periodically (at most once per minute) during start_session to clean up
    sessions that were started but never finished.
    """
    global _last_cleanup_time

    # Only run cleanup at most once per minute
    current_time = time.time()
    if current_time - _last_cleanup_time < 60:
        return

    _last_cleanup_time = current_time

    stale_sessions = []
    for session_id, session_data in _session_cache.items():
        if session_data.is_stale(timeout_seconds=_session_timeout_seconds):
            stale_sessions.append(session_id)

    for session_id in stale_sessions:
        logger.warning(f"Removing stale session: {session_id}")
        _session_cache.pop(session_id, None)

    if stale_sessions:
        logger.info(f"Cleaned up {len(stale_sessions)} stale sessions")


@app.post(f"/{RL_START_SESSION_PATHNAME}")
def start_session(request: StartSessionRequest) -> StartSessionResponse:
    """Start a new RL session."""
    global _capacity
    task_id = request.task_id

    with _lock:
        # Periodically cleanup stale sessions
        _cleanup_stale_sessions()

        if _capacity <= 0:
            raise HTTPException(
                status_code=429,
                detail="No available capacity to start a new session",
            )

        # Generate unique session ID
        idx = 0
        while (session_id := f"{task_id}-{idx}") in _session_cache:
            idx += 1

        _capacity -= 1
        _session_cache[session_id] = SessionData(session_id=session_id)

    return StartSessionResponse(session_id=session_id)


@app.post("/{session_id}/" + RL_END_SESSION_PATHNAME)
def end_session(session_id: str):
    """End an RL session."""
    with _lock:
        if session_id not in _session_cache:
            raise HTTPException(status_code=400, detail="Session not found")
        session = _session_cache[session_id]

    session.finish()
    return {"message": "success"}


@app.post("/{session_id}/" + RL_SET_REWARD_PATHNAME)
def set_reward(request: SetRewardRequest, session_id: str):
    """Set reward for an interaction in a session."""
    interaction_id = request.interaction_id
    reward = request.reward

    with _lock:
        if session_id not in _session_cache:
            raise HTTPException(
                status_code=400, detail=f"Session {session_id} not found"
            )
        session_data = _session_cache[session_id]

    session_data.update_last_access()

    completions = session_data.completions
    if interaction_id is None:
        # Take the last interaction id
        if len(completions) == 0:
            logger.error(f"No interactions in session {session_id}")
            raise HTTPException(status_code=400, detail="No interactions in session")
        interaction_id = completions.last_interaction_id
    elif interaction_id not in completions:
        logger.error(f"Interaction {interaction_id} not found in session {session_id}")
        raise HTTPException(
            status_code=400, detail=f"Interaction {interaction_id} not found"
        )
    session_data.completions.set_reward(interaction_id, reward)
    return {"message": "success"}


# =============================================================================
# OpenAI-Compatible Endpoints
# =============================================================================


async def _call_client_create(
    create_fn,
    request: dict[str, Any] | BaseModel,
    session_id: str,
    extra_ignored_args: list[str] | None = None,
) -> ChatCompletion | Response:
    """Common logic for chat completions and responses."""
    if _openai_client is None:
        raise HTTPException(
            status_code=500,
            detail='Proxy server not initialized. Send requests to /create_engine then /call "initialize" first.',
        )

    with _lock:
        if session_id not in _session_cache:
            raise HTTPException(
                status_code=400, detail=f"Session {session_id} not found"
            )
        session_data = _session_cache[session_id]

    session_data.update_last_access()

    sig = inspect.signature(create_fn)
    areal_client_ignored_args = ["model"] + (extra_ignored_args or [])
    areal_client_disallowed_args = ["areal_cache"]
    areal_client_allowed_args = list(
        k
        for k in sig.parameters.keys()
        if k not in areal_client_ignored_args and k not in areal_client_disallowed_args
    )

    kwargs = request.model_dump() if isinstance(request, BaseModel) else dict(request)
    dropped_args = []
    for k, v in kwargs.items():
        if k not in areal_client_allowed_args:
            dropped_args.append((k, v))

    for k, _ in dropped_args:
        del kwargs[k]

    def _is_default_value(k: str, v: Any) -> bool:
        if isinstance(request, BaseModel):
            return v == type(request).model_fields[k].default
        return False

    dropped_non_default_args = [
        (k, v)
        for k, v in dropped_args
        if k not in areal_client_ignored_args and not _is_default_value(k, v)
    ]
    if len(dropped_non_default_args):
        dropped_args_str = "\n".join(
            [f"  {k}: {v}" for k, v in dropped_non_default_args]
        )
        logger.warning(
            f"dropped unsupported non-default arguments for areal client:\n"
            f"{dropped_args_str}"
        )

    if "temperature" not in kwargs:
        kwargs["temperature"] = 1.0
        logger.warning("temperature not set in request, defaulting to 1.0")
    if "top_p" not in kwargs:
        kwargs["top_p"] = 1.0
        logger.warning("top_p not set in request, defaulting to 1.0")

    try:
        return await create_fn(areal_cache=session_data.completions, **kwargs)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/{session_id}/" + CHAT_COMPLETIONS_PATHNAME,
    dependencies=[Depends(validate_json_request)],
)
async def chat_completions(
    request: CompletionCreateParams, session_id: str
) -> ChatCompletion:
    """OpenAI-compatible chat completions endpoint."""
    if _openai_client is None:
        raise HTTPException(
            status_code=500,
            detail='Proxy server not initialized. Send requests to /create_engine then /call "initialize" first.',
        )
    return await _call_client_create(
        create_fn=_openai_client.chat.completions.create,
        request=request,
        session_id=session_id,
    )


@app.post(
    "/{session_id}/" + RESPONSES_PATHNAME,
    dependencies=[Depends(validate_json_request)],
)
async def responses(request: ResponseCreateParams, session_id: str) -> Response:
    """OpenAI-compatible responses endpoint."""
    if _openai_client is None:
        raise HTTPException(
            status_code=500,
            detail='Proxy server not initialized. Send requests to /create_engine then /call "initialize" first.',
        )
    return await _call_client_create(
        create_fn=_openai_client.responses.create,
        request=request,
        session_id=session_id,
    )


@app.post(
    "/{session_id}/" + ANTHROPIC_MESSAGES_PATHNAME,
    dependencies=[Depends(validate_json_request)],
)
async def anthropic_messages(raw_request: Request, session_id: str) -> Message:
    """Anthropic Messages API compatible endpoint.

    Converts Anthropic format requests to OpenAI format, processes through
    the OpenAI-compatible endpoint, then converts the response back to
    Anthropic format.

    Uses LiteLLM's AnthropicAdapter for format conversion.
    """

    if _openai_client is None:
        raise HTTPException(
            status_code=500,
            detail='Proxy server not initialized. Send requests to /create_engine then /call "initialize" first.',
        )

    # Parse Anthropic request
    anthropic_request = await raw_request.json()

    try:
        openai_request = _adapter.translate_completion_input_params(
            anthropic_request.copy()
        )
        if openai_request is None:
            raise ValueError("Failed to translate request")
        openai_request = dict(openai_request)
    except (ValueError, TypeError, KeyError) as e:
        logger.warning(
            f"Failed to convert Anthropic request to OpenAI format due to invalid input: {e}"
        )
        raise HTTPException(
            status_code=400, detail=f"Invalid Anthropic request format: {e}"
        )
    except Exception as e:
        logger.error(
            f"Unexpected error converting Anthropic request to OpenAI format: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail="Internal server error during request conversion."
        )

    # Call OpenAI-compatible endpoint
    openai_response = await _call_client_create(
        create_fn=_openai_client.chat.completions.create,
        request=openai_request,
        session_id=session_id,
    )

    # Convert OpenAI response to Anthropic format using LiteLLM's adapter
    try:
        # Convert ChatCompletion to LitellmModelResponse
        openai_response_dict = openai_response.model_dump()
        model_response = LitellmModelResponse(**openai_response_dict)
        anthropic_response = _adapter.translate_completion_output_params(model_response)
        if anthropic_response is None:
            raise ValueError("Failed to translate response")

        # LiteLLM returns Pydantic BaseModel objects in content list,
        # Convert them to dict.
        if "content" in anthropic_response and anthropic_response["content"]:
            anthropic_response["content"] = [
                block.model_dump() if hasattr(block, "model_dump") else block
                for block in anthropic_response["content"]
            ]
        return Message(**anthropic_response)
    except Exception as e:
        logger.error(f"Failed to convert OpenAI response to Anthropic format: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to convert response: {e}")


# =============================================================================
# Trajectory Export
# =============================================================================


@app.post("/export_trajectories")
async def export_trajectories(
    request: ExportTrajectoriesRequest,
) -> ExportTrajectoriesResponse:
    """Export interactions for a session."""
    session_id = request.session_id

    with _lock:
        if session_id not in _session_cache:
            raise HTTPException(
                status_code=404, detail=f"Session {session_id} not found"
            )
        session_data = _session_cache[session_id]

    # Wait for session to complete (non-blocking, outside lock)
    await session_data.wait_for_finish()

    # Export interactions
    interactions = session_data.export_interactions(
        discount=request.discount,
        style=request.style,
    )

    # Remove session from cache
    with _lock:
        _session_cache.pop(session_id, None)

    # Serialize for HTTP transport
    serialized = serialize_interactions(interactions)
    return ExportTrajectoriesResponse(interactions=serialized)


# =============================================================================
# Cleanup
# =============================================================================


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


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Main entry point for the proxy rollout server."""
    parser = argparse.ArgumentParser(description="Proxy Rollout Server")
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Port to serve on (default: 0 = auto-assign)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    # name_resolve config (same as rpc_server.py)
    parser.add_argument("--experiment-name", type=str, required=True)
    parser.add_argument("--trial-name", type=str, required=True)
    parser.add_argument("--role", type=str, required=True)
    parser.add_argument("--worker-index", type=int, default=-1)
    parser.add_argument("--name-resolve-type", type=str, default="nfs")
    parser.add_argument(
        "--nfs-record-root", type=str, default="/tmp/areal/name_resolve"
    )
    parser.add_argument("--etcd3-addr", type=str, default="localhost:2379")
    parser.add_argument(
        "--fileroot",
        type=str,
        default=None,
        help="Root directory for log files (unused, for compatibility with rpc_server)",
    )

    args, _ = parser.parse_known_args()

    # Set global server address variables
    global _server_host, _server_port
    global \
        _experiment_name, \
        _trial_name, \
        _name_resolve_type, \
        _nfs_record_root, \
        _etcd3_addr
    _server_host = args.host
    if _server_host == "0.0.0.0":
        _server_host = gethostip()

    # Set global config for name_resolve
    _experiment_name = args.experiment_name
    _trial_name = args.trial_name
    _name_resolve_type = args.name_resolve_type
    _nfs_record_root = args.nfs_record_root
    _etcd3_addr = args.etcd3_addr

    # Get worker identity
    worker_role = args.role
    worker_index = args.worker_index

    if "SLURM_PROCID" in os.environ:
        # Overwriting with slurm task id
        worker_index = int(os.environ["SLURM_PROCID"])
    if worker_index == -1:
        raise ValueError("Invalid worker index. Not found from SLURM environ or args.")
    worker_id = f"{worker_role}/{worker_index}"

    # Determine port
    _server_port = args.port if args.port != 0 else find_free_ports(1)[0]

    # Configure name_resolve and register this server
    name_resolve.reconfigure(
        NameResolveConfig(
            type=args.name_resolve_type,
            nfs_record_root=args.nfs_record_root,
            etcd3_addr=args.etcd3_addr,
        )
    )
    key = names.worker_discovery(
        args.experiment_name, args.trial_name, args.role, worker_index
    )
    name_resolve.add(key, f"{_server_host}:{_server_port}", replace=True)

    logger.info(
        f"Starting proxy rollout server on {_server_host}:{_server_port} for worker {worker_id}"
    )

    try:
        # Run uvicorn directly (blocking)
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=_server_port,
            log_level="warning",
            timeout_keep_alive=300,
        )
    except KeyboardInterrupt:
        logger.info("Shutting down proxy rollout server")
    finally:
        cleanup_engine()
        logger.info("Proxy rollout server stopped.")


if __name__ == "__main__":
    main()
