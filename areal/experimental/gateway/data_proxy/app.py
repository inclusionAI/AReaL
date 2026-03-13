from __future__ import annotations

import hmac
import inspect
import json
import logging
import types
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from openai.types.chat.completion_create_params import CompletionCreateParams
from pydantic import BaseModel

from areal.experimental.gateway.data_proxy.backend import (
    GenerationResult,
    SGLangBackend,
)
from areal.experimental.gateway.data_proxy.chat import ChatCompletionHandler
from areal.experimental.gateway.data_proxy.config import DataProxyConfig
from areal.experimental.gateway.data_proxy.pause import (
    PauseState,
    pause_backend,
    resume_backend,
)
from areal.experimental.gateway.data_proxy.session import (
    ExportTrajectoriesRequest,
    ExportTrajectoriesResponse,
    SessionStore,
    SetRewardRequest,
    StartSessionRequest,
    StartSessionResponse,
    serialize_interactions,
)
from areal.experimental.gateway.data_proxy.tokenizer_proxy import TokenizerProxy
from areal.experimental.gateway.data_proxy.weight_update import (
    init_weights_update_group,
    set_version,
    update_weights_from_disk,
    update_weights_from_distributed,
)

logger = logging.getLogger("DataProxy")

_warned_params: set[str] = set()


def _warn_once(msg: str) -> None:
    if msg not in _warned_params:
        _warned_params.add(msg)
        logger.warning(msg)


class GenerateRequest(BaseModel):
    """Request body for POST /generate."""

    text: Optional[str] = None
    input_ids: Optional[list[int]] = None
    sampling_params: Optional[dict[str, Any]] = None


async def stream_tokens(result: GenerationResult, tok: TokenizerProxy):
    """Async generator that yields one SSE chunk per output token."""
    for i, (token_id, logprob) in enumerate(
        zip(result.output_tokens, result.output_logprobs)
    ):
        is_last = i == len(result.output_tokens) - 1
        text_piece = tok.decode_token(token_id)
        chunk = {
            "token": token_id,
            "text": text_piece,
            "logprob": logprob,
            "finished": is_last,
        }
        if is_last:
            chunk["stop_reason"] = result.stop_reason
        yield f"data: {json.dumps(chunk)}\n\n".encode()


# =============================================================================
# API Key helpers (for RL control-plane endpoints only)
# =============================================================================


def _extract_bearer_token(request: Request) -> str:
    """Extract API token from Authorization header.

    Raises HTTPException(401) if missing or malformed.
    """
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    raise HTTPException(
        status_code=401,
        detail="Missing or malformed Authorization header. Expected 'Bearer <token>'.",
    )


def _require_admin_key(request: Request, store: SessionStore) -> str:
    """Validate that the request carries the admin API key."""
    token = _extract_bearer_token(request)
    if not hmac.compare_digest(token, store.admin_api_key):
        raise HTTPException(status_code=403, detail="Invalid admin API key.")
    return token


def _require_session_key(request: Request, store: SessionStore) -> str:
    """Resolve session_id from the session API key in the Authorization header."""
    token = _extract_bearer_token(request)
    session = store.get_session_by_api_key(token)
    if session is None:
        raise HTTPException(
            status_code=401, detail="Invalid or expired session API key."
        )
    return session.session_id


def _try_extract_bearer_token(request: Request) -> str | None:
    """Extract bearer token if present. Returns None if missing/malformed.

    Unlike _extract_bearer_token, this never raises — it's for endpoints
    that accept requests with or without auth.
    """
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    return None


async def _call_client_create(
    create_fn,
    request: dict[str, Any] | CompletionCreateParams,
    session_data,
    extra_ignored_args: list[str] | None = None,
    stream: bool = False,
):
    """Common logic for chat completions — mirrors proxy_rollout_server._call_client_create.

    Introspects create_fn signature, filters kwargs, passes areal_cache.
    """
    sig = inspect.signature(create_fn)
    areal_client_ignored_args = ["model"] + (extra_ignored_args or [])
    areal_client_disallowed_args = ["areal_cache"]

    # Check if the function accepts **kwargs (VAR_KEYWORD).
    # If so, any key not explicitly ignored/disallowed is allowed through.
    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if has_var_keyword:
        areal_client_allowed_args = None  # sentinel: allow everything
    else:
        areal_client_allowed_args = list(
            k
            for k in sig.parameters.keys()
            if k not in areal_client_ignored_args
            and k not in areal_client_disallowed_args
        )

    if isinstance(request, BaseModel):
        kwargs = request.model_dump()
    elif isinstance(request, dict):
        kwargs = dict(request)
    else:
        kwargs = dict(request)

    dropped_args = []
    for k, v in list(kwargs.items()):
        # If areal_client_allowed_args is None, only drop ignored/disallowed args
        if areal_client_allowed_args is not None:
            if k not in areal_client_allowed_args:
                dropped_args.append((k, v))
        elif k in areal_client_ignored_args or k in areal_client_disallowed_args:
            dropped_args.append((k, v))

    for k, _ in dropped_args:
        del kwargs[k]

    def _is_default_value(k: str, v: Any) -> bool:
        if isinstance(request, BaseModel):
            field_info = type(request).model_fields.get(k)
            if field_info is not None:
                return v == field_info.default
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
        _warn_once(
            f"dropped unsupported non-default arguments for data proxy handler:\n"
            f"{dropped_args_str}"
        )

    if "temperature" not in kwargs:
        kwargs["temperature"] = 1.0
        _warn_once("temperature not set in request, defaulting to 1.0")
    if "top_p" not in kwargs:
        kwargs["top_p"] = 1.0
        _warn_once("top_p not set in request, defaulting to 1.0")

    if stream:
        kwargs["stream"] = True

    try:
        return await create_fn(areal_cache=session_data.completions, **kwargs)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


def create_app(config: DataProxyConfig) -> FastAPI:
    """Factory that creates the FastAPI app with lifespan-managed resources."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info(
            "Data proxy starting — backend=%s, tokenizer=%s",
            config.backend_addr,
            config.tokenizer_path,
        )
        tok = TokenizerProxy(config.tokenizer_path)
        pause_state = PauseState()
        backend = SGLangBackend(
            backend_addr=config.backend_addr,
            pause_state=pause_state,
            request_timeout=config.request_timeout,
            max_resubmit_retries=config.max_resubmit_retries,
            resubmit_wait=config.resubmit_wait,
        )
        app.state.tokenizer = tok
        app.state.backend = backend
        app.state.pause_state = pause_state
        app.state.config = config
        app.state.session_store = SessionStore()
        app.state.session_store.set_admin_key(config.admin_api_key)
        app.state.chat_handler = ChatCompletionHandler(backend, tok)
        yield
        logger.info("Data proxy shutting down")

    app = FastAPI(title="AReaL Data Proxy", lifespan=lifespan)

    # =========================================================================
    # Health
    # =========================================================================

    @app.get("/health")
    async def health():
        store: SessionStore = app.state.session_store
        pause_state: PauseState = app.state.pause_state
        return {
            "status": "ok",
            "backend": config.backend_addr,
            "sessions": store.session_count,
            "paused": await pause_state.is_paused(),
        }

    # =========================================================================
    # Low-level generate — no authentication
    # =========================================================================

    @app.post("/generate")
    async def generate(req: GenerateRequest):
        if req.text is None and req.input_ids is None:
            raise HTTPException(
                status_code=400,
                detail="Either 'text' or 'input_ids' must be provided",
            )

        tok: TokenizerProxy = app.state.tokenizer
        backend: SGLangBackend = app.state.backend

        # Resolve input_ids
        if req.input_ids is not None:
            input_ids = req.input_ids
        else:
            input_ids = await tok.tokenize(req.text)  # type: ignore[arg-type]  # guarded by HTTPException above

        # Merge sampling params with defaults
        defaults = {
            "max_new_tokens": 512,
            "temperature": 1.0,
            "top_p": 1.0,
            "skip_special_tokens": False,
            "stop_token_ids": [tok.eos_token_id],
        }
        sampling_params = {**defaults, **(req.sampling_params or {})}

        # Call SGLang (with transparent pause/resubmit) — get all tokens at once
        result = await backend.generate(input_ids, sampling_params)

        # Stream back one token per SSE chunk
        return StreamingResponse(
            stream_tokens(result, tok),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # =========================================================================
    # Pause/Resume — internal control plane (no auth at data proxy level)
    # =========================================================================

    @app.post("/pause_generation")
    async def pause_generation():
        state: PauseState = app.state.pause_state
        await state.set_paused(True)
        await pause_backend(config.backend_addr)
        return {"status": "ok", "paused": True}

    @app.post("/continue_generation")
    async def continue_generation():
        state: PauseState = app.state.pause_state
        await resume_backend(config.backend_addr)
        await state.set_paused(False)
        return {"status": "ok", "paused": False}

    # =========================================================================
    # Session management (admin key / session key required)
    # =========================================================================

    @app.post("/rl/start_session", status_code=201)
    async def start_session(
        body: StartSessionRequest, request: Request
    ) -> StartSessionResponse:
        store: SessionStore = app.state.session_store
        _require_admin_key(request, store)
        try:
            session_id, session_api_key = store.start_session(
                body.task_id, body.api_key
            )
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))
        return StartSessionResponse(session_id=session_id, api_key=session_api_key)

    @app.post("/rl/end_session")
    async def end_session(request: Request):
        store: SessionStore = app.state.session_store
        session_id = _require_session_key(request, store)
        try:
            interaction_count = store.end_session(session_id)
        except KeyError:
            raise HTTPException(
                status_code=410, detail="Session already ended or expired"
            )
        return {"message": "success", "interaction_count": interaction_count}

    @app.post("/rl/set_reward")
    async def set_reward(body: SetRewardRequest, request: Request):
        store: SessionStore = app.state.session_store
        session_id = _require_session_key(request, store)
        session_data = store.get_session(session_id)
        if session_data is None:
            raise HTTPException(
                status_code=410, detail="Session already ended or expired"
            )
        session_data.update_last_access()

        completions = session_data.completions
        interaction_id = body.interaction_id
        if interaction_id is None:
            if len(completions) == 0:
                raise HTTPException(
                    status_code=400, detail="No interactions in session"
                )
            interaction_id = completions.last_interaction_id
        elif interaction_id not in completions:
            raise HTTPException(
                status_code=400,
                detail=f"Interaction {interaction_id} not found",
            )
        completions.set_reward(interaction_id, body.reward)
        return {"message": "success"}

    # =========================================================================
    # Chat completions — OpenAI-compatible
    #
    # If the bearer token is a known session key, use session cache.
    # Otherwise (no token, admin key, unknown key) → standalone mode.
    # Data proxy never rejects requests on /chat/completions.
    # =========================================================================

    @app.post("/chat/completions")
    async def chat_completions(body: CompletionCreateParams, request: Request):
        store: SessionStore = app.state.session_store

        # Try to resolve a session from the bearer token
        token = _try_extract_bearer_token(request)
        session_data_obj = None
        if token is not None:
            session_obj = store.get_session_by_api_key(token)
            if session_obj is not None:
                session_data_obj = session_obj
                session_data_obj.update_last_access()

        if session_data_obj is not None:
            # Session mode: use session cache
            session_data = session_data_obj
        else:
            # Standalone mode: no session, no caching
            session_data = types.SimpleNamespace(completions=None)

        chat_handler: ChatCompletionHandler = app.state.chat_handler

        # Determine if streaming before _call_client_create (for SSE wrapping)
        request_dict = body if isinstance(body, dict) else body
        is_streaming = False
        if isinstance(request_dict, BaseModel):
            dumped = request_dict.model_dump()
            is_streaming = dumped.get("stream", False) or False
        elif isinstance(request_dict, dict):
            is_streaming = request_dict.get("stream", False) or False

        result = await _call_client_create(
            create_fn=chat_handler.create,
            request=body,
            session_data=session_data,
            stream=is_streaming,
        )

        if is_streaming:
            # result is an async generator of ChatCompletionChunk

            async def _sse_stream():
                async for chunk in result:
                    yield f"data: {chunk.model_dump_json()}\n\n".encode()
                yield b"data: [DONE]\n\n"

            return StreamingResponse(
                _sse_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        return result

    # =========================================================================
    # Trajectory export (admin key required)
    # =========================================================================

    @app.post("/export_trajectories")
    async def export_trajectories(
        body: ExportTrajectoriesRequest, request: Request
    ) -> ExportTrajectoriesResponse:
        store: SessionStore = app.state.session_store
        _require_admin_key(request, store)

        session_data = store.get_session(body.session_id)
        if session_data is None:
            raise HTTPException(
                status_code=404, detail=f"Session {body.session_id} not found"
            )

        # Wait for session to complete (non-blocking)
        await session_data.wait_for_finish()

        # Export interactions
        interactions = session_data.export_interactions(
            discount=body.discount,
            style=body.style,
        )

        # Remove session from store
        store.remove_session(body.session_id)

        # Serialize for HTTP transport
        serialized = serialize_interactions(interactions)
        return ExportTrajectoriesResponse(interactions=serialized)

    # =========================================================================
    # Weight update forwarding (data proxy → co-located SGLang server)
    # =========================================================================

    app.post("/update_weights_from_disk")(update_weights_from_disk)
    app.post("/update_weights_from_distributed")(update_weights_from_distributed)
    app.post("/init_weights_update_group")(init_weights_update_group)
    app.post("/set_version")(set_version)

    # =========================================================================
    # Capacity management (mirrors proxy_rollout_server.grant_capacity)
    # =========================================================================

    @app.post("/grant_capacity")
    async def grant_capacity(request: Request):
        store: SessionStore = app.state.session_store
        _require_admin_key(request, store)
        return {"status": "ok"}


    # =========================================================================
    # Runtime backend reconfiguration (for fork-based deployment)
    # =========================================================================

    @app.post("/configure_backend")
    async def configure_backend(request: Request):
        """Reconfigure the SGLang backend address after process start.

        Called by the controller after ``fork_workers`` to tell this data
        proxy which SGLang server to connect to.
        """
        store: SessionStore = app.state.session_store
        _require_admin_key(request, store)
        body = await request.json()
        new_addr = body.get("backend_addr")
        if not new_addr:
            raise HTTPException(status_code=400, detail="backend_addr is required")
        pause_state: PauseState = app.state.pause_state
        new_backend = SGLangBackend(
            backend_addr=new_addr,
            pause_state=pause_state,
            request_timeout=app.state.config.request_timeout,
            max_resubmit_retries=app.state.config.max_resubmit_retries,
            resubmit_wait=app.state.config.resubmit_wait,
        )
        app.state.backend = new_backend
        app.state.config.backend_addr = new_addr
        # Re-create chat handler with new backend
        app.state.chat_handler = ChatCompletionHandler(new_backend, app.state.tokenizer)
        logger.info("Backend reconfigured to %s", new_addr)
        return {"status": "ok", "backend_addr": new_addr}

    return app
