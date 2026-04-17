# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import hmac
import json
from contextlib import asynccontextmanager
from typing import Any

import httpx
import orjson
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.responses import Response as RawResponse
from flask import Flask

from areal.experimental.inference_service.data_proxy.backend import (
    SGLangBridgeBackend,
    VLLMBridgeBackend,
)
from areal.experimental.inference_service.data_proxy.config import DataProxyConfig
from areal.experimental.inference_service.data_proxy.inf_bridge import InfBridge
from areal.experimental.inference_service.data_proxy.pause import PauseState
from areal.experimental.inference_service.data_proxy.session import (
    ExportTrajectoriesRequest,
    ExportTrajectoriesResponse,
    ReadyNotification,
    SessionData,
    SessionStore,
    SetRewardRequest,
    StartSessionRequest,
    StartSessionResponse,
)
from areal.experimental.inference_service.data_proxy.tokenizer_proxy import (
    TokenizerProxy,
)
from areal.experimental.openai.client import ArealOpenAI
from areal.experimental.openai.proxy.server import serialize_interactions
from areal.infra.rpc.guard.data_blueprint import (
    data_bp,
)
from areal.utils import logging

logger = logging.getLogger("InferenceDataProxy")


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


def _resolve_session_from_token(
    token: str | None,
    store: SessionStore,
) -> SessionData | None:
    """Resolve a session from the bearer token.

    Session key → lookup by API key.
    Admin key → persistent HITL session.
    """
    if token is None:
        return None
    session = store.get_session_by_api_key(token)
    if session is not None:
        return session
    if hmac.compare_digest(token, store.admin_api_key):
        return store.get_or_create_hitl_session()
    return None


def _try_extract_bearer_token(request: Request) -> str | None:
    """Extract bearer token if present. Returns None if missing/malformed.

    Unlike _extract_bearer_token, this never raises — it's for endpoints
    that accept requests with or without auth.
    """
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    return None


def _create_inf_bridge(
    backend_addr: str,
    pause_state: PauseState,
    config: DataProxyConfig,
) -> InfBridge:
    """Create an InfBridge instance from proxy config."""
    if config.backend_type == "sglang":
        backend = SGLangBridgeBackend()
    elif config.backend_type == "vllm":
        backend = VLLMBridgeBackend()
    else:
        raise ValueError(f"Unsupported backend_type: {config.backend_type!r}")

    return InfBridge(
        backend=backend,
        backend_addr=backend_addr,
        pause_state=pause_state,
        request_timeout=config.request_timeout,
        max_resubmit_retries=config.max_resubmit_retries,
        resubmit_wait=config.resubmit_wait,
    )


def _create_areal_client(
    inf_bridge: InfBridge,
    tok: TokenizerProxy,
    config: DataProxyConfig,
) -> ArealOpenAI:
    """Create an ArealOpenAI client backed by the given InfBridge."""
    return ArealOpenAI(
        engine=inf_bridge,
        tokenizer=tok._tok,
        tool_call_parser=config.tool_call_parser,
        reasoning_parser=config.reasoning_parser,
        engine_max_tokens=config.engine_max_tokens,
        chat_template_type=config.chat_template_type,
    )


async def _post_online_ready_callback(
    callback_server_addr: str,
    admin_api_key: str,
    notification: ReadyNotification,
    timeout: float,
) -> bool:
    if not callback_server_addr:
        return False

    callback_base = callback_server_addr.rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{callback_base}/callback/online_ready",
                json={
                    "session_id": notification.session_id,
                    "trajectory_id": notification.trajectory_id,
                },
                headers={"Authorization": f"Bearer {admin_api_key}"},
            )
        if resp.status_code >= 400:
            logger.warning(
                "Online ready callback failed for %s/%s with %d: %s",
                notification.session_id,
                notification.trajectory_id,
                resp.status_code,
                resp.text,
            )
            return False
        return True
    except Exception as exc:
        logger.warning(
            "Online ready callback unreachable for %s/%s: %s",
            notification.session_id,
            notification.trajectory_id,
            exc,
        )
        return False


async def _flush_ready_trajectories(app: FastAPI) -> None:
    store: SessionStore = app.state.session_store
    config: DataProxyConfig = app.state.config

    for ready_result in store.finalize_rewarded_trajectories():
        logger.info(
            "Trajectory ready: session=%s trajectory=%s interactions=%s",
            ready_result.session_id,
            ready_result.trajectory_id,
            ready_result.interaction_count,
        )

    pending_notifications = store.pending_online_callbacks()
    for notification in pending_notifications:
        delivered = await _post_online_ready_callback(
            config.callback_server_addr,
            config.admin_api_key,
            notification,
            config.request_timeout,
        )
        if delivered:
            store.mark_online_callback_delivered(
                notification.session_id,
                notification.trajectory_id,
            )


async def _ready_trajectory_loop(app: FastAPI) -> None:
    while True:
        await _flush_ready_trajectories(app)
        await asyncio.sleep(0.1)


def create_app(config: DataProxyConfig) -> FastAPI:
    """Factory that creates the FastAPI app with lifespan-managed resources."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info(
            "Data proxy starting — backend=%s, tokenizer=%s",
            config.backend_addr or "(none)",
            config.tokenizer_path,
        )

        pause_state = PauseState()
        app.state.pause_state = pause_state
        app.state.config = config
        app.state.session_store = SessionStore(
            set_reward_finish_timeout=config.set_reward_finish_timeout,
        )
        app.state.session_store.set_admin_key(config.admin_api_key)
        app.state.version = 0

        if not config.backend_addr:
            app.state.tokenizer = None
            app.state.inf_bridge = None
            app.state.areal_client = None
        else:
            tok = TokenizerProxy(config.tokenizer_path)
            inf_bridge = _create_inf_bridge(config.backend_addr, pause_state, config)
            areal_client = _create_areal_client(inf_bridge, tok, config)
            app.state.tokenizer = tok
            app.state.inf_bridge = inf_bridge
            app.state.areal_client = areal_client

        ready_task = asyncio.create_task(_ready_trajectory_loop(app))
        try:
            yield
        finally:
            ready_task.cancel()
            try:
                await ready_task
            except asyncio.CancelledError:
                pass
        logger.info("Data proxy shutting down")

    app = FastAPI(title="AReaL Data Proxy", lifespan=lifespan)
    _registered_models: dict[str, dict[str, str | None]] = {}

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
            "version": app.state.version,
        }

    @app.post("/configure")
    async def configure():
        return {"status": "ok"}

    # =========================================================================
    # Pause/Resume — internal control plane (no auth at data proxy level)
    # =========================================================================

    @app.post("/pause_generation")
    async def pause_generation():
        inf_bridge: InfBridge | None = app.state.inf_bridge
        if inf_bridge is None:
            raise HTTPException(
                status_code=503,
                detail="No inference backend configured (external model mode).",
            )
        await inf_bridge.pause()
        return {"status": "ok", "paused": True}

    @app.post("/continue_generation")
    async def continue_generation():
        inf_bridge: InfBridge | None = app.state.inf_bridge
        if inf_bridge is None:
            raise HTTPException(
                status_code=503,
                detail="No inference backend configured (external model mode).",
            )
        await inf_bridge.resume()
        return {"status": "ok", "paused": False}

    # =========================================================================
    # Version management — internal control plane (no auth at data proxy level)
    # =========================================================================

    @app.post("/set_version")
    async def set_version(request: Request):
        body = await request.json()
        version = body.get("version")
        if version is None or not isinstance(version, int):
            raise HTTPException(status_code=400, detail="'version' (int) is required")
        app.state.version = version
        return {"status": "ok", "version": version}

    @app.get("/get_version")
    async def get_version():
        return {"version": app.state.version}

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

    @app.post("/rl/set_reward")
    async def set_reward(body: SetRewardRequest, request: Request):
        store: SessionStore = app.state.session_store
        token = _extract_bearer_token(request)
        session = _resolve_session_from_token(token, store)
        if session is None:
            raise HTTPException(
                status_code=401, detail="Invalid or expired session API key."
            )

        try:
            reward_result = session.set_reward(
                interaction_id=body.interaction_id,
                reward=body.reward,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return {
            "message": "success",
            "interaction_count": reward_result.interaction_count,
            "session_id": reward_result.session_id,
            "trajectory_id": reward_result.trajectory_id,
            "trajectory_ready": reward_result.trajectory_id is not None,
            "ready_transition": reward_result.ready_transition,
        }

    # =========================================================================
    # Chat completions — OpenAI-compatible
    #
    # If the bearer token is a known session key, use session cache.
    # Otherwise (no token, admin key, unknown key) → standalone mode.
    # Data proxy never rejects requests on /chat/completions.
    # =========================================================================

    @app.post("/chat/completions")
    async def chat_completions(request: Request):
        raw_body = await request.body()
        try:
            body_json = json.loads(raw_body)
        except (json.JSONDecodeError, AttributeError):
            raise HTTPException(status_code=400, detail="Invalid JSON body")

        model_name = body_json.get("model")
        store: SessionStore = app.state.session_store

        token = _try_extract_bearer_token(request)
        session = _resolve_session_from_token(token, store)
        if session is not None:
            session.update_last_access()

        # -----------------------------------------------------------------
        # External model path: model is a registered external model name
        # -----------------------------------------------------------------
        ext_info = _registered_models.get(model_name) if model_name else None
        if ext_info is not None and ext_info.get("url"):
            ext_url = (ext_info["url"] or "").rstrip("/")
            ext_model = ext_info["model"]
            provider_api_key = ext_info.get("api_key")

            forward_body = dict(body_json)
            if ext_model is not None:
                forward_body["model"] = ext_model
            else:
                forward_body.pop("model", None)

            _skip = {"host", "content-length", "transfer-encoding", "authorization"}
            forward_headers = {
                k: v for k, v in dict(request.headers).items() if k.lower() not in _skip
            }
            if provider_api_key:
                forward_headers["authorization"] = f"Bearer {provider_api_key}"

            is_streaming = forward_body.get("stream", False) or False
            messages = body_json.get("messages", [])

            if is_streaming:
                collected_chunks: list[str] = []

                async def _stream_and_cache():
                    success = False
                    try:
                        async with httpx.AsyncClient(
                            timeout=httpx.Timeout(config.request_timeout)
                        ) as client:
                            async with client.stream(
                                "POST",
                                f"{ext_url}/chat/completions",
                                json=forward_body,
                                headers=forward_headers,
                            ) as resp:
                                if resp.status_code != 200:
                                    error_body = await resp.aread()
                                    yield (
                                        f"data: {json.dumps({'error': error_body.decode()})}\n\n".encode()
                                    )
                                    return
                                async for chunk in resp.aiter_bytes():
                                    decoded = chunk.decode("utf-8", errors="replace")
                                    collected_chunks.append(decoded)
                                    yield chunk
                                success = True
                    except Exception as exc:
                        logger.error(
                            "External stream error for %s: %s", model_name, exc
                        )
                        yield f"data: {json.dumps({'error': str(exc)})}\n\n".encode()
                    finally:
                        if success and collected_chunks and session is not None:
                            session.add_string_interaction(
                                messages,
                                "".join(collected_chunks),
                            )

                return StreamingResponse(
                    _stream_and_cache(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                    },
                )

            full_url = f"{ext_url}/chat/completions"
            try:
                async with httpx.AsyncClient(timeout=config.request_timeout) as client:
                    resp = await client.post(
                        full_url,
                        json=forward_body,
                        headers=forward_headers,
                    )
            except Exception as exc:
                raise HTTPException(
                    status_code=502, detail=f"External API error: {exc}"
                )

            if resp.status_code != 200:
                logger.error(
                    "External API returned %d for %s: %s",
                    resp.status_code,
                    full_url,
                    resp.text[:500],
                )

            response_str = resp.text

            if resp.status_code == 200 and session is not None:
                session.add_string_interaction(messages, response_str)

            return RawResponse(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type"),
            )

        # -----------------------------------------------------------------
        # Internal model path: use AReaL inference server
        # -----------------------------------------------------------------
        areal_client: ArealOpenAI = app.state.areal_client

        if session is not None:
            areal_cache: Any = session.active_completions
        else:
            areal_cache = None

        # Build kwargs from request body
        kwargs = dict(body_json)
        # Remove model (ArealOpenAI ignores it)
        kwargs.pop("model", None)

        # Determine streaming
        is_streaming = kwargs.get("stream", False) or False

        # Apply defaults for temperature/top_p if not set
        if "temperature" not in kwargs:
            kwargs["temperature"] = 1.0
        if "top_p" not in kwargs:
            kwargs["top_p"] = 1.0

        create_fn: Any = areal_client.chat.completions.create

        try:
            result = await create_fn(
                areal_cache=areal_cache,
                **kwargs,
            )
        except ValueError as e:
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

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

    @app.post("/register_model")
    async def register_model(request: Request):
        store: SessionStore = app.state.session_store
        body = await request.json()
        name = body.get("name") or body.get("model")
        url = body.get("url", "")
        model = body.get("model", name)
        api_key = body.get("api_key")
        if not name:
            raise HTTPException(status_code=400, detail="model name is required")
        _registered_models[name] = {"url": url, "model": model, "api_key": api_key}
        logger.info("Model registered: name=%s url=%s", name, url or "(internal)")
        return {"status": "ok", "name": name}

    # =========================================================================
    # Trajectory export (admin key required)
    # =========================================================================

    @app.post("/export_trajectories")
    async def export_trajectories(
        body: ExportTrajectoriesRequest, request: Request
    ) -> ExportTrajectoriesResponse:
        store: SessionStore = app.state.session_store
        _require_admin_key(request, store)

        session = store.get_session(body.session_id)
        if session is None:
            raise HTTPException(
                status_code=404, detail=f"Session {body.session_id} not found"
            )

        try:
            _, interactions = session.export_trajectory(
                discount=body.discount,
                style=body.style,
                trajectory_id=body.trajectory_id,
            )
        except KeyError as exc:
            detail = str(exc).strip('"')
            status_code = 404 if body.trajectory_id is not None else 409
            raise HTTPException(status_code=status_code, detail=detail) from exc

        if body.remove_session:
            store.remove_session(body.session_id)

        # Serialize for HTTP transport, storing tensors locally as RTensor shards
        from areal.infra.rpc.rtensor import RTensor

        for item in interactions.values():
            if item.has_tensor_data:
                # Set the internal cache
                item.to_tensor_dict()
                # Remotize the tensor dict cache
                item._cache = RTensor.remotize(
                    item._cache, node_addr=config.serving_addr
                )

        # serialize RTensors
        serialized = serialize_interactions(interactions)
        return ExportTrajectoriesResponse(interactions=serialized)

    # NOTE: /grant_capacity has been removed from data proxy. Capacity-based
    # staleness control is now managed at the router level — see
    # areal.experimental.inference_service.router.app for the /grant_capacity
    # endpoint.

    # =========================================================================
    # Runtime backend reconfiguration (for fork-based deployment)
    # =========================================================================

    @app.post("/configure_backend")
    async def configure_backend(request: Request):
        """Reconfigure the inference backend address after process start.

        Administrative endpoint to dynamically change which SGLang server
        this data proxy connects to.
        """
        store: SessionStore = app.state.session_store
        _require_admin_key(request, store)
        body = await request.json()
        new_addr = body.get("backend_addr")
        if not new_addr:
            raise HTTPException(status_code=400, detail="backend_addr is required")
        pause_state: PauseState = app.state.pause_state
        tok: TokenizerProxy = app.state.tokenizer

        # Recreate InfBridge + ArealOpenAI with new backend address
        new_inf_bridge = _create_inf_bridge(new_addr, pause_state, app.state.config)
        new_areal_client = _create_areal_client(new_inf_bridge, tok, app.state.config)

        # Build updated config copy, then swap all three state fields.
        # Concurrent requests already hold their own references so they
        # finish with the old backend; new requests see the new one.
        from dataclasses import replace as _dc_replace

        new_config = _dc_replace(app.state.config, backend_addr=new_addr)
        app.state.config = new_config
        app.state.inf_bridge = new_inf_bridge
        app.state.areal_client = new_areal_client

        logger.info("Backend reconfigured to %s", new_addr)
        return {"status": "ok", "backend_addr": new_addr}

    # =========================================================================
    # RTensor data storage endpoints
    #
    # These endpoints are now mounted from the legacy Flask data_blueprint
    # (areal.infra.rpc.guard.data_blueprint) to ensure a single source of
    # truth for RTensor storage logic.
    #
    # This mount provides:
    # - POST   /data/batch
    # - PUT    /data/<shard_id>
    # - GET    /data/<shard_id>
    # - DELETE /data/clear
    # =========================================================================
    flask_shim = Flask("data_proxy_shim")
    flask_shim.register_blueprint(data_bp)
    app.mount("/", WSGIMiddleware(flask_shim))

    return app
