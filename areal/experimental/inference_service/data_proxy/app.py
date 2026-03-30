from __future__ import annotations

import hmac
import types
from contextlib import asynccontextmanager

import orjson
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response as RawResponse
from fastapi.responses import StreamingResponse
from openai.types.chat.completion_create_params import CompletionCreateParams
from pydantic import BaseModel

from areal.experimental.inference_service.data_proxy.backend import SGLangBridgeBackend
from areal.experimental.inference_service.data_proxy.config import DataProxyConfig
from areal.experimental.inference_service.data_proxy.inf_bridge import InfBridge
from areal.experimental.inference_service.data_proxy.pause import PauseState
from areal.experimental.inference_service.data_proxy.session import (
    ExportTrajectoriesRequest,
    ExportTrajectoriesResponse,
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
from areal.infra.rpc import rtensor as rtensor_storage
from areal.infra.rpc.serialization import deserialize_value, serialize_value
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
    return InfBridge(
        backend=SGLangBridgeBackend(),
        backend_addr=backend_addr,
        pause_state=pause_state,
        request_timeout=config.request_timeout,
        max_resubmit_retries=config.max_resubmit_retries,
        resubmit_wait=config.resubmit_wait,
    )


def _create_areal_client(
    inf_bridge: InfBridge,
    tok: TokenizerProxy,
) -> ArealOpenAI:
    """Create an ArealOpenAI client backed by the given InfBridge."""
    return ArealOpenAI(
        engine=inf_bridge,
        tokenizer=tok._tok,
    )


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

        # InfBridge + ArealOpenAI for /chat/completions
        inf_bridge = _create_inf_bridge(config.backend_addr, pause_state, config)
        areal_client = _create_areal_client(inf_bridge, tok)

        app.state.tokenizer = tok
        app.state.inf_bridge = inf_bridge
        app.state.areal_client = areal_client
        app.state.pause_state = pause_state
        app.state.config = config
        app.state.session_store = SessionStore()
        app.state.session_store.set_admin_key(config.admin_api_key)
        app.state.version = 0
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
        inf_bridge: InfBridge = app.state.inf_bridge
        await inf_bridge.pause()
        return {"status": "ok", "paused": True}

    @app.post("/continue_generation")
    async def continue_generation():
        inf_bridge: InfBridge = app.state.inf_bridge
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
        areal_client: ArealOpenAI = app.state.areal_client

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

        # Build kwargs from request body
        if isinstance(body, BaseModel):
            kwargs = body.model_dump()
        else:
            kwargs = dict(body)

        # Remove model (ArealOpenAI ignores it)
        kwargs.pop("model", None)

        # Determine streaming
        is_streaming = kwargs.get("stream", False) or False

        # Apply defaults for temperature/top_p if not set
        if "temperature" not in kwargs:
            kwargs["temperature"] = 1.0
        if "top_p" not in kwargs:
            kwargs["top_p"] = 1.0

        try:
            result = await areal_client.chat.completions.create(
                areal_cache=session_data.completions,
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

        # Serialize for HTTP transport, storing tensors locally as RTensor shards
        from areal.infra.rpc.rtensor import RTensor

        for item in interactions.values():
            # Set the internal cache
            item.to_tensor_dict()
            # Remotize the tensor dict cache
            item._cache = RTensor.remotize(item._cache, node_addr=config.serving_addr)

        # serialize RTensors
        serialized = serialize_interactions(interactions)
        return ExportTrajectoriesResponse(interactions=serialized)

    # NOTE: /grant_capacity has been removed from data proxy. Capacity-based
    # staleness control is now managed at the router level — see
    # areal.experimental.inference_service.router.app for the /grant_capacity
    # endpoint.

    # =========================================================================
    # RTensor data storage endpoints
    #
    # These endpoints mirror the /data/ endpoints on rpc_server.py so that
    # RTensor.localize() can fetch tensor shards stored on this data proxy
    # via HttpRTensorBackend._fetch_tensor().
    # =========================================================================

    @app.post("/data/batch")
    async def retrieve_data_shard_batch(request: Request):
        """Retrieve multiple tensor shards in one request.

        Mirrors the ``POST /data/batch`` endpoint on the Flask RPC server
        (``rpc_server.py``) so that ``HttpRTensorBackend._fetch_shard_group``
        works against data-proxy addresses.
        """
        payload = await request.json()
        shard_ids = payload.get("shard_ids", [])
        if not isinstance(shard_ids, list) or not all(
            isinstance(sid, str) for sid in shard_ids
        ):
            raise HTTPException(
                status_code=400,
                detail="Expected JSON body with string list field 'shard_ids'",
            )

        data = []
        missing: list[str] = []
        for sid in shard_ids:
            try:
                data.append(rtensor_storage.fetch(sid))
            except KeyError:
                missing.append(sid)

        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"One or more requested shards were not found: {missing}",
            )

        serialized_data = serialize_value(data)
        data_bytes = orjson.dumps(serialized_data)
        logger.debug(
            "Retrieved %d RTensor shards in batch (size=%d bytes)",
            len(shard_ids),
            len(data_bytes),
        )
        return RawResponse(content=data_bytes, media_type="application/octet-stream")

    @app.put("/data/{shard_id}")
    async def store_data_shard(shard_id: str, request: Request):
        """Store a tensor shard in local RTensor storage."""
        data_bytes = await request.body()
        serialized_data = orjson.loads(data_bytes)
        data = deserialize_value(serialized_data)
        rtensor_storage.store(shard_id, data)
        logger.debug(
            "Stored RTensor shard %s (size=%d bytes)", shard_id, len(data_bytes)
        )
        return {"status": "ok", "shard_id": shard_id}

    @app.get("/data/{shard_id}")
    async def retrieve_data_shard(shard_id: str):
        """Retrieve a tensor shard from local RTensor storage."""
        try:
            data = rtensor_storage.fetch(shard_id)
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail=f"Shard {shard_id} not found",
            )
        serialized_data = serialize_value(data)
        data_bytes = orjson.dumps(serialized_data)
        return RawResponse(content=data_bytes, media_type="application/octet-stream")

    @app.delete("/data/clear")
    async def clear_data_shards(request: Request):
        """Clear specified tensor shards from local RTensor storage."""
        body = await request.json()
        shard_ids = body.get("shard_ids", [])
        if not isinstance(shard_ids, list):
            raise HTTPException(status_code=400, detail="'shard_ids' must be a list")
        cleared_count = sum(rtensor_storage.remove(sid) for sid in shard_ids)
        stats = dict(cleared_count=cleared_count, **rtensor_storage.storage_stats())
        logger.info("Cleared %d RTensor shards. Stats: %s", cleared_count, stats)
        stats.update({"status": "ok"})
        return stats

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
        new_areal_client = _create_areal_client(new_inf_bridge, tok)

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

    return app
