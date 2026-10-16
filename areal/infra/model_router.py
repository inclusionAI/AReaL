# SPDX-License-Identifier: Apache-2.0

"""Job-owned, sticky model router. Training data never enters this service.

Only nonstreaming SGLang /generate is supported. Payloads are forwarded as raw
bytes so model-specific extensions (images, R3, etc.) are not schema-filtered.
The controller alone closes admission, drains requests and publishes versions.
"""

from __future__ import annotations

import argparse
import asyncio
import hmac
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from areal.infra.model_routing import TRAJECTORY_HEADER, VERSION_HEADER
from areal.infra.utils.http import (
    get_default_httpx_limits,
    get_default_uvicorn_kwargs,
    validate_admin_api_key,
)
from areal.utils import logging

logger = logging.getLogger("ModelRouter")


@dataclass
class ModelRouterConfig:
    backends: tuple[str, ...]
    admin_api_key: str
    request_timeout: float = 120.0

    def __post_init__(self) -> None:
        self.backends = tuple(addr.rstrip("/") for addr in self.backends)
        if not self.backends or len(set(self.backends)) != len(self.backends):
            raise ValueError("Model Router requires distinct backend addresses")
        for addr in self.backends:
            url = httpx.URL(addr)
            if url.scheme not in ("http", "https") or not url.host or url.path != "/":
                raise ValueError(f"Expected a backend base URL, got {addr!r}")
        if not self.admin_api_key or self.request_timeout <= 0:
            raise ValueError("Model Router requires an admin key and positive timeout")


class ModelRouterState:
    """Single-event-loop accounting; selection and reservation are atomic."""

    def __init__(self, backends: tuple[str, ...]) -> None:
        self.backends = backends
        self.active = [0] * len(backends)
        self.bindings: dict[str, int] = {}
        self.active_keys: dict[str, int] = {}
        self.accepting = True
        self.failed = False
        self.version = 0
        self.cursor = 0
        self.condition = asyncio.Condition()

    async def admit(self, key: str, version: int) -> tuple[int, int]:
        async with self.condition:
            if self.failed:
                raise HTTPException(503, detail="model_router_failed")
            if not self.accepting:
                raise HTTPException(503, detail="model_router_paused")
            if version != self.version:
                raise HTTPException(409, detail="model_router_version_mismatch")
            index = self.bindings.get(key)
            if index is None:
                # Rotating ties avoid always preferring backend zero at low load.
                index = min(
                    range(len(self.backends)),
                    key=lambda i: (
                        self.active[i],
                        (i - self.cursor) % len(self.backends),
                    ),
                )
                self.bindings[key] = index
                self.cursor = (index + 1) % len(self.backends)
            self.active[index] += 1
            self.active_keys[key] = self.active_keys.get(key, 0) + 1
            return index, self.version

    async def finish(self, key: str, index: int, *, failed: bool) -> None:
        async with self.condition:
            self.active[index] -= 1
            self.active_keys[key] -= 1
            if not self.active_keys[key]:
                del self.active_keys[key]
            if failed:
                self.failed = True
                self.accepting = False
            self.condition.notify_all()


def create_app(config: ModelRouterConfig) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with httpx.AsyncClient(
            timeout=config.request_timeout,
            limits=get_default_httpx_limits(),
            trust_env=False,
        ) as client:
            app.state.http_client = client
            yield

    app = FastAPI(title="AReaL Model Router", lifespan=lifespan)
    state = ModelRouterState(config.backends)
    app.state.model_router = state

    def authorize(request: Request) -> None:
        expected = f"Bearer {config.admin_api_key}"
        if not hmac.compare_digest(request.headers.get("authorization", ""), expected):
            raise HTTPException(401, detail="Invalid Model Router credentials")

    @app.get("/health")
    async def health():
        return {"status": "failed" if state.failed else "ok"}

    @app.get("/status")
    async def status(request: Request):
        authorize(request)
        return {
            "accepting": state.accepting,
            "failed": state.failed,
            "version": state.version,
            "bindings": len(state.bindings),
            "backends": dict(zip(state.backends, state.active, strict=True)),
        }

    @app.post("/generate")
    async def generate(request: Request):
        authorize(request)
        key = request.headers.get(TRAJECTORY_HEADER, "")
        if not key or len(key) > 512:
            raise HTTPException(400, detail="A trajectory routing key is required")
        try:
            version = int(request.headers[VERSION_HEADER])
            body = await request.body()
            payload = json.loads(body)
            if not isinstance(payload, dict) or payload.get("stream"):
                raise ValueError("Only nonstreaming generation is supported")
            if not isinstance(payload.get("rid"), str) or not payload["rid"]:
                raise ValueError("A unique generation rid is required")
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(400, detail=str(exc)) from exc

        index, segment_version = await state.admit(key, version)
        failed = True
        try:
            response = await app.state.http_client.post(
                f"{state.backends[index]}/generate",
                content=body,
                headers={"Content-Type": "application/json"},
                timeout=config.request_timeout,
            )
            # A backend 5xx may hide a still-running request. Fail closed until
            # recover; never migrate/retry an active trajectory automatically.
            failed = response.status_code >= 500
            return Response(
                content=response.content,
                status_code=response.status_code,
                media_type=response.headers.get("content-type"),
                headers={VERSION_HEADER: str(segment_version)},
            )
        except httpx.HTTPError:
            logger.exception("Model backend transport failed; closing pool admission")
            return JSONResponse({"detail": "model_router_failed"}, status_code=502)
        finally:
            await state.finish(key, index, failed=failed)

    @app.post("/close")
    async def close(request: Request):
        authorize(request)
        async with state.condition:
            state.accepting = False
        return {"status": "ok"}

    @app.post("/fail")
    async def fail(request: Request):
        authorize(request)
        async with state.condition:
            state.accepting = False
            state.failed = True
            state.condition.notify_all()
        return {"status": "ok"}

    @app.post("/drain")
    async def drain(request: Request):
        authorize(request)
        async with state.condition:
            if state.accepting:
                raise HTTPException(409, detail="Close admission before draining")
            try:
                await asyncio.wait_for(
                    state.condition.wait_for(lambda: not any(state.active)),
                    timeout=config.request_timeout,
                )
            except TimeoutError as exc:
                state.failed = True
                raise HTTPException(503, detail="model_router_drain_timeout") from exc
            if state.failed:
                raise HTTPException(503, detail="model_router_failed")
        return {"status": "ok"}

    @app.post("/set_version")
    async def set_version(request: Request):
        authorize(request)
        body = await request.json()
        version = body.get("version")
        async with state.condition:
            if type(version) is int and version == state.version and not state.failed:
                return {"status": "ok", "version": version}
            if state.accepting or any(state.active) or state.failed:
                raise HTTPException(
                    409, detail="Close and drain before setting version"
                )
            if type(version) is not int or version < state.version:
                raise HTTPException(
                    400, detail="Expected a nondecreasing integer version"
                )
            state.version = version
        return {"status": "ok", "version": version}

    @app.post("/open")
    async def reopen(request: Request):
        authorize(request)
        async with state.condition:
            if any(state.active) or state.failed:
                raise HTTPException(409, detail="Model pool is not quiescent")
            state.accepting = True
        return {"status": "ok"}

    @app.post("/release")
    async def release(request: Request):
        authorize(request)
        body = await request.json()
        keys = body.get("keys")
        if not isinstance(keys, list) or any(not isinstance(k, str) for k in keys):
            raise HTTPException(400, detail="Expected a list of trajectory keys")
        async with state.condition:
            if any(k in state.active_keys for k in keys):
                raise HTTPException(409, detail="Cannot release active trajectories")
            for key in keys:
                state.bindings.pop(key, None)
        return {"status": "ok"}

    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="AReaL Model Router")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--backend", action="append", required=True)
    parser.add_argument("--admin-api-key", required=True)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    args = parser.parse_args()
    validate_admin_api_key(args.host, args.admin_api_key)
    app = create_app(
        ModelRouterConfig(tuple(args.backend), args.admin_api_key, args.request_timeout)
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        access_log=False,
        **get_default_uvicorn_kwargs(),
    )


if __name__ == "__main__":
    main()
