from __future__ import annotations

import argparse
import asyncio
import os
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from areal.engine.sglang_ext._scheduler_patch import _AWEX_SCHEDULER_LAUNCHER
from areal.utils import logging

logger = logging.getLogger("ARealSGLangServer")


def _maybe_await(obj):
    if asyncio.iscoroutine(obj):
        return obj
    return None


def _to_json_response(success: bool, message: str):
    content = {"success": success, "message": message}
    status_code = 200 if success else 400
    return JSONResponse(content, status_code=status_code)


class AwexInitRequest(BaseModel):
    meta_server_addr: str
    engine_rank: int = 0
    num_engines: int = 1
    comm_backend: str = "file"
    enable_debug_mode: bool = False
    debug_mode_config: dict[str, Any] | None = None
    disable_weights_exchange_pipeline: bool = False
    enable_colocate_mode: bool = False
    weights_exchange_ipc_backend: str = "cuda"
    weights_comm_nccl_group_size: int = 1
    nnodes: int | None = None
    node_rank: int | None = None
    weights_validation_steps: int = 0
    validate_weights_every_n_steps: int = 1
    dump_weights_list_for_validation: list[str] | None = None
    dump_weights_dir_for_validation: str | None = None


class AwexUpdateRequest(BaseModel):
    step_id: int
    kwargs: dict[str, Any] | None = None


class _MockAwexAdapter:
    """Test-only adapter used when AREAL_AWEX_USE_MOCK_ADAPTER=1."""

    def __init__(self, **_kwargs):
        self._initialized = False

    def initialize(self):
        self._initialized = True

    def update_weights(self, step_id: int, **_kwargs):
        if not self._initialized:
            raise RuntimeError("Mock awex adapter not initialized")
        logger.info("Mock awex update accepted for step_id=%s", step_id)


def create_app(engine) -> FastAPI:
    app = FastAPI(title="AReaL SGLang Server")
    app.state.engine = engine
    app.state.awex_adapter = None

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models():
        model_path = (
            getattr(getattr(engine, "server_args", None), "model_path", None)
            or "unknown"
        )
        return {"data": [{"id": model_path, "object": "model"}]}

    @app.post("/generate")
    async def generate(request: Request):
        payload = await request.json()
        fn = getattr(engine, "async_generate", None)
        if fn is None:
            return JSONResponse(
                {"error": "engine.async_generate is unavailable"}, status_code=500
            )

        try:
            result = await fn(**payload)
        except TypeError:
            result = await fn(payload)

        if hasattr(result, "model_dump"):
            return result.model_dump()
        return result

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await generate(request)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await generate(request)

    @app.post("/encode")
    async def encode(request: Request):
        payload = await request.json()
        fn = getattr(engine, "encode", None)
        if fn is None:
            return JSONResponse(
                {"error": "engine.encode is unavailable"}, status_code=500
            )
        result = fn(**payload)
        if hasattr(result, "model_dump"):
            return result.model_dump()
        return result

    @app.post("/flush_cache")
    async def flush_cache():
        fn = getattr(engine, "flush_cache", None)
        if fn is None:
            return {"success": True, "message": "flush_cache unsupported"}
        ret = fn()
        maybe = _maybe_await(ret)
        if maybe is not None:
            await maybe
        return {"success": True}

    @app.post("/pause_generation")
    async def pause_generation():
        tm = getattr(engine, "tokenizer_manager", None)
        if tm is None or not hasattr(tm, "pause_generation"):
            return _to_json_response(
                False, "tokenizer_manager.pause_generation unavailable"
            )

        try:
            from sglang.srt.managers.io_struct import PauseGenerationReqInput

            obj = PauseGenerationReqInput()
            ret = tm.pause_generation(obj)
        except Exception:
            ret = tm.pause_generation()
        maybe = _maybe_await(ret)
        if maybe is not None:
            await maybe
        return _to_json_response(True, "Generation paused")

    @app.post("/continue_generation")
    async def continue_generation():
        tm = getattr(engine, "tokenizer_manager", None)
        if tm is None or not hasattr(tm, "continue_generation"):
            return _to_json_response(
                False, "tokenizer_manager.continue_generation unavailable"
            )

        try:
            from sglang.srt.managers.io_struct import ContinueGenerationReqInput

            obj = ContinueGenerationReqInput()
            ret = tm.continue_generation(obj)
        except Exception:
            ret = tm.continue_generation()
        maybe = _maybe_await(ret)
        if maybe is not None:
            await maybe
        return _to_json_response(True, "Generation continued")

    @app.post("/areal_pause_generation")
    async def areal_pause_generation():
        return await pause_generation()

    @app.post("/areal_continue_generation")
    async def areal_continue_generation():
        return await continue_generation()

    @app.post("/areal_awex_init")
    async def awex_init(request: AwexInitRequest):
        try:
            if os.environ.get("AREAL_AWEX_USE_MOCK_ADAPTER", "0") == "1":
                adapter = _MockAwexAdapter()
            else:
                from areal.engine.sglang_ext.sglang_awex_adapter import (
                    AwexSGLangServerAdapter,
                )

                adapter = AwexSGLangServerAdapter(
                    sgl_engine=engine,
                    meta_server_addr=request.meta_server_addr,
                    engine_rank=request.engine_rank,
                    num_engines=request.num_engines,
                    comm_backend=request.comm_backend,
                    enable_debug_mode=request.enable_debug_mode,
                    debug_mode_config=request.debug_mode_config,
                    disable_weights_exchange_pipeline=request.disable_weights_exchange_pipeline,
                    enable_colocate_mode=request.enable_colocate_mode,
                    weights_exchange_ipc_backend=request.weights_exchange_ipc_backend,
                    weights_comm_nccl_group_size=request.weights_comm_nccl_group_size,
                    nnodes=request.nnodes,
                    node_rank=request.node_rank,
                    weights_validation_steps=request.weights_validation_steps,
                    validate_weights_every_n_steps=request.validate_weights_every_n_steps,
                    dump_weights_list_for_validation=request.dump_weights_list_for_validation,
                    dump_weights_dir_for_validation=request.dump_weights_dir_for_validation,
                )
            await asyncio.to_thread(adapter.initialize)
            app.state.awex_adapter = adapter
            return _to_json_response(True, "Awex initialized")
        except Exception as exc:
            logger.exception("Awex init failed")
            return JSONResponse(
                {"success": False, "message": f"Awex init failed: {exc}"},
                status_code=500,
            )

    @app.post("/areal_awex_update")
    async def awex_update(request: AwexUpdateRequest):
        adapter = app.state.awex_adapter
        if adapter is None:
            return JSONResponse(
                {"success": False, "message": "Awex adapter not initialized."},
                status_code=400,
            )
        try:
            kwargs = request.kwargs or {}
            await asyncio.to_thread(adapter.update_weights, request.step_id, **kwargs)
            return _to_json_response(True, "Awex update done")
        except Exception as exc:
            logger.exception("Awex update failed")
            return JSONResponse(
                {"success": False, "message": f"Awex update failed: {exc}"},
                status_code=500,
            )

    return app


def _parse_args() -> tuple[Any, str, int]:
    """Parse SGLang server arguments from CLI.

    We parse host/port minimally here and pass through the rest to ServerArgs.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=30000)
    known, remaining = parser.parse_known_args()

    try:
        from sglang.srt.server_args import ServerArgs
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "sglang is required for areal.engine.sglang_ext.areal_sglang_server"
        ) from exc

    server_parser = argparse.ArgumentParser(add_help=False)
    ServerArgs.add_cli_args(server_parser)
    server_namespace, unknown = server_parser.parse_known_args(remaining)
    if unknown:
        logger.warning("Ignoring unknown SGLang server args: %s", unknown)

    # SGLang ServerArgs.from_cli_args expects argparse.Namespace in current
    # versions (not raw argv list).
    server_args = ServerArgs.from_cli_args(server_namespace)
    # Keep host/port from the outer parser so behavior matches old launcher.
    setattr(server_args, "host", known.host)
    setattr(server_args, "port", known.port)
    return server_args, known.host, known.port


def _build_engine(server_args):
    from sglang.srt.entrypoints.engine import Engine

    class AwexEngine(Engine):
        run_scheduler_process_func = _AWEX_SCHEDULER_LAUNCHER

    return AwexEngine(server_args=server_args)


def main() -> None:
    server_args, host, port = _parse_args()
    engine = _build_engine(server_args)

    app = create_app(engine)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
