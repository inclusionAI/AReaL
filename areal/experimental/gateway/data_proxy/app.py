from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from areal.experimental.gateway.data_proxy.backend import (
    GenerationResult,
    SGLangBackend,
)
from areal.experimental.gateway.data_proxy.config import DataProxyConfig
from areal.experimental.gateway.data_proxy.tokenizer_proxy import TokenizerProxy

logger = logging.getLogger("DataProxy")


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


def create_app(config: DataProxyConfig) -> FastAPI:
    """Factory that creates the FastAPI app with lifespan-managed resources."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info(
            "Data proxy starting — backend=%s, tokenizer=%s",
            config.backend_addr,
            config.tokenizer_path,
        )
        app.state.tokenizer = TokenizerProxy(config.tokenizer_path)
        app.state.backend = SGLangBackend(config.backend_addr, config.request_timeout)
        app.state.config = config
        yield
        logger.info("Data proxy shutting down")

    app = FastAPI(title="AReaL Data Proxy", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok", "backend": config.backend_addr}

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

        # Call SGLang (non-streaming) — get all tokens at once
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

    return app
