"""RPC Server for Agent Service.

This module provides the FastAPI application factory (``create_app``) and
lifespan management for the Agent Service. It exposes HTTP endpoints that
accept requests from OpenAIProxyWorkflow (mode="service") to execute
agent.run() in an independent process.

The server implements a simple interface:
- POST /run_episode: Execute agent.run() with provided data and session_url
- GET /health: Health check

CLI entry point and argument parsing live in ``__main__.py``.
"""

from __future__ import annotations

import json
import os
import traceback
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from areal.infra.rpc.serialization import deserialize_value
from areal.utils import logging, seeding

from .config import AgentServiceConfig
from .service import AgentService

logger = logging.getLogger("AgentRPCServer")

# Environment variable names for multi-worker mode configuration
ENV_AGENT_HOST = "AREAL_AGENT_HOST"
ENV_AGENT_PORT = "AREAL_AGENT_PORT"
ENV_AGENT_WORKERS = "AREAL_AGENT_WORKERS"
ENV_AGENT_IMPORT_PATH_INTERNAL = "AREAL_AGENT_IMPORT_PATH_INTERNAL"
ENV_AGENT_REUSE_INTERNAL = "AREAL_AGENT_REUSE_INTERNAL"
ENV_AGENT_INIT_KWARGS_INTERNAL = "AREAL_AGENT_INIT_KWARGS_INTERNAL"

# Global service instance for multi-worker mode (each worker has its own)
_service: AgentService | None = None


class RunEpisodeRequest(BaseModel):
    """Request model for /run_episode endpoint."""

    data: dict[str, Any]
    session_url: str
    agent_kwargs: dict[str, Any] | None = None
    agent_import_path: str | None = None  # Per-request agent selection


class RunEpisodeResponse(BaseModel):
    """Response model for /run_episode endpoint."""

    status: str
    result: Any | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    """Response model for /health endpoint."""

    status: str
    running: bool
    agent_import_path: str
    agent_reuse: bool


class ConfigureRequest(BaseModel):
    """Request model for /configure endpoint."""

    config: dict[str, Any] | None = None
    role: str | None = None
    rank: int | None = None


class ConfigureResponse(BaseModel):
    """Response model for /configure endpoint."""

    status: str
    message: str


def _get_agent_config_from_env() -> tuple[str | None, bool, dict[str, Any]]:
    """Get agent configuration from internal environment variables (multi-worker mode).

    Returns
    -------
    tuple[str | None, bool, dict[str, Any]]
        (agent_import_path, agent_reuse, agent_init_kwargs)
    """
    agent_import_path = os.environ.get(ENV_AGENT_IMPORT_PATH_INTERNAL) or None
    agent_reuse = os.environ.get(ENV_AGENT_REUSE_INTERNAL, "false").lower() in (
        "true",
        "1",
    )
    agent_init_kwargs: dict[str, Any] = {}
    kwargs_str = os.environ.get(ENV_AGENT_INIT_KWARGS_INTERNAL, "")
    if kwargs_str:
        try:
            agent_init_kwargs = json.loads(kwargs_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON for agent_init_kwargs, using empty dict: {e}")
            pass
    return agent_import_path, agent_reuse, agent_init_kwargs


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager.

    Each worker process independently initializes its own AgentService instance.
    Configuration is read from environment variables set by the main process
    before forking.
    """
    global _service

    # Read configuration from environment variables
    agent_import_path, agent_reuse, agent_init_kwargs = _get_agent_config_from_env()
    host = os.environ.get(ENV_AGENT_HOST, "0.0.0.0")
    port = int(os.environ.get(ENV_AGENT_PORT, "8300"))
    workers = int(os.environ.get(ENV_AGENT_WORKERS, "1"))

    config = AgentServiceConfig(host=host, port=port, workers=workers)
    _service = AgentService(
        agent_import_path=agent_import_path,
        config=config,
        agent_reuse=agent_reuse,
        agent_init_kwargs=agent_init_kwargs,
    )

    await _service.start()
    logger.info(f"Worker {os.getpid()} started AgentService")

    yield

    await _service.stop()
    logger.info(f"Worker {os.getpid()} stopped AgentService")


def create_app() -> FastAPI:
    """Factory function for creating FastAPI app.

    Returns
    -------
    FastAPI
        The FastAPI application with configured endpoints.
    """
    app = FastAPI(
        title="Agent Service RPC Server",
        description="RPC endpoints for Agent Service",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Health check endpoint."""
        if _service is None:
            raise HTTPException(status_code=503, detail="Service not initialized")
        health_info = await _service.health_check()
        return HealthResponse(**health_info)

    @app.post("/run_episode", response_model=RunEpisodeResponse)
    async def run_episode(request: RunEpisodeRequest) -> RunEpisodeResponse:
        """Execute a single agent episode."""
        if _service is None:
            raise HTTPException(status_code=503, detail="Service not initialized")
        try:
            result = await _service.run_episode(
                data=request.data,
                session_url=request.session_url,
                agent_kwargs=request.agent_kwargs,
                agent_import_path=request.agent_import_path,
            )
            return RunEpisodeResponse(status="success", result=result)
        except RuntimeError as e:
            logger.warning(f"run_episode failed (service not running): {e}")
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            logger.warning(f"run_episode failed: {e}\n{traceback.format_exc()}")
            return RunEpisodeResponse(status="error", result=None, error=str(e))

    @app.post("/configure")
    async def configure(raw_request: Request):
        """Configure the Agent Service worker.

        Called by scheduler to configure the worker with experiment settings.
        Sets random seed based on config for reproducibility.
        """
        data = await raw_request.json()
        config = deserialize_value(data.get("config"))
        rank = data.get("rank", 0)
        seeding.set_random_seed(config.seed, key=f"agent{rank}")
        return {"status": "success"}

    return app
