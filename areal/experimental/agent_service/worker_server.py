"""RPC Server for Agent Service.

This module provides the FastAPI application factory (``create_app``) and
lifespan management for the Agent Service. It exposes HTTP endpoints that
accept requests from OpenAIProxyWorkflow (mode="service") to execute
agent.run() in an independent process.

The server implements a simple interface:
- POST /run_episode: Execute agent.run() with provided data and session_url
- GET /health: Health check

This module also provides a ``main()`` entry point for direct invocation.
"""

from __future__ import annotations

import json
import os
import traceback
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request

from areal.infra.rpc.serialization import deserialize_value
from areal.utils import logging, seeding

from .config import AgentServiceConfig
from .schemas import (
    HealthResponse,
    RunEpisodeRequest,
    RunEpisodeResponse,
)
from .service import AgentService

logger = logging.getLogger("AgentRPCServer")

# Environment variable names for multi-worker mode configuration
ENV_AGENT_HOST = "AREAL_AGENT_HOST"
ENV_AGENT_PORT = "AREAL_AGENT_PORT"
ENV_AGENT_IMPORT_PATH_INTERNAL = "AREAL_AGENT_IMPORT_PATH_INTERNAL"
ENV_AGENT_REUSE_INTERNAL = "AREAL_AGENT_REUSE_INTERNAL"
ENV_AGENT_INIT_KWARGS_INTERNAL = "AREAL_AGENT_INIT_KWARGS_INTERNAL"

# Environment variable names for Gateway worker registration
ENV_AGENT_GATEWAY_ADDR = "AREAL_AGENT_GATEWAY_ADDR"
ENV_AGENT_WORKER_ID = "AREAL_AGENT_WORKER_ID"
ENV_AGENT_WORKER_ADDR = "AREAL_AGENT_WORKER_ADDR"
# Global service instance for multi-worker mode (each worker has its own)
_service: AgentService | None = None


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

    In worker mode (AREAL_AGENT_GATEWAY_ADDR is set), the worker registers itself
    with the Gateway on startup and unregisters on shutdown.
    """
    global _service

    # Read configuration from environment variables
    agent_import_path, agent_reuse, agent_init_kwargs = _get_agent_config_from_env()
    host = os.environ.get(ENV_AGENT_HOST, "0.0.0.0")
    port = int(os.environ.get(ENV_AGENT_PORT, "8300"))

    config = AgentServiceConfig(host=host, port=port)
    _service = AgentService(
        agent_import_path=agent_import_path,
        config=config,
        agent_reuse=agent_reuse,
        agent_init_kwargs=agent_init_kwargs,
    )

    await _service.start()
    logger.info(f"Worker {os.getpid()} started AgentService")

    # Gateway registration (worker mode only)
    gateway_addr = os.environ.get(ENV_AGENT_GATEWAY_ADDR)
    worker_id = os.environ.get(ENV_AGENT_WORKER_ID, f"worker-{os.getpid()}")
    worker_addr = os.environ.get(ENV_AGENT_WORKER_ADDR, f"http://{host}:{port}")
    if gateway_addr:
        await _register_with_gateway(gateway_addr, worker_id, worker_addr)

    yield

    # Gateway unregistration (worker mode only)
    if gateway_addr:
        await _unregister_from_gateway(gateway_addr, worker_id)

    await _service.stop()
    logger.info(f"Worker {os.getpid()} stopped AgentService")


async def _register_with_gateway(
    gateway_addr: str, worker_id: str, worker_addr: str
) -> None:
    """Register this worker with the Gateway.

    Parameters
    ----------
    gateway_addr : str
        HTTP address of the Gateway (e.g., 'http://host:8300').
    worker_id : str
        Unique identifier for this worker.
    worker_addr : str
        HTTP address of this worker (e.g., 'http://host:8301').
    """
    import aiohttp

    url = f"{gateway_addr}/register_worker"
    payload = {"worker_id": worker_id, "address": worker_addr}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status == 200:
                    logger.info(
                        "Registered worker %s at %s with Gateway %s",
                        worker_id,
                        worker_addr,
                        gateway_addr,
                    )
                else:
                    body = await resp.text()
                    logger.warning(
                        "Gateway registration returned %d: %s", resp.status, body
                    )
    except Exception as e:
        logger.warning("Failed to register with Gateway %s: %s", gateway_addr, e)


async def _unregister_from_gateway(gateway_addr: str, worker_id: str) -> None:
    """Unregister this worker from the Gateway.

    Parameters
    ----------
    gateway_addr : str
        HTTP address of the Gateway.
    worker_id : str
        Unique identifier for this worker.
    """
    import aiohttp

    url = f"{gateway_addr}/unregister_worker"
    payload = {"worker_id": worker_id, "address": ""}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status == 200:
                    logger.info(
                        "Unregistered worker %s from Gateway %s",
                        worker_id,
                        gateway_addr,
                    )
                else:
                    body = await resp.text()
                    logger.warning(
                        "Gateway unregistration returned %d: %s", resp.status, body
                    )
    except Exception as e:
        logger.warning("Failed to unregister from Gateway %s: %s", gateway_addr, e)


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


def main() -> None:
    """Entry point for the Agent Worker process (internal, launched by Scheduler)."""
    import argparse

    import uvicorn

    from areal.utils.network import find_free_ports, gethostip

    parser = argparse.ArgumentParser(description="Agent Service Worker")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=0)
    args, _ = parser.parse_known_args()

    host = gethostip() if args.host == "0.0.0.0" else args.host
    port = args.port if args.port != 0 else find_free_ports(1)[0]

    # Set env vars that worker_server.lifespan() reads
    gateway_addr = os.environ.get(ENV_AGENT_GATEWAY_ADDR, "")
    worker_id = os.environ.get(ENV_AGENT_WORKER_ID, f"worker-{host}-{port}")
    worker_addr = os.environ.get(ENV_AGENT_WORKER_ADDR, f"http://{host}:{port}")
    os.environ[ENV_AGENT_WORKER_ID] = worker_id
    os.environ[ENV_AGENT_WORKER_ADDR] = worker_addr

    # Set config env vars for lifespan() to read
    os.environ[ENV_AGENT_HOST] = host
    os.environ[ENV_AGENT_PORT] = str(port)
    # AREAL_AGENT_IMPORT_PATH_INTERNAL etc are set by Scheduler env_vars

    logger.info(
        "Starting Agent Worker on %s:%d (gateway=%s, worker_id=%s)",
        host,
        port,
        gateway_addr or "none",
        worker_id,
    )
    try:
        uvicorn.run(
            "areal.experimental.agent_service.worker_server:create_app",
            factory=True,
            host="0.0.0.0",
            port=port,
            workers=1,
            log_level="warning",
            access_log=False,
        )
    except KeyboardInterrupt:
        logger.info("Shutting down Agent Worker")


if __name__ == "__main__":
    main()
