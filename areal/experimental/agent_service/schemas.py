"""Data models for the Agent Service Gateway and Worker.

These Pydantic models define the request/response schemas for:
- Gateway-Worker communication and worker registration.
- Worker RPC server endpoints (/run_episode, /health, /configure).
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel

# ------------------------------------------------------------------
# Gateway / Worker-pool models
# ------------------------------------------------------------------


class WorkerInfo(BaseModel):
    """Information about a registered Agent Worker."""

    worker_id: str
    address: str
    status: str = "idle"  # "idle", "busy", "unhealthy"
    last_heartbeat: float = 0.0

    @classmethod
    def create(cls, worker_id: str, address: str) -> WorkerInfo:
        """Create a new WorkerInfo with current timestamp."""
        return cls(worker_id=worker_id, address=address, last_heartbeat=time.time())


class DispatchRequest(BaseModel):
    """Request sent from Gateway to Agent Worker."""

    data: dict[str, Any]
    session_url: str
    agent_kwargs: dict[str, Any] | None = None
    agent_import_path: str | None = None


class DispatchResponse(BaseModel):
    """Response from Agent Worker to Gateway."""

    status: str
    result: Any | None = None
    error: str | None = None


class RegisterWorkerRequest(BaseModel):
    """Request from Agent Worker to register with Gateway."""

    worker_id: str
    address: str


# ------------------------------------------------------------------
# Worker RPC server models
# ------------------------------------------------------------------


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
