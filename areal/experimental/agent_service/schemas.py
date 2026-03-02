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
    status: str = "healthy"  # "healthy", "unhealthy"
    last_heartbeat: float = 0.0

    @classmethod
    def create(cls, worker_id: str, address: str) -> WorkerInfo:
        """Create a new WorkerInfo with current timestamp."""
        return cls(worker_id=worker_id, address=address, last_heartbeat=time.time())


class RegisterWorkerRequest(BaseModel):
    """Request from Agent Worker to register with Gateway."""

    worker_id: str
    address: str


# ------------------------------------------------------------------
# Episode request / response
# ------------------------------------------------------------------


class RunEpisodeRequest(BaseModel):
    """Request to run an episode via the Agent Service."""

    data: dict[str, Any]
    session_url: str
    agent_kwargs: dict[str, Any] | None = None
    agent_import_path: str | None = None


class RunEpisodeResponse(BaseModel):
    """Response from running an episode."""

    status: str
    result: float | dict[str, float] | None = None
    error: str | None = None


# ------------------------------------------------------------------
# Health
# ------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Response for health check endpoint.

    Used by both Gateway (/health with workers stats) and
    Worker (/health with running/agent info).
    """

    status: str
    # Gateway fields
    workers: dict[str, int] | None = None
    # Worker fields
    running: bool | None = None
    agent_import_path: str | None = None
    agent_reuse: bool | None = None
