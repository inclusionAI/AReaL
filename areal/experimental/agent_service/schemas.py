"""Data models for the Agent Service Gateway and Worker.

These Pydantic models define the request/response schemas for:
- Submit/poll pattern via /submit and /result/{task_id} endpoints (ZMQ-based worker registration).
- Worker RPC server endpoints (/run_episode, /health, /configure).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

# ------------------------------------------------------------------
# Async submit/result models (ZMQ worker pattern)
# ------------------------------------------------------------------


class SubmitEpisodeResponse(BaseModel):
    """Response from submitting an episode for execution.

    The Gateway returns a task_id that can be polled via /result/{task_id}
    to retrieve the result when ready.
    """

    task_id: str
    status: str = "submitted"


class TaskResultResponse(BaseModel):
    """Response with the result of a submitted task.

    Returned by /result/{task_id} endpoint. Status field indicates
    the current state: 'pending' (not done), 'completed' (success),
    'error' (execution failed), or 'timeout' (took too long).
    """

    task_id: str
    status: str  # "pending", "completed", "error", "timeout"
    result: float | dict[str, float] | None = None
    error: str | None = None


class CollectResultsRequest(BaseModel):
    """Request to collect results for multiple tasks (batch operation).

    Used for batch polling of results across multiple task_ids.
    Supports future scalability for bulk result retrieval.
    """

    task_ids: list[str]


class CollectResultsResponse(BaseModel):
    """Response containing results for multiple tasks.

    Returns a mapping of task_id -> TaskResultResponse for bulk operations.
    """

    results: dict[str, TaskResultResponse]


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

    Used by both Gateway (/health with agent info) and
    Worker (/health with running/agent info).
    """

    status: str
    # Worker fields
    running: bool | None = None
    agent_import_path: str | None = None
    agent_reuse: bool | None = None
