"""Session lifecycle management for the data proxy.

Adapted from areal/experimental/openai/proxy/server.py — uses cache and
type definitions from areal.experimental.openai.
"""

from __future__ import annotations

import asyncio
import secrets
import threading
import time
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from areal.experimental.openai.cache import InteractionCache

if TYPE_CHECKING:
    from areal.experimental.openai.types import (
        InteractionWithTokenLogpReward,
    )

# Session timeout for cleanup (1 hour)
SESSION_TIMEOUT_SECONDS = 3600


# =============================================================================
# Request/Response Models
# =============================================================================


class StartSessionRequest(BaseModel):
    """Request to start a new RL session."""

    task_id: str
    api_key: str | None = None  # Reuse a previously-issued key (refresh)


class StartSessionResponse(BaseModel):
    """Response from start_session endpoint."""

    session_id: str
    api_key: str


class SetRewardRequest(BaseModel):
    """Request to set reward for an interaction."""

    interaction_id: str | None = None
    reward: float


class ExportTrajectoriesRequest(BaseModel):
    """Request to export trajectories for a session."""

    session_id: str
    discount: float = 1.0
    style: str = "individual"


class ExportTrajectoriesResponse(BaseModel):
    """Response containing serialized interactions."""

    interactions: dict[str, Any]


# =============================================================================
# Session Data
# =============================================================================


class SessionData:
    """Data associated with a single RL session."""

    def __init__(self, session_id: str):
        self.session_id = session_id

        self._completed = False
        self._completions = InteractionCache()
        self._completed_event = threading.Event()
        self._start_time = time.time()
        self._last_access_time = time.time()
        self._end_time: float | None = None
        self._lock = threading.Lock()

    def update_last_access(self):
        """Update the last access time for this session."""
        with self._lock:
            self._last_access_time = time.time()

    def is_stale(self, timeout_seconds: float = SESSION_TIMEOUT_SECONDS) -> bool:
        """Check if this session has been inactive for too long."""
        with self._lock:
            return time.time() - self._last_access_time > timeout_seconds

    def finish(self):
        with self._lock:
            self._completed = True
            self._end_time = time.time()
        self._completed_event.set()

    @property
    def is_completed(self) -> bool:
        """Whether this session has been completed via ``finish()``."""
        return self._completed

    @property
    def completions(self):
        return self._completions

    async def wait_for_finish(self, timeout: float | None = None) -> bool:
        loop = asyncio.get_running_loop()
        deadline = time.monotonic() + timeout if timeout else None
        while not self._completed_event.is_set():
            remaining = (deadline - time.monotonic()) if deadline else 1.0
            if deadline and remaining <= 0:
                return False
            poll = min(remaining, 1.0)  # Poll every 1s so cancellation works
            await loop.run_in_executor(None, self._completed_event.wait, poll)
        return True

    def export_interactions(
        self, discount: float, style: str
    ) -> dict[str, InteractionWithTokenLogpReward]:
        if len(self.completions) == 0:
            return {}
        self.completions.apply_reward_discount(turn_discount=discount)
        return self.completions.export_interactions(style=style)


# =============================================================================
# Session Store
# =============================================================================


class SessionStore:
    """Thread-safe store for session lifecycle management."""

    def __init__(self):
        self._sessions: dict[str, SessionData] = {}
        self._api_key_to_session: dict[str, str] = {}  # api_key -> session_id
        self._session_to_api_key: dict[str, str] = {}  # session_id -> api_key
        self._lock = threading.Lock()
        self._capacity: int = 0
        self._admin_api_key: str = "areal-admin-key"  # default admin key

    def set_capacity(self, n: int) -> None:
        with self._lock:
            self._capacity = n

    def set_admin_key(self, key: str) -> None:
        with self._lock:
            self._admin_api_key = key

    @property
    def admin_api_key(self) -> str:
        return self._admin_api_key

    def start_session(
        self, task_id: str, api_key: str | None = None
    ) -> tuple[str, str]:
        """Start a new session, returning (session_id, session_api_key)."""
        with self._lock:
            # Generate unique session ID
            idx = 0
            while f"{task_id}-{idx}" in self._sessions:
                idx += 1
            session_id = f"{task_id}-{idx}"

            # Resolve session API key
            if api_key:
                session_api_key = api_key
                # Clean up stale mapping if key was previously used
                existing_sid = self._api_key_to_session.get(session_api_key)
                if existing_sid is not None:
                    existing_session = self._sessions.get(existing_sid)
                    if (
                        existing_session is not None
                        and not existing_session.is_completed
                    ):
                        raise ValueError(
                            f"API key is already bound to active session {existing_sid}."
                        )
                    self._remove_api_keys_for_session(existing_sid)
            else:
                session_api_key = secrets.token_urlsafe(32)
                while (
                    session_api_key in self._api_key_to_session
                    or session_api_key == self._admin_api_key
                ):
                    session_api_key = secrets.token_urlsafe(32)

            self._sessions[session_id] = SessionData(session_id=session_id)
            self._api_key_to_session[session_api_key] = session_id
            self._session_to_api_key[session_id] = session_api_key

        return (session_id, session_api_key)

    def end_session(self, session_id: str) -> int:
        """End a session. Returns interaction_count."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session {session_id} not found")
            if session.is_completed:
                raise KeyError(f"Session {session_id} already ended")
            interaction_count = len(session.completions)
            session.finish()
        return interaction_count

    def get_session_by_api_key(self, api_key: str) -> SessionData | None:
        with self._lock:
            session_id = self._api_key_to_session.get(api_key)
            if session_id is None:
                return None
            return self._sessions.get(session_id)

    def get_session(self, session_id: str) -> SessionData | None:
        with self._lock:
            return self._sessions.get(session_id)

    def remove_session(self, session_id: str) -> None:
        """Remove session from store and clean up API key mapping."""
        with self._lock:
            self._sessions.pop(session_id, None)
            self._remove_api_keys_for_session(session_id)

    def _remove_api_keys_for_session(self, session_id: str) -> None:
        """Remove the API key mapping for the given session. Must be called with _lock held."""
        api_key = self._session_to_api_key.pop(session_id, None)
        if api_key:
            self._api_key_to_session.pop(api_key, None)

    def cleanup_stale(self, timeout_seconds: float = SESSION_TIMEOUT_SECONDS) -> None:
        with self._lock:
            stale = [
                sid
                for sid, sd in self._sessions.items()
                if sd.is_stale(timeout_seconds)
            ]
            for sid in stale:
                self._sessions.pop(sid, None)
                self._remove_api_keys_for_session(sid)

    @property
    def session_count(self) -> int:
        with self._lock:
            return len(self._sessions)
