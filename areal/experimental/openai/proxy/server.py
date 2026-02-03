from __future__ import annotations

import asyncio
import threading
import time
from typing import TYPE_CHECKING, Any

import torch
from pydantic import BaseModel

from areal.experimental.openai.cache import InteractionCache

if TYPE_CHECKING:
    from areal.experimental.openai.types import InteractionWithTokenLogpReward

# Session timeout for cleanup (1 hour)
SESSION_TIMEOUT_SECONDS = 3600


# =============================================================================
# Request/Response Models
# =============================================================================


class StartSessionRequest(BaseModel):
    """Request to start a new RL session."""

    task_id: str


class StartSessionResponse(BaseModel):
    """Response from start_session endpoint."""

    session_id: str


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
        self._completed_event = asyncio.Event()

        self._start_time = time.time()
        self._last_access_time = time.time()
        self._end_time = None
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
        self._completed = True
        self._end_time = time.time()
        self._completed_event.set()

    @property
    def completions(self):
        return self._completions

    async def wait_for_finish(self, timeout: float | None = None) -> bool:
        """Asynchronously wait for session to finish. Returns True if finished, False if timeout."""
        if timeout:
            try:
                await asyncio.wait_for(self._completed_event.wait(), timeout)
                return True
            except TimeoutError:
                return False
        await self._completed_event.wait()
        return True

    def export_interactions(
        self, discount: float, style: str
    ) -> dict[str, InteractionWithTokenLogpReward]:
        if len(self.completions) == 0:
            return {}
        self.completions.apply_reward_discount(turn_discount=discount)
        return self.completions.export_interactions(style=style)


# =============================================================================
# Serialization Helpers
# =============================================================================


def serialize_interactions(
    interactions: dict[str, InteractionWithTokenLogpReward],
) -> dict[str, Any]:
    """Serialize interactions for HTTP transport."""
    result = {}
    for key, interaction in interactions.items():
        tensor_dict = interaction.to_tensor_dict()
        result[key] = {
            "tensor_dict": {k: v.tolist() for k, v in tensor_dict.items()},
            "reward": interaction.reward,
            "interaction_id": interaction.interaction_id,
        }
    return result


def deserialize_interactions(
    data: dict[str, Any],
) -> dict[str, InteractionWithTokenLogpReward]:
    """Deserialize interactions from HTTP response."""
    from areal.experimental.openai.types import InteractionWithTokenLogpReward

    result = {}
    for key, item in data.items():
        tensor_dict = {k: torch.tensor(v) for k, v in item["tensor_dict"].items()}
        # Create a minimal InteractionWithTokenLogpReward with cached tensor dict
        interaction = InteractionWithTokenLogpReward()
        interaction._cache = tensor_dict
        interaction.reward = item["reward"]
        result[key] = interaction
    return result


# =============================================================================
# Path Constants (must match client_session.py expectations)
# =============================================================================

RL_START_SESSION_PATHNAME = "rl/start_session"
RL_END_SESSION_PATHNAME = "rl/end_session"
RL_SET_REWARD_PATHNAME = "rl/set_reward"
CHAT_COMPLETIONS_PATHNAME = "chat/completions"
RESPONSES_PATHNAME = "responses"
ANTHROPIC_MESSAGES_PATHNAME = "v1/messages"
