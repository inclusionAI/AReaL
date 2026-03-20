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
# Serialization Helpers
# =============================================================================


def serialize_interactions(
    interactions: dict[str, InteractionWithTokenLogpReward],
    node_addr: str = "",
) -> dict[str, Any]:
    """Serialize interactions for HTTP transport.

    When ``node_addr`` is provided, tensors are stored in the local RTensor
    storage (on the data proxy process) and only shard metadata is serialized.
    This enables ``RTensor.localize()`` on the client to fetch tensors via
    HTTP GET ``/data/{shard_id}`` from the data proxy.

    When ``node_addr`` is empty (legacy mode), tensors are serialized as
    plain lists for backward compatibility.

    Parameters
    ----------
    interactions : dict[str, InteractionWithTokenLogpReward]
        Interactions to serialize
    node_addr : str
        Data proxy's serving address (host:port) for RTensor shard storage.
        If empty, falls back to plain tensor serialization.
    """
    from areal.infra.rpc.rtensor import get_backend

    result = {}
    for key, interaction in interactions.items():
        tensor_dict = interaction.to_tensor_dict()

        if node_addr:
            # Store tensors locally on the data proxy and serialize shard metadata
            shard_dict = {}
            shapes = {}
            dtypes = {}
            for k, v in tensor_dict.items():
                tensor = v.detach().cpu()
                shard_id = get_backend().store(tensor)
                shard_dict[k] = {
                    "shard_id": shard_id,
                    "node_addr": node_addr,
                }
                shapes[k] = list(tensor.shape)
                dtypes[k] = str(tensor.dtype)
            result[key] = {
                "shard_dict": shard_dict,
                "shapes": shapes,
                "dtypes": dtypes,
                "reward": interaction.reward,
                "interaction_id": interaction.interaction_id,
            }
        else:
            # Legacy mode: serialize tensors as plain lists
            result[key] = {
                "tensor_dict": {k: v.tolist() for k, v in tensor_dict.items()},
                "reward": interaction.reward,
                "interaction_id": interaction.interaction_id,
            }
    return result


def deserialize_interactions(
    data: dict[str, Any],
) -> dict[str, InteractionWithTokenLogpReward]:
    """Deserialize interactions from HTTP response.

    Supports two formats:

    1. **Shard metadata format** (``shard_dict`` key present): RTensors are
       reconstructed from shard metadata. The actual tensor data stays on the
       data proxy and will be fetched lazily via ``RTensor.localize()``.

    2. **Legacy format** (``tensor_dict`` key present): Tensors are
       reconstructed from plain lists and re-remotized locally. This path
       exists for backward compatibility with data proxies that don't
       support RTensor storage.
    """
    from areal.experimental.openai.types import (
        InteractionWithTokenLogpReward,
    )
    from areal.infra.rpc.rtensor import RTensor, TensorShardInfo

    result = {}
    for key, item in data.items():
        if "shard_dict" in item:
            # Shard metadata format: reconstruct RTensors from shard info
            tensor_dict = {}
            for k, shard_info in item["shard_dict"].items():
                shape = item["shapes"][k]
                dtype_str = item["dtypes"][k].replace("torch.", "")
                dtype = getattr(torch, dtype_str)
                shard = TensorShardInfo(
                    shard_id=shard_info["shard_id"],
                    node_addr=shard_info["node_addr"],
                )
                # Create RTensor with meta placeholder — data fetched on localize()
                tensor_dict[k] = RTensor(
                    shard=shard,
                    data=torch.empty(shape, dtype=dtype, device="meta"),
                )
        else:
            # Legacy format: reconstruct tensors and remotize locally
            from areal.utils.network import gethostip

            node_addr = gethostip()
            tensor_dict = {k: torch.tensor(v) for k, v in item["tensor_dict"].items()}
            tensor_dict = RTensor.remotize(tensor_dict, node_addr=node_addr)

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
GRANT_CAPACITY_PATHNAME = "grant_capacity"
EXPORT_TRAJECTORIES_PATHNAME = "export_trajectories"
INTERNAL_WAIT_FOR_SESSION_PATHNAME = "internal/wait_for_session"

# Shared default for admin API key — used by cli_args.py and workflow.py
# to avoid independent duplication.
DEFAULT_ADMIN_API_KEY = "areal-admin-key"


class WaitForSessionRequest(BaseModel):
    """Request from _OnlineAgent to register a worker and wait for a session."""

    worker_addr: str


class WaitForSessionResponse(BaseModel):
    """Response with completed session credentials."""

    session_api_key: str
    session_id: str
    worker_addr: str
