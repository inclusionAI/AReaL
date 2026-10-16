# SPDX-License-Identifier: Apache-2.0

"""Task-local model routing identity, independent of group/data placement."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

TRAJECTORY_HEADER = "X-AReaL-Trajectory-Key"
VERSION_HEADER = "X-AReaL-Weight-Version"
_trajectory_key: ContextVar[str | None] = ContextVar(
    "model_trajectory_key", default=None
)


@contextmanager
def model_routing_context(trajectory_key: str) -> Iterator[None]:
    """Bind a trusted trajectory identity across async model calls."""
    if not trajectory_key:
        raise ValueError("Model routing requires a nonempty trajectory key")
    token = _trajectory_key.set(trajectory_key)
    try:
        yield
    finally:
        _trajectory_key.reset(token)


def get_model_routing_key() -> str:
    key = _trajectory_key.get()
    if key is None:
        raise RuntimeError("Model Router generation requires a trajectory context")
    return key
