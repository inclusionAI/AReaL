# SPDX-License-Identifier: Apache-2.0

"""Optional CPU policy hooks; the native trainer owns every GPU operation."""

from pathlib import Path
from typing import Any, Literal, Protocol

from ..state.batch_queue import QueuedBatch


def validate_unbound_policy(runtime: Any) -> None:
    """Reject shared/closed helpers before a new trainer takes cleanup ownership."""
    if runtime is not None and (
        getattr(runtime, "owner", None) is not None or getattr(runtime, "closed", False)
    ):
        raise ValueError(
            "Policy runtime is already bound or closed; create a new instance"
        )


class ActorPolicyRuntime(Protocol):
    def bind(self, owner: Any) -> None: ...

    def decide(
        self, batch: QueuedBatch | None
    ) -> Literal["train", "keep", "release"]: ...

    def poll(self) -> None: ...

    def begin_update(self, batch: QueuedBatch, trajectories: list[dict]) -> None: ...

    def record_phase(self, phase: str, seconds: float) -> None: ...

    def checkpoint_state(self, path: Path) -> None: ...

    def published(self) -> None: ...

    def recover(self) -> None: ...

    def released(self, seconds: float, gpu_seconds: float) -> None: ...

    def close(self) -> None: ...
