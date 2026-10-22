# SPDX-License-Identifier: Apache-2.0

"""Serialized control boundary and snapshots for native Elastic execution."""

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Protocol


class ControlBusy(RuntimeError):
    """The native lifecycle is executing another operation."""


@dataclass(frozen=True)
class ControlSnapshot:
    """Live applied/published progress and independent durable checkpoint progress.

    ``boundary_ready`` requires completed publication and no active update;
    it does not require a checkpoint at the applied version.
    """

    closed: bool
    boundary_ready: bool
    generation: int
    applied_version: int
    published_version: int
    boundary_id: str
    checkpoint_version: int = 0


class NativeControlPort(Protocol):
    def control_snapshot(self) -> ControlSnapshot: ...
    def control_operation(
        self, *, blocking: bool = False
    ) -> AbstractContextManager[ControlSnapshot]: ...
    def fence_admission(self) -> None: ...
