# SPDX-License-Identifier: Apache-2.0

"""Public native control boundary; fixed layouts remain the only capability."""

from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from typing import Protocol

from areal.utils.atomic import json_digest as digest

from .config import ConfigSnapshot, NativeConfig


class ControlBusy(RuntimeError):
    """The native lifecycle is executing another operation."""


@dataclass(frozen=True)
class ControlSnapshot:
    closed: bool
    boundary_ready: bool
    generation: int
    weights_version: int
    published_version: int
    boundary_id: str


class NativeControlPort(Protocol):
    def control_snapshot(self) -> ControlSnapshot: ...
    def control_operation(
        self, *, blocking: bool = False
    ) -> AbstractContextManager[ControlSnapshot]: ...
    def fence_admission(self) -> None: ...


class NativeConfigExecutor:
    """Observe and verify configuration; never create resources or replay training."""

    def __init__(
        self, control: NativeControlPort, config: NativeConfig, *, policy_context: str
    ):
        self.control, self.config = control, config
        self.context_id = digest(
            {"native": asdict(config), "policy_context": policy_context}
        )

    def observe(self):
        state = self.control.control_snapshot()
        return ConfigSnapshot(
            self.config, state.weights_version, state.boundary_id, state.boundary_ready
        )

    def validate(self, target):
        if target != self.config:
            raise ValueError(
                "Native Actor/Rollout layout and GPU pools are currently fixed"
            )

    def apply(self, target):
        self.validate(target)

    def recover(self, source):
        self.validate(source.config)
        if self.observe() != source:
            raise RuntimeError(
                "Native configuration recovery cannot replay training or change its cursor"
            )
