# SPDX-License-Identifier: Apache-2.0

"""Durable configuration application, separate from planning and weight updates.

The caller must hold its admission/execution gate for preflight, apply and recovery.
An executor owns role draining, resource allocation and verified restoration. This
module records intent before mutation and publishes only observed configurations.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from typing import Generic, Protocol, TypeVar

from areal.utils.atomic import atomic_json
from areal.utils.atomic import json_digest as digest

from .boundary import NativeControlPort

ConfigT = TypeVar("ConfigT")


def _digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class NativeConfig:
    """Full native execution hash plus explicit layouts and physical placement."""

    execution_id: str
    actor_backend: str
    rollout_backend: str
    actor_devices: tuple[int, ...]
    rollout_devices: tuple[int, ...]

    def __post_init__(self):
        if not re.fullmatch(r"[a-f0-9]{64}", self.execution_id):
            raise ValueError("Native configuration requires a complete execution hash")
        if (
            not self.actor_backend
            or not self.rollout_backend
            or not self.actor_devices
            or not self.rollout_devices
        ):
            raise ValueError("Native configuration requires both layouts and GPU pools")


@dataclass(frozen=True)
class ConfigRequest(Generic[ConfigT]):
    request_id: str
    target: ConfigT
    expected_generation: int
    expected_weights_version: int
    policy_bundle_id: str

    def __post_init__(self):
        if not isinstance(self.request_id, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", self.request_id
        ):
            raise ValueError("Invalid config request identity")
        if not is_dataclass(self.target) or isinstance(self.target, type):
            raise TypeError(
                "Target must be a complete execution configuration dataclass"
            )
        if any(
            type(n) is not int or n < 0
            for n in (self.expected_generation, self.expected_weights_version)
        ):
            raise ValueError("Expected generations/versions must be nonnegative")
        if not isinstance(self.policy_bundle_id, str) or not self.policy_bundle_id:
            raise ValueError("Scheduling policy provenance is required")


@dataclass(frozen=True)
class ConfigSnapshot(Generic[ConfigT]):
    config: ConfigT
    weights_version: int
    boundary_id: str
    committed: bool


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
            self.config, state.applied_version, state.boundary_id, state.boundary_ready
        )

    def decode_config(self, value: dict) -> NativeConfig:
        return NativeConfig(
            **{
                **value,
                "actor_devices": tuple(value["actor_devices"]),
                "rollout_devices": tuple(value["rollout_devices"]),
            }
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


class ConfigExecutor(Protocol[ConfigT]):
    """Implementations must verify all changed roles, not just assign fields."""

    @property
    def context_id(self) -> str: ...

    def observe(self) -> ConfigSnapshot[ConfigT]: ...

    def decode_config(self, value: dict) -> ConfigT:
        """Reconstruct this executor's schema without importing planning models."""
        ...

    def validate(self, target: ConfigT) -> None: ...

    def apply(self, target: ConfigT) -> None: ...

    def recover(self, source: ConfigSnapshot[ConfigT]) -> None: ...


class ConfigApplication(Generic[ConfigT]):
    """One application journal; partial requests recover to their source boundary.

    Config generations count committed logical changes, not training updates.
    Reusing an id returns its historical receipt, even after later transitions.
    Source/target capabilities and logical training semantics belong to the executor.
    """

    def __init__(self, executor: ConfigExecutor[ConfigT], root: Path):
        self.executor = executor
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _read(self) -> dict:
        path = self.root / "state.json"
        if not path.exists():
            current = self.executor.observe()
            if not current.committed:
                raise RuntimeError("Cannot initialize from uncommitted execution")
            atomic_json(
                path,
                {
                    "format_version": 1,
                    "context_id": self.executor.context_id,
                    "generation": 0,
                    "config": asdict(current.config),
                    "pending": None,
                    "receipts": {},
                },
            )
        state = json.loads(path.read_text())
        if (
            state["format_version"] != 1
            or state["context_id"] != self.executor.context_id
        ):
            raise ValueError("Config journal belongs to another execution context")
        return state

    def state(self) -> dict:
        with (self.root / "config.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return self._read()

    def _preflight(self, state: dict, request: ConfigRequest[ConfigT]):
        if state["pending"] is not None:
            raise RuntimeError("Recover the pending config application first")
        current = self.executor.observe()
        if (
            not current.committed
            or json.loads(json.dumps(asdict(current.config))) != state["config"]
        ):
            raise RuntimeError("Execution differs from the committed config journal")
        old = state["receipts"].get(request.request_id)
        if old is not None:
            if old["request"] != json.loads(json.dumps(asdict(request))):
                raise ValueError(
                    "Config request identity was reused with different input"
                )
            return old
        if (
            request.expected_generation != state["generation"]
            or request.expected_weights_version != current.weights_version
        ):
            raise ValueError("Stale config generation or training weight version")
        self.executor.validate(request.target)
        return None

    def preflight(self, request: ConfigRequest[ConfigT]) -> None:
        """Check before closing admission; apply repeats checks under the same gate."""
        with (self.root / "config.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._preflight(self._read(), request)

    def _verify(
        self, source: ConfigSnapshot[ConfigT], target: ConfigT
    ) -> ConfigSnapshot[ConfigT]:
        actual = self.executor.observe()
        if (
            not actual.committed
            or actual.config != target
            or actual.weights_version != source.weights_version
            or actual.boundary_id != source.boundary_id
        ):
            raise RuntimeError(
                "Config application changed progress or failed verification"
            )
        return actual

    def _finish(self, state, pending, actual, status, error=None):
        if status == "applied":
            state["generation"] += 1
        receipt = {
            "request": pending["request"],
            "status": status,
            "generation": state["generation"],
            "actual": json.loads(json.dumps(asdict(actual))),
            "seconds": max(0, time.time() - pending["started"]),
            "error": error,
        }
        state["config"] = asdict(actual.config)
        state["receipts"][pending["request"]["request_id"]] = receipt
        state["pending"] = None
        atomic_json(self.root / "state.json", state)
        return receipt

    def apply(self, request: ConfigRequest[ConfigT]) -> dict:
        with (self.root / "config.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            state = self._read()
            previous = self._preflight(state, request)
            if previous is not None:
                return previous
            source = self.executor.observe()
            pending = {
                "request": json.loads(json.dumps(asdict(request))),
                "source": json.loads(json.dumps(asdict(source))),
                "phase": "applying",
                "started": time.time(),
            }
            state["pending"] = pending
            atomic_json(self.root / "state.json", state)
            if request.target == source.config:
                return self._finish(state, pending, source, "unchanged")
            try:
                self.executor.apply(request.target)
                actual = self._verify(source, request.target)
            except Exception as error:
                pending["phase"] = "recovering"
                pending["error"] = f"{type(error).__name__}: {error}"
                atomic_json(self.root / "state.json", state)
                self.executor.recover(source)
                actual = self._verify(source, source.config)
                return self._finish(
                    state, pending, actual, "rolled_back", pending["error"]
                )
            return self._finish(state, pending, actual, "applied")

    def recover(self) -> dict | None:
        """Reconcile a partial request, or restore active config on a fresh trainer."""
        with (self.root / "config.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            state = self._read()
            pending = state["pending"]
            if pending is not None:
                row = pending["source"]
                source = ConfigSnapshot(
                    self.executor.decode_config(row["config"]),
                    row["weights_version"],
                    row["boundary_id"],
                    row["committed"],
                )
                self.executor.recover(source)
                actual = self._verify(source, source.config)
                return self._finish(
                    state, pending, actual, "rolled_back", "Interrupted application"
                )
            current = self.executor.observe()
            if not current.committed:
                raise RuntimeError(
                    "Recover training progress before config application"
                )
            active = self.executor.decode_config(state["config"])
            if current.config != active:
                source = replace(current, config=active)
                self.executor.recover(source)
                self._verify(source, active)
            return None
