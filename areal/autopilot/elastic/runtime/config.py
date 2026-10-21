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
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Protocol

from areal.autopilot.contracts import TrainingConfig
from areal.autopilot.elastic.policy.basic.cost_model.rollout import RolloutConfig
from areal.autopilot.elastic.policy.basic.joint import JointConfig
from areal.autopilot.elastic.runtime.state.checkpoint import atomic_json
from areal.autopilot.elastic.runtime.training.trainer import QueuedTrainer


def _digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _config(value: dict) -> JointConfig:
    return JointConfig(
        TrainingConfig(**value["actor"]), RolloutConfig(**value["rollout"])
    )


@dataclass(frozen=True)
class ConfigRequest:
    request_id: str
    target: JointConfig
    expected_generation: int
    expected_weights_version: int
    policy_bundle_id: str

    def __post_init__(self):
        if not isinstance(self.request_id, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", self.request_id
        ):
            raise ValueError("Invalid config request identity")
        if not isinstance(self.target, JointConfig):
            raise TypeError("Target must be a complete JointConfig")
        if any(
            type(n) is not int or n < 0
            for n in (self.expected_generation, self.expected_weights_version)
        ):
            raise ValueError("Expected generations/versions must be nonnegative")
        if not isinstance(self.policy_bundle_id, str) or not self.policy_bundle_id:
            raise ValueError("Scheduling policy provenance is required")


@dataclass(frozen=True)
class ConfigSnapshot:
    config: JointConfig
    weights_version: int
    boundary_id: str
    committed: bool


class ConfigExecutor(Protocol):
    """Implementations must verify all changed roles, not just assign fields."""

    @property
    def context_id(self) -> str: ...

    def observe(self) -> ConfigSnapshot: ...

    def validate(self, target: JointConfig) -> None: ...

    def apply(self, target: JointConfig) -> None: ...

    def recover(self, source: ConfigSnapshot) -> None: ...


class ActorConfigExecutor:
    """Reuse real Actor resize/recovery, with caller-declared fixed Rollout identity.

    This adapter neither owns nor restarts inference. Rollout changes are rejected
    before admission closes. Recovery uses a committed checkpoint, or the same
    immutable base model and seeded initialization before the first update.
    """

    def __init__(self, trainer: QueuedTrainer, rollout: RolloutConfig):
        self.trainer, self.rollout = trainer, rollout
        self.measurements = []
        group = trainer.group
        self._context_id = _digest(
            {
                "adapter": "actor-config-v1",
                "compatibility": group.compatibility,
                "pool": list(getattr(group, "pool_uuids", group.pool_devices)),
                "rollout": asdict(rollout),
                "queue_id": trainer.queue.queue_id,
                "queue_root": str(trainer.queue.root.resolve()),
                "checkpoint_root": str(group.store.root.resolve()),
            }
        )

    @property
    def context_id(self) -> str:
        return self._context_id

    def observe(self) -> ConfigSnapshot:
        trainer = self.trainer
        if trainer.consumer is None:
            raise RuntimeError("Prepare training before config application")
        return ConfigSnapshot(
            JointConfig(trainer.training_config, self.rollout),
            trainer.version,
            _digest(
                {
                    "checkpoint": str(trainer.latest_checkpoint),
                    "cursor": trainer.consumer.cursor,
                    "samples": trainer.committed_samples,
                    "tokens": trainer.committed_tokens,
                }
            ),
            trainer.is_committed,
        )

    def validate(self, target: JointConfig) -> None:
        if target.rollout != self.rollout:
            raise ValueError("This executor does not support Rollout reconfiguration")
        self.trainer.validate_config(target.actor)
        if (
            target.actor != self.trainer.training_config
            and self.trainer.latest_checkpoint is None
            and self.trainer.version != 0
        ):
            raise ValueError("Actor reconfiguration requires a committed checkpoint")

    def apply(self, target: JointConfig) -> None:
        self.measurements = self.trainer.configure(target.actor)
        if self.trainer.last_reconfiguration_error is not None:
            raise RuntimeError(self.trainer.last_reconfiguration_error)
        if not self.trainer.is_active:
            started = time.monotonic()
            self.trainer.activate()
            self.measurements.append(
                ("start", target.actor, time.monotonic() - started)
            )
        if self.trainer.training_config != target.actor:
            raise RuntimeError("Actor did not activate the requested configuration")
        if not self.trainer.is_active:
            raise RuntimeError("Requested Actor configuration is not running")

    def recover(self, source: ConfigSnapshot) -> None:
        # An already verified rollback requires no second restart.
        if (
            self.trainer.consumer is not None
            and self.trainer.is_active
            and self.observe() == source
        ):
            return
        self.trainer.stop()  # Retains the consumer fence if cleanup is uncertain.
        self.trainer.group.configure(source.config.actor)
        self.trainer.prepare()
        if self.observe() != source:
            raise RuntimeError("Recovery checkpoint/cursor differs from config intent")
        self.trainer.activate()


class JointConfigExecutor:
    """Coordinate real roles on one local pool; Rollout uses the pool prefix.

    All clients must use the loop's whole-episode gate, including session removal.
    Model weights and training progress stay fixed during either role's restart.
    """

    def __init__(self, trainer, rollout, publisher):
        if publisher.service is not rollout:
            raise ValueError("Publisher must reference the managed Rollout group")
        base = trainer.group.compatibility
        if (
            not base.get("model")
            or Path(base["model"]).resolve() != rollout.model
            or base.get("model_revision") != rollout.model_revision
        ):
            raise ValueError(
                "Actor and Rollout must share the immutable base model and revision"
            )
        self.trainer, self.rollout, self.publisher = trainer, rollout, publisher
        self.actor = ActorConfigExecutor(trainer, rollout.config)
        self.measurements, self.role_events = [], []
        group = trainer.group
        self._context_id = _digest(
            {
                "adapter": "joint-config-v1",
                "compatibility": group.compatibility,
                "pool": list(getattr(group, "pool_uuids", group.pool_devices)),
                "queue_id": trainer.queue.queue_id,
                "queue_root": str(trainer.queue.root.resolve()),
                "checkpoint_root": str(group.store.root.resolve()),
                "rollout": rollout.context_id,
            }
        )

    @property
    def context_id(self):
        return self._context_id

    @property
    def max_train_gpus(self):
        return len(self.trainer.group.pool_devices) - self.rollout.config.world_size

    def placement(self, config):
        pool = self.trainer.group.pool_devices
        if config.world_size > len(pool):
            raise ValueError("Joint configuration exceeds the shared GPU pool")
        count = config.rollout.world_size
        return tuple(pool[count : count + config.actor.world_size]), tuple(pool[:count])

    def observe(self):
        actor = self.actor.observe()
        config = JointConfig(actor.config.actor, self.rollout.config)
        actor_devices, rollout_devices = self.placement(config)
        placed = tuple(self.rollout.devices) == rollout_devices and (
            not self.trainer.is_active
            or tuple(self.trainer.group.devices) == actor_devices
        )
        return replace(
            actor,
            config=config,
            committed=(
                actor.committed
                and placed
                and self.rollout.verified(actor.weights_version)
            ),
        )

    def validate(self, target):
        if (
            not getattr(self.trainer.group, "is_v2", False)
            or self.trainer.group.compatibility.get("elastic_dp") != "fsdp-pure-dp-v1"
        ):
            raise ValueError("Joint placement requires v2 FSDP pure-DP training")
        self.trainer.validate_config(target.actor)
        if not callable(getattr(self.trainer.group, "configure_placement", None)):
            raise ValueError("Actor executor cannot accept joint GPU placement")
        self.rollout.validate(target.rollout)
        self.placement(target)
        self.rollout.assert_drained()

    def _load_rollout(self):
        self.publisher.initialize(self.trainer.latest_checkpoint, admit=False)

    def prepare(self):
        """Prepare initial services before journal recovery, with admission closed."""
        config = JointConfig(self.trainer.training_config, self.rollout.config)
        self.validate(config)
        actor_devices, rollout_devices = self.placement(config)
        if self.trainer.is_active:
            raise RuntimeError("Prepare joint execution before activating training")
        self.trainer.group.configure_placement(list(actor_devices))
        if self.rollout.service is None:
            self.rollout.start(config.rollout, rollout_devices)
        self._load_rollout()

    def apply(self, target):
        self.measurements = []
        current = self.observe().config
        actor_devices, rollout_devices = self.placement(target)
        change_actor = current.actor != target.actor or (
            self.placement(current)[0] != actor_devices
        )
        change_rollout = (
            current.rollout != target.rollout
            or tuple(self.rollout.devices) != rollout_devices
        )
        self.rollout.quiesce()
        if change_actor and self.trainer.is_active:
            started = time.monotonic()
            self.trainer.suspend()
            self.measurements.append(
                ("release", current.actor, time.monotonic() - started)
            )
        if change_rollout:
            self.rollout.stop()
        if change_actor:
            self.trainer.group.configure(target.actor)
            self.trainer.group.configure_placement(list(actor_devices))
        if change_rollout:
            self.rollout.start(target.rollout, rollout_devices)
            self._load_rollout()
        if change_actor:
            started = time.monotonic()
            self.trainer.activate()
            phase = "restore" if self.trainer.latest_checkpoint is not None else "start"
            self.measurements.append((phase, target.actor, time.monotonic() - started))
        self.role_events.append(
            {
                "target": asdict(target),
                "actor_devices": list(actor_devices),
                "rollout_devices": list(rollout_devices),
                "rollout_owner": self.rollout.owner,
            }
        )

    def recover(self, source):
        # Retain the queue fence until both target roles have verified cleanup.
        self.trainer.group.stop()
        self.rollout.stop()
        self.trainer.stop()
        self.trainer.group.configure(source.config.actor)
        actor_devices, rollout_devices = self.placement(source.config)
        self.trainer.group.configure_placement(list(actor_devices))
        self.trainer.prepare()
        self.rollout.start(source.config.rollout, rollout_devices)
        self._load_rollout()
        self.trainer.activate()
        if self.observe() != source:
            raise RuntimeError(
                "Joint recovery differs from source checkpoint/cursor/config"
            )

    def resume(self):
        if not self.observe().committed:
            raise RuntimeError("Cannot resume unverified joint execution")
        self.rollout.resume()

    def close(self):
        self.trainer.group.stop()
        self.rollout.stop()


class ConfigApplication:
    """One application journal; partial requests recover to their source boundary.

    Config generations count committed logical changes, not training updates.
    Reusing an id returns its historical receipt, even after later transitions.
    Source/target capabilities and logical training semantics belong to the executor.
    """

    def __init__(self, executor: ConfigExecutor, root: Path):
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

    def _preflight(self, state: dict, request: ConfigRequest):
        if state["pending"] is not None:
            raise RuntimeError("Recover the pending config application first")
        current = self.executor.observe()
        if not current.committed or asdict(current.config) != state["config"]:
            raise RuntimeError("Execution differs from the committed config journal")
        old = state["receipts"].get(request.request_id)
        if old is not None:
            if old["request"] != asdict(request):
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

    def preflight(self, request: ConfigRequest) -> None:
        """Check before closing admission; apply repeats checks under the same gate."""
        with (self.root / "config.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._preflight(self._read(), request)

    def _verify(self, source: ConfigSnapshot, target: JointConfig) -> ConfigSnapshot:
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
            "actual": asdict(actual),
            "seconds": max(0, time.time() - pending["started"]),
            "error": error,
        }
        state["config"] = asdict(actual.config)
        state["receipts"][pending["request"]["request_id"]] = receipt
        state["pending"] = None
        atomic_json(self.root / "state.json", state)
        return receipt

    def apply(self, request: ConfigRequest) -> dict:
        with (self.root / "config.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            state = self._read()
            previous = self._preflight(state, request)
            if previous is not None:
                return previous
            source = self.executor.observe()
            pending = {
                "request": asdict(request),
                "source": asdict(source),
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
                    _config(row["config"]),
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
            active = _config(state["config"])
            if current.config != active:
                source = replace(current, config=active)
                self.executor.recover(source)
                self._verify(source, active)
            return None
