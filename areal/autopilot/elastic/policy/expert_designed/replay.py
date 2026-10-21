# SPDX-License-Identifier: Apache-2.0

"""Observed-window accounting, kept separate from counterfactual prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .core import code_revision, fingerprint, integer, nonnegative
from .scenario import Event, ExpertResult
from .simulator import AllocationLedger


@dataclass(frozen=True)
class ReplayInitial:
    time: float
    actor_devices: tuple[str, ...] = ()
    rollout_devices: tuple[str, ...] = ()
    accepted_total: int = 0
    rejected_total: int = 0
    running_groups: int = 0
    queued_groups: int = 0
    training_version: int = 0
    serving_version: int = 0
    provider_version: int = 0
    ready_ids: tuple[str, ...] | None = None
    running_ids: tuple[str, ...] | None = None
    actor_busy: bool = False
    complete: bool = True

    def __post_init__(self) -> None:
        nonnegative("time", self.time)
        for name in (
            "accepted_total",
            "rejected_total",
            "running_groups",
            "queued_groups",
            "training_version",
            "serving_version",
            "provider_version",
        ):
            integer(name, getattr(self, name))
        for name in ("actor_devices", "rollout_devices", "ready_ids", "running_ids"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, tuple(value))
                if len(set(value)) != len(value):
                    raise ValueError("Duplicate initial replay identities")
        if self.accepted_total < self.queued_groups:
            raise ValueError("Accepted counter is smaller than ready queue")
        if self.ready_ids is not None and len(self.ready_ids) != self.queued_groups:
            raise ValueError("Initial ready IDs do not match queue count")
        if (
            self.running_ids is not None
            and len(self.running_ids) != self.running_groups
        ):
            raise ValueError("Initial running IDs do not match running count")
        if set(self.ready_ids or ()) & set(self.running_ids or ()):
            raise ValueError("Initial ready/running groups overlap")


@dataclass(frozen=True)
class ReplayRecord:
    time: float
    kind: Literal[
        "acquire",
        "release",
        "group_submit",
        "group_accept",
        "group_reject",
        "group_cancel",
        "actor_start",
        "actor_complete",
        "publish",
        "provider_version",
        "admission_pause",
        "admission_resume",
    ]
    count: int = 1
    group_ids: tuple[str, ...] = ()
    role: Literal["actor", "rollout"] = "actor"
    devices: tuple[str, ...] = ()
    version: int | None = None
    successful: bool = True

    def __post_init__(self) -> None:
        nonnegative("time", self.time)
        integer("count", self.count, minimum=1)
        object.__setattr__(self, "group_ids", tuple(self.group_ids))
        object.__setattr__(self, "devices", tuple(self.devices))
        if self.version is not None:
            integer("version", self.version)
        if self.group_ids and (
            len(self.group_ids) != self.count or len(set(self.group_ids)) != self.count
        ):
            raise ValueError("Replay group IDs must match count and be unique")


def replay(
    initial: ReplayInitial,
    records: tuple[ReplayRecord, ...],
    *,
    stop_at: float,
    source_id: str,
    clock_aligned: bool = True,
) -> ExpertResult:
    """Replay explicit records in source order, including ties; never invent arrivals."""
    nonnegative("stop_at", stop_at)
    if not source_id or stop_at < initial.time:
        raise ValueError("Replay needs a source and a valid observation interval")
    snapshot = fingerprint(code_revision(), initial, records, stop_at, source_id)
    ledger = AllocationLedger(initial.time)
    ledger.change("acquire", "actor", initial.actor_devices)
    ledger.change("acquire", "rollout", initial.rollout_devices)
    accepted, rejected = initial.accepted_total, initial.rejected_total
    running, queued = initial.running_groups, initial.queued_groups
    training, serving, provider = (
        initial.training_version,
        initial.serving_version,
        initial.provider_version,
    )
    busy = initial.actor_busy
    busy_devices = ledger.devices("actor") if busy else set()
    if busy and not busy_devices:
        raise ValueError("Active replay update requires actor resources")
    attempted = completed = cancelled = 0
    ready_ids, running_ids = (
        set(initial.ready_ids or ()),
        set(initial.running_ids or ()),
    )
    identities_known = (initial.ready_ids is not None or not queued) and (
        initial.running_ids is not None or not running
    )
    wait_known = not queued
    accepted_at = {}
    queue_wait = 0.0
    events = []
    if not initial.complete or not clock_aligned:
        return ExpertResult(
            "unsupported",
            "Replay requires complete initial counters/allocations and aligned clocks",
            0,
            0,
            0,
            0,
            0,
            accepted,
            rejected,
            0,
            queued,
            running,
            None,
            (),
            "unsupported",
            snapshot,
            tuple(sorted(ledger.devices("actor") | ledger.devices("rollout"))),
        )
    for row in records:
        if row.time > stop_at:
            raise ValueError("Record extends beyond the declared observation interval")
        ledger.advance(row.time)
        ids = set(row.group_ids)
        if row.kind.startswith("group_") or row.kind == "actor_start":
            if not ids:
                identities_known = wait_known = False
        if row.kind in {"acquire", "release"}:
            if not row.devices:
                raise ValueError("Resource events need device identities")
            ledger.change(row.kind, row.role, row.devices)
            if busy and not busy_devices <= ledger.devices("actor"):
                raise ValueError(
                    "Cannot release active actor resources during an update"
                )
        elif row.kind == "group_submit":
            if identities_known and (ids & (ready_ids | running_ids)):
                raise ValueError("Group submitted twice")
            running += row.count
            running_ids.update(ids)
        elif row.kind in {"group_accept", "group_reject", "group_cancel"}:
            if running < row.count or (identities_known and not ids <= running_ids):
                raise ValueError("Group completion without matching submission")
            running -= row.count
            running_ids.difference_update(ids)
            if row.kind == "group_accept":
                accepted += row.count
                queued += row.count
                ready_ids.update(ids)
                accepted_at.update(dict.fromkeys(ids, row.time))
            elif row.kind == "group_reject":
                rejected += row.count
            else:
                cancelled += row.count
        elif row.kind == "actor_start":
            if (
                busy
                or not ledger.devices("actor")
                or queued < row.count
                or (identities_known and not ids <= ready_ids)
            ):
                raise ValueError("Actor start violates queue/resource state")
            queued -= row.count
            ready_ids.difference_update(ids)
            if wait_known:
                queue_wait += sum(row.time - accepted_at[gid] for gid in ids)
            busy = True
            busy_devices = ledger.devices("actor")
        elif row.kind == "actor_complete":
            if not busy:
                raise ValueError("Actor completion without an active update")
            busy = False
            attempted += 1
            completed += int(row.successful)
            training = provider + 1 if row.version is None else row.version
            if training <= provider:
                raise ValueError("Actor completion must produce a newer version tag")
        elif row.kind == "publish":
            if busy or row.version != training or not row.successful:
                raise ValueError(
                    "Replay publish must confirm the pending training version"
                )
            serving = row.version
        elif row.kind == "provider_version":
            if row.version != serving or row.version < provider:
                raise ValueError("Provider cannot advance ahead of published weights")
            provider = row.version
        elif row.kind not in {"admission_pause", "admission_resume"}:
            raise ValueError(f"Unknown replay event: {row.kind}")
        events.append(
            Event(
                row.time,
                row.kind,
                details=(
                    ("count", row.count),
                    ("group_ids", row.group_ids),
                    ("accepted_total", accepted),
                    ("queued", queued),
                    ("provider_version", provider),
                    ("serving_version", serving),
                ),
            )
        )
    ledger.advance(stop_at)
    notes = (
        "Observed-window replay only; not rollout calibration or a counterfactual prediction",
        "Exact group membership checked"
        if identities_known
        else "Aggregate counts only; task membership cannot be verified",
        "Terminal state is retained; costs outside the supplied observation window are unknown",
    )
    return ExpertResult(
        "complete",
        "Observed interval accounted",
        stop_at - initial.time,
        ledger.seconds["actor"],
        ledger.seconds["rollout"],
        completed,
        attempted,
        accepted,
        rejected,
        cancelled,
        queued,
        running,
        queue_wait if wait_known else None,
        tuple(events),
        "measured",
        snapshot,
        tuple(sorted(ledger.devices("actor") | ledger.devices("rollout"))),
        notes,
    )
