# SPDX-License-Identifier: Apache-2.0

"""Fixed Actor allocation against absolute, exogenous group arrivals; CPU only."""

from __future__ import annotations

import math
from dataclasses import dataclass


def _seconds(value: float) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError(
            "Replay durations and timestamps must be finite and nonnegative"
        )


@dataclass(frozen=True)
class GroupArrival:
    group_id: str
    seconds: float
    samples: int

    def __post_init__(self):
        _seconds(self.seconds)
        if (
            not isinstance(self.group_id, str)
            or not self.group_id
            or type(self.samples) is not int
            or self.samples < 1
        ):
            raise ValueError("Arrivals need an identity and a positive sample count")


@dataclass(frozen=True)
class ActorPhase:
    name: str
    seconds: float | None
    evidence: str = "measured"

    def __post_init__(self):
        if not self.name or self.evidence not in {
            "measured",
            "calibrated",
            "assumed",
            "unsupported",
        }:
            raise ValueError("Actor phases need a name and explicit evidence")
        if (self.seconds is None) != (self.evidence == "unsupported"):
            raise ValueError("Unknown phase durations must remain unsupported")
        if self.seconds is not None:
            _seconds(self.seconds)


@dataclass(frozen=True)
class ActorReplayStep:
    group_ids: tuple[str, ...]
    phases: tuple[ActorPhase, ...]
    sequence_tokens: int
    effective_tokens: int
    before_seconds: float = 0
    after_seconds: float = 0

    def __post_init__(self):
        object.__setattr__(self, "group_ids", tuple(self.group_ids))
        object.__setattr__(self, "phases", tuple(self.phases))
        _seconds(self.before_seconds)
        _seconds(self.after_seconds)
        if not self.group_ids or len(set(self.group_ids)) != len(self.group_ids):
            raise ValueError("A replay batch requires distinct groups")
        if not self.phases or len({p.name for p in self.phases}) != len(self.phases):
            raise ValueError("Replay steps require uniquely named service phases")
        if any(
            type(x) is not int or x < 1
            for x in (self.sequence_tokens, self.effective_tokens)
        ):
            raise ValueError("Replay workload counts must be positive integers")
        if self.effective_tokens > self.sequence_tokens:
            raise ValueError("Effective tokens exceed sequence tokens")


def replay_actor(
    arrivals: tuple[GroupArrival, ...],
    steps: tuple[ActorReplayStep, ...],
    *,
    actor_gpus: int,
) -> dict:
    """Complete the same ordered batches, without moving any future arrival.

    Actor is already allocated at t=0 and remains allocated through the last
    phase. No startup, resize, release, Rollout GPU or training-quality model.
    Measured controller delays, when supplied, are relative durations, never
    absolute Actor start times or historical batch waits.
    """
    if type(actor_gpus) is not int or actor_gpus < 1 or not steps:
        raise ValueError("Replay requires an allocation and nonempty work")
    by_id = {a.group_id: a for a in arrivals}
    if len(by_id) != len(arrivals):
        raise ValueError("Duplicate arrival identity")
    consumed = [g for s in steps for g in s.group_ids]
    if len(set(consumed)) != len(consumed) or not set(consumed) <= by_id.keys():
        raise ValueError("Every consumed group needs one arrival and one consumption")
    unsupported = [
        i for i, s in enumerate(steps, 1) if any(p.seconds is None for p in s.phases)
    ]
    if unsupported:
        return {
            "status": "unsupported",
            "unsupported_steps": unsupported,
            "seconds": None,
            "actor_gpu_hours": None,
            "reason": "Full-window cost is unknown; incomplete work is not scored",
        }
    ordered = sorted(enumerate(arrivals), key=lambda x: (x[1].seconds, x[0]))
    now, cursor = 0.0, 0
    ready, events, intervals, rows = set(), [], [], []

    def advance(seconds, phase, step, evidence="measured"):
        nonlocal now, cursor
        end = now + seconds
        while cursor < len(ordered) and ordered[cursor][1].seconds <= end:
            arrival = ordered[cursor][1]
            ready.add(arrival.group_id)
            events.append(
                {
                    "time": arrival.seconds,
                    "kind": "group_ready",
                    "group_id": arrival.group_id,
                }
            )
            cursor += 1
        if seconds:
            intervals.append(
                {
                    "start": now,
                    "end": end,
                    "phase": phase,
                    "step": step,
                    "actor_gpus": actor_gpus,
                    "evidence": evidence,
                }
            )
        now = end

    for index, step in enumerate(steps, 1):
        advance(step.before_seconds, "controller", index)
        started = now
        available = max(by_id[g].seconds for g in step.group_ids)
        wait = max(0, available - now)
        advance(wait, "wait", index)
        if not set(step.group_ids) <= ready:
            raise RuntimeError("Actor cannot consume a group before arrival")
        queued = {g: now - by_id[g].seconds for g in step.group_ids}
        ready.difference_update(step.group_ids)
        events.append(
            {
                "time": now,
                "kind": "actor_start",
                "step": index,
                "groups": step.group_ids,
            }
        )
        actor_start = now
        for phase in step.phases:
            advance(phase.seconds, phase.name, index, phase.evidence)
        advance(step.after_seconds, "controller", index)
        events.append({"time": now, "kind": "update_complete", "step": index})
        rows.append(
            {
                "step": index,
                "start": started,
                "actor_start": actor_start,
                "end": now,
                "wait_seconds": wait,
                "queue_wait_seconds": queued,
            }
        )
    phase_seconds = {}
    for interval in intervals:
        phase = interval["phase"]
        phase_seconds[phase] = (
            phase_seconds.get(phase, 0) + interval["end"] - interval["start"]
        )
    return {
        "status": "complete",
        "seconds": now,
        "actor_gpu_hours": sum(
            (i["end"] - i["start"]) * i["actor_gpus"] for i in intervals
        )
        / 3600,
        "actor_gpus": actor_gpus,
        "updates": len(steps),
        "groups": len(consumed),
        "samples": sum(by_id[g].samples for g in consumed),
        "effective_tokens": sum(s.effective_tokens for s in steps),
        "sequence_tokens": sum(s.sequence_tokens for s in steps),
        "unused_arrived_groups": len(ready),
        "future_groups": len(arrivals) - cursor,
        "phase_seconds": phase_seconds,
        "steps": rows,
        "events": events,
        "intervals": intervals,
        "unsupported_steps": [],
    }


def steps_from_dict(
    rows: list[dict], phase_key="phases"
) -> tuple[ActorReplayStep, ...]:
    return tuple(
        ActorReplayStep(
            group_ids=tuple(r["group_ids"]),
            phases=tuple(ActorPhase(**p) for p in r[phase_key]),
            sequence_tokens=r["sequence_tokens"],
            effective_tokens=r["effective_tokens"],
            before_seconds=r.get("before_seconds", 0),
            after_seconds=r.get("after_seconds", 0),
        )
        for r in rows
    )
