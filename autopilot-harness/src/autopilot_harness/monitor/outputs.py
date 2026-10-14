"""Build Monitor's specification, configuration, observations, and history."""

import hashlib
import json
import math
from pathlib import Path
from typing import Any
from uuid import uuid4

from autopilot_harness.contracts.monitor import (
    CurrentConfiguration,
    Observation,
    TrainingSpecification,
)

from ._io import redacted, timestamp

# Training specification: explicit code, workflow, model, and resource context.


def build_specification(value: dict | None) -> TrainingSpecification:
    if value is not None and not isinstance(value, dict):
        raise ValueError("training_specification must be an object")
    value = value or {}
    missing = []
    for section, fields in {
        "runtime_code": ("uri", "version"),
        "workflow_code": ("uri", "version", "entrypoint"),
        "model": ("uri", "version"),
        "resources": ("gpu_type", "gpu_count", "memory_per_gpu_gib", "topology"),
    }.items():
        item = value.get(section, {})
        if not isinstance(item, dict):
            raise ValueError(f"training_specification.{section} must be an object")
        for key in fields:
            if item.get(key) is None or item.get(key) == "":
                missing.append(f"{section}.{key}")
    return TrainingSpecification(
        redacted(value), missing, "monitor_config:training_specification"
    )


# Current configuration: distinguish recorded values from application evidence.


def build_configuration(
    explicit: dict | None,
    recorded: dict | None,
    source_ref: str | None,
    segment_id: str,
    collected_at: float,
) -> CurrentConfiguration:
    if explicit is None and recorded is None:
        return CurrentConfiguration({}, "missing", None, None)
    status = "recorded"
    confirmation = None
    if explicit is not None:
        if not isinstance(explicit, dict):
            raise ValueError("current_configuration must be an object")
        if not isinstance(explicit.get("values"), dict):
            raise ValueError("current_configuration.values must be an object")
        recorded = explicit["values"]
        source_ref = explicit.get("source_ref", "monitor_config:current_configuration")
        status = explicit.get("status", "recorded")
        if status not in {"recorded", "confirmed"}:
            raise ValueError(
                "current_configuration.status must be recorded or confirmed"
            )
        confirmation = explicit.get("confirmation")
        if status == "confirmed":
            if (
                not isinstance(confirmation, dict)
                or not confirmation.get("event_id")
                or confirmation.get("segment_id") != segment_id
                or confirmation.get("applied_at") is None
            ):
                raise ValueError(
                    "confirmed configuration requires event_id, segment_id and applied_at"
                )
            if timestamp(confirmation["applied_at"]) > collected_at:
                raise ValueError("configuration confirmation cannot be in the future")
    values = redacted(recorded)
    digest = hashlib.sha256(
        json.dumps(values, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    return CurrentConfiguration(
        values, status, source_ref, "redacted-sha256:" + digest, redacted(confirmation)
    )


# Runtime observations: normalize samples without guessing alignment or filling zeros.


def numeric(value: Any) -> tuple[float | None, str | None]:
    if value is None:
        return None, "source_null"
    if isinstance(value, bool):
        return None, "not_numeric"
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None, "not_numeric"
    if not math.isfinite(result):
        return None, "non_finite"
    return result, None


def step_number(value: Any) -> int | None:
    result, _ = numeric(value)
    return (
        int(result)
        if result is not None and result >= 0 and result.is_integer()
        else None
    )


def sample(
    *,
    metric: str,
    value: Any,
    unit: str | None,
    entity: dict[str, str],
    observed_at: Any,
    available_at: float,
    timestamp_basis: str,
    window: dict,
    coverage: dict,
    source_ref: str,
    mode: str,
    logging_step: Any = None,
    train_step: Any = None,
) -> Observation:
    normalized, missing_reason = numeric(value)
    try:
        observed = timestamp(observed_at) if observed_at is not None else None
    except (TypeError, ValueError, OverflowError):
        observed = None
    return Observation(
        metric,
        normalized,
        unit,
        entity,
        observed,
        available_at,
        timestamp_basis,
        window,
        coverage,
        "assumed" if mode == "fixture" else "measured",
        source_ref,
        step_number(logging_step),
        step_number(train_step),
        missing_reason,
    )


# Execution history: append-only events; corrections are new events.


def make_event(
    run_id: str,
    segment_id: str,
    kind: str,
    recorded_at: float,
    payload: dict[str, Any],
    links: dict[str, str] | None = None,
) -> dict:
    return {
        "event_id": str(uuid4()),
        "run_id": run_id,
        "segment_id": segment_id,
        "kind": kind,
        "recorded_at": timestamp(recorded_at),
        "payload": redacted(payload),
        "links": links or {},
    }


class HistoryStore:
    def __init__(self, path: Path):
        self.path = path

    @staticmethod
    def validate(event: dict) -> None:
        for key in (
            "event_id",
            "run_id",
            "segment_id",
            "kind",
            "recorded_at",
            "payload",
        ):
            if key not in event:
                raise ValueError(f"history event missing {key}")
        timestamp(event["recorded_at"])
        if not isinstance(event["payload"], dict):
            raise ValueError("history payload must be an object")

    def append(self, event: dict) -> None:
        self.validate(event)
        existing = self.read() if self.path.exists() else []
        if any(item["event_id"] == event["event_id"] for item in existing):
            raise ValueError("duplicate history event_id")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(redacted(event), ensure_ascii=False, allow_nan=False) + "\n"
            )

    def read(
        self,
        *,
        run_id: str | None = None,
        segment_id: str | None = None,
        as_of: float | None = None,
    ) -> list[dict]:
        events = []
        seen = set()
        with self.path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                event = json.loads(line)
                self.validate(event)
                if event["event_id"] in seen:
                    raise ValueError("duplicate history event_id")
                seen.add(event["event_id"])
                if run_id is not None and event["run_id"] != run_id:
                    continue
                if segment_id is not None and event["segment_id"] != segment_id:
                    continue
                if as_of is not None and timestamp(event["recorded_at"]) > as_of:
                    continue
                events.append(redacted(event))
        return events
