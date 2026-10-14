"""One-shot collection into the four-part Monitor context."""

import json
import time
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from autopilot_harness import __version__
from autopilot_harness.contracts.monitor import MonitorSnapshot

from . import sources
from ._io import redacted, resolve_path, write_json
from .outputs import HistoryStore, build_configuration, build_specification, make_event


def collect(
    config: dict,
    *,
    root: Path = Path("."),
    wandb_api_factory=None,
    prometheus_transport=None,
) -> MonitorSnapshot:
    started = time.time()
    for key in ("run_id", "segment_id"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"{key} must be a non-empty string")
    if "wandb" not in config and "prometheus" not in config:
        raise ValueError("configure at least one source: wandb or prometheus")
    spec = build_specification(config.get("training_specification"))
    # Validate before any network work; a malformed local config is not a source outage.
    build_configuration(
        config.get("current_configuration"), None, None, config["segment_id"], started
    )
    for key, validate in (
        ("wandb", sources.validate_wandb),
        ("prometheus", sources.validate_prometheus),
    ):
        if key in config:
            if not isinstance(config[key], dict):
                raise ValueError(f"{key} must be an object")
            validate(config[key])
    prior = []
    if config.get("history_path"):
        prior = HistoryStore(resolve_path(config["history_path"], root)).read(
            run_id=config["run_id"], as_of=started
        )
    batches = []
    recorded, reference = None, None
    if "wandb" in config:
        batch = sources.collect_wandb(
            config["wandb"], root, api_factory=wandb_api_factory
        )
        batches.append(batch)
        recorded, reference = (
            batch.recorded_configuration,
            "wandb:" + batch.report.reference + "#config",
        )
    if "prometheus" in config:
        batches.append(
            sources.collect_prometheus(
                config["prometheus"], root, transport=prometheus_transport
            )
        )
    finished = time.time()
    observations = [
        observation for batch in batches for observation in batch.observations
    ]
    for observation in observations:
        # Conservative availability: all evidence is available when this capture completes.
        # An old metric timestamp is not evidence of historical online availability.
        observation.available_at = finished
    reports = [batch.report for batch in batches]
    snapshot_id = str(uuid4())
    context_event = make_event(
        config["run_id"],
        config["segment_id"],
        "monitor.captured",
        finished,
        {
            "snapshot_id": snapshot_id,
            "samples": len(observations),
            "sources": {report.source: report.status for report in reports},
        },
        {"snapshot_id": snapshot_id},
    )
    return MonitorSnapshot(
        "0.1",
        snapshot_id,
        config["run_id"],
        config["segment_id"],
        started,
        finished,
        spec,
        build_configuration(
            config.get("current_configuration"),
            recorded,
            reference,
            config["segment_id"],
            finished,
        ),
        {
            "samples": [asdict(item) for item in observations],
            "sources": [asdict(report) for report in reports],
            "availability_basis": "collection_completed_at",
            "alignment": "source timestamps retained; no automatic train-step join",
        },
        prior + [context_event],
    )


def save(snapshot: MonitorSnapshot, config: dict, output: Path) -> Path:
    """Each capture owns a new directory. Never replace an existing capture."""
    output.mkdir(parents=True, exist_ok=False)
    snapshots = output / "snapshots"
    snapshots.mkdir()
    write_json(snapshots / "monitor.json", snapshot.to_dict())
    with (output / "events.jsonl").open("x", encoding="utf-8") as stream:
        for event in snapshot.execution_history:
            stream.write(
                json.dumps(redacted(event), ensure_ascii=False, allow_nan=False) + "\n"
            )
    write_json(
        output / "manifest.json",
        {
            "schema_version": snapshot.schema_version,
            "harness_version": __version__,
            "run_id": snapshot.run_id,
            "segment_id": snapshot.segment_id,
            "snapshot_id": snapshot.snapshot_id,
            "collected_at": snapshot.collected_at,
            "sources": snapshot.runtime_observations["sources"],
            "input_config": config,
            "snapshot": "snapshots/monitor.json",
            "credential_fields_redacted": True,
        },
    )
    return snapshots / "monitor.json"
