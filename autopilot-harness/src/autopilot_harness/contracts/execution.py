"""Frozen, JSON-backed execution plans; mutable caller inputs never leak in."""

import hashlib
import json
from dataclasses import asdict, dataclass

APPLY_ADAPTER = "areal-v1-drained-restart-v1"


@dataclass(frozen=True)
class ApplyPlan:
    document_json: str

    @property
    def plan_id(self) -> str:
        return "sha256:" + hashlib.sha256(self.document_json.encode()).hexdigest()

    def to_dict(self) -> dict:
        # Every read is a copy, including nested configurations and evidence.
        return json.loads(self.document_json)


@dataclass(frozen=True)
class ApplyResult:
    status: str  # applied | unchanged
    plan_id: str
    baseline_id: str
    configuration_id: str
    source_segment: str
    segment_id: str
    event_id: str
    applied_at: float
    elapsed_seconds: float
    next_step: int | None = None
    snapshot_path: str | None = None
    # Wall duration includes work overlapped with training before the boundary.
    # Exposed delay needs a separate runtime timing observation.
    exposed_delay_seconds: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)
