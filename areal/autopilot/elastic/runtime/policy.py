# SPDX-License-Identifier: Apache-2.0

"""Immutable, accepted lifecycle components and durable scheduling-policy activation.

Bundles contain trusted Python, not a sandbox. Lifecycle optimizer acceptance can
bind a separately accepted simulator. Serial joint sources use a separate bundle
scope; finite prediction tables are never installed as executable models.
"""

from __future__ import annotations

import fcntl
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from areal.autopilot.contracts import BatchWorkload, HistoryWindow, Measurement
from areal.autopilot.elastic.policy.basic.policy import BasicPolicy, IdleTimeoutPolicy
from areal.autopilot.elastic.runtime.state.checkpoint import atomic_json
from areal.autopilot.revision.contracts import (
    BASELINE_SOURCE,
    deployment_context,
    digest,
    lifecycle_profile_digest,
    source_digest,
)
from areal.autopilot.revision.phase import (
    SCOPE,
    build_phase_simulator,
    verify_phase_reference,
)


def _evidence(store):
    active = store.active()
    if active is None:
        raise ValueError("No accepted lifecycle revision is available")
    path, pointer = active
    return {
        "context": json.loads((store.root / "context.json").read_text()),
        "report": json.loads((path.parent / "evaluation.json").read_text()),
        "acceptance": json.loads((path.parent / "acceptance.json").read_text()),
        "source_sha256": pointer["source_sha256"],
    }, path.read_text()


def export_accepted_policy(
    root: Path,
    evaluator,
    destination: Path,
    *,
    idle_seconds=30,
    model_revision_root: Path | None = None,
    model_verification_root: Path | None = None,
    parent_bundle: Path | None = None,
    optimizer_verification_root: Path | None = None,
):
    """Freeze accepted source, fitted phase profiles, settings and evaluation identity."""
    from areal.autopilot.revision.loop import RevisionStore

    if evaluator.target not in {"simulator", "optimizer"}:
        raise ValueError("Only executable lifecycle revisions can be exported")
    if evaluator.context.get("model_scope") != SCOPE:
        raise ValueError("Online policy bundles require fixed phase-model accounting")
    if evaluator.target == "simulator" and optimizer_verification_root is None:
        raise ValueError("Every model bundle requires optimizer revalidation")
    store = RevisionStore(root)
    evidence, source = _evidence(store)
    context = evidence["context"]
    if (
        context["benchmark_sha256"] != evaluator.benchmark_sha256
        or context["target"] != evaluator.target
        or context["deployment_context"] != evaluator.context
        or evaluator._code_fingerprint() != evaluator._code
    ):
        raise ValueError("Accepted revision belongs to another evaluation context")
    IdleTimeoutPolicy(idle_seconds)
    sources = dict(BASELINE_SOURCE)
    sources[evaluator.target] = source
    parent = None if parent_bundle is None else json.loads(parent_bundle.read_text())
    if parent is not None:
        validate_bundle(parent)
    model_evidence = None
    model_verification = None
    optimizer_evidence = None
    phase_reference = (
        {
            "cases": json.loads(evaluator._public_json),
            "prediction_sha256": evidence["report"]["metrics"]["prediction_sha256"],
        }
        if evaluator.target == "simulator"
        else evaluator.phase_reference
    )
    if evaluator.target == "optimizer":
        sources["simulator"] = evaluator.reference_simulator_source
        if sources["simulator"] != BASELINE_SOURCE["simulator"]:
            if model_revision_root is not None:
                model_evidence, model_source = _evidence(
                    RevisionStore(model_revision_root)
                )
            elif parent is not None:
                model_source = parent["components"]["simulator"]["source"]
                model_evidence = (
                    parent["evaluation"]
                    if parent["accepted_component"] == "simulator"
                    else parent["model_evaluation"]
                )
            else:
                raise ValueError("A revised simulator requires its model acceptance")
            if model_source != sources["simulator"]:
                raise ValueError("Optimizer was evaluated against another simulator")
            if model_verification_root is not None:
                verified = Path(model_verification_root)
                if (verified / "initial.py").read_text() != model_source:
                    raise ValueError("Model revalidation used another initial source")
                model_verification = {
                    "context": json.loads((verified / "context.json").read_text()),
                    "report": json.loads((verified / "baseline.json").read_text()),
                }
    elif optimizer_verification_root is not None:
        verified = Path(optimizer_verification_root)
        optimizer_evidence = {
            "context": json.loads((verified / "context.json").read_text()),
            "report": json.loads((verified / "baseline.json").read_text()),
        }
        sources["optimizer"] = (verified / "initial.py").read_text()
        if sources["optimizer"] != BASELINE_SOURCE["optimizer"] and (
            parent is None
            or parent["components"]["optimizer"]["source"] != sources["optimizer"]
        ):
            raise ValueError("Only the current optimizer can be revalidated unchanged")
    payload = {
        "format_version": 1,
        "kind": "lifecycle-policy-v1",
        "profile": evaluator.calibration,
        "settings": evaluator.settings,
        "idle_seconds": idle_seconds,
        "components": {
            name: {"source": sources[name], "sha256": source_digest(sources[name])}
            for name in ("simulator", "optimizer")
        },
        "accepted_component": evaluator.target,
        "evaluation": evidence,
        "model_evaluation": model_evidence,
        "model_verification": model_verification,
        "optimizer_verification": optimizer_evidence,
        "parent_bundle_id": None if parent is None else parent["bundle_id"],
        "protected_files": evaluator._code,
        "phase_reference": phase_reference,
    }
    value = {**payload, "bundle_id": digest(payload)}
    validate_bundle(value)
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / f"{value['bundle_id']}.json"
    if output.exists():
        if json.loads(output.read_text()) != value:
            raise ValueError("Immutable policy bundle changed")
    else:
        atomic_json(output, value)
    return output


def validate_bundle(value: dict) -> None:
    """Check content and receipt consistency before executing any factory code."""
    if value.get("kind") == "serial-joint-policy-v1":
        from .serial_policy import validate_serial_bundle

        return validate_serial_bundle(value)
    payload = {k: v for k, v in value.items() if k != "bundle_id"}
    if (
        value.get("format_version") != 1
        or value.get("kind") != "lifecycle-policy-v1"
        or value.get("bundle_id") != digest(payload)
        or value.get("accepted_component") not in {"simulator", "optimizer"}
    ):
        raise ValueError("Unsupported or changed policy bundle")
    target = value["accepted_component"]
    evidence = value["evaluation"]
    report, acceptance, context = (
        evidence["report"],
        evidence["acceptance"],
        evidence["context"],
    )
    for name in ("simulator", "optimizer"):
        component = value["components"][name]
        if source_digest(component["source"]) != component["sha256"]:
            raise ValueError("Policy component source changed")
        expected = (
            context["deployment_context"].get(
                "simulator_source", BASELINE_SOURCE["simulator"]
            )
            if target == "optimizer" and name == "simulator"
            else BASELINE_SOURCE[name]
        )
        if (
            target == "simulator"
            and name == "optimizer"
            and value.get("optimizer_verification")
        ):
            expected = component["source"]
        if name != target and component["source"] != expected:
            raise ValueError("The fixed policy counterpart differs from acceptance")
    component_id = value["components"][target]["sha256"]
    if (
        acceptance.get("accepted") is not True
        or report.get("valid") is not True
        or context["target"] != target
        or report["target"] != target
        or evidence["source_sha256"] != component_id
        or report["candidate_sha256"] != component_id
        or acceptance["candidate_sha256"] != component_id
        or report["benchmark_sha256"] != context["benchmark_sha256"]
        or acceptance["benchmark_sha256"] != context["benchmark_sha256"]
    ):
        raise ValueError("Policy bundle has no matching acceptance")
    expected_context = deployment_context(value["profile"], value["settings"])
    expected_context["profile_sha256"] = lifecycle_profile_digest(value["profile"])
    phase_reference = value.get("phase_reference")
    if (
        not phase_reference
        or not phase_reference.get("cases")
        or any(
            "phase" not in case or "observed" in case
            for case in phase_reference["cases"]
        )
    ):
        raise ValueError("Policy bundle is missing public phase probes")
    expected_context.update(
        model_scope=SCOPE, phase_cases_sha256=digest(phase_reference["cases"])
    )
    if target == "optimizer":
        expected_context["phase_prediction_sha256"] = phase_reference[
            "prediction_sha256"
        ]
    elif (
        evidence["report"]["metrics"]["prediction_sha256"]
        != phase_reference["prediction_sha256"]
    ):
        raise ValueError("Policy phase probes do not match accepted predictions")
    if "simulator_source" in context["deployment_context"]:
        expected_context["simulator_source"] = value["components"]["simulator"][
            "source"
        ]
    if context["deployment_context"] != expected_context:
        raise ValueError("Policy profile/settings differ from acceptance")
    if (
        target == "optimizer"
        and value["components"]["simulator"]["source"] != BASELINE_SOURCE["simulator"]
    ):
        model = value.get("model_evaluation")
        if not model:
            raise ValueError("Policy is missing simulator acceptance")
        model_id = value["components"]["simulator"]["sha256"]
        current_model_context = {
            k: v
            for k, v in expected_context.items()
            if k not in {"simulator_source", "phase_prediction_sha256"}
        }
        revalidation = value.get("model_verification")
        accepted_context = model["context"]["deployment_context"]
        if (
            model["context"]["target"] != "simulator"
            or {k: v for k, v in accepted_context.items() if k != "phase_cases_sha256"}
            != {
                k: v
                for k, v in current_model_context.items()
                if k != "phase_cases_sha256"
            }
            or model["source_sha256"] != model_id
            or model["report"]["candidate_sha256"] != model_id
            or model["acceptance"]["candidate_sha256"] != model_id
            or model["report"].get("valid") is not True
            or model["acceptance"].get("accepted") is not True
            or model["report"]["benchmark_sha256"]
            != model["context"]["benchmark_sha256"]
            or model["acceptance"]["benchmark_sha256"]
            != model["context"]["benchmark_sha256"]
        ):
            raise ValueError("Policy simulator acceptance does not match")
        if revalidation is None:
            if (
                accepted_context != current_model_context
                or model["report"]["metrics"]["prediction_sha256"]
                != phase_reference["prediction_sha256"]
            ):
                raise ValueError("Simulator needs revalidation on the current window")
        else:
            checked, result = revalidation["context"], revalidation["report"]
            if (
                checked["target"] != "simulator"
                or checked["deployment_context"] != current_model_context
                or result.get("valid") is not True
                or result["target"] != "simulator"
                or result["candidate_sha256"] != model_id
                or result["benchmark_sha256"] != checked["benchmark_sha256"]
                or result["metrics"]["prediction_sha256"]
                != phase_reference["prediction_sha256"]
            ):
                raise ValueError("Current-window model revalidation does not match")
    verification = value.get("optimizer_verification")
    if target == "simulator" and verification is None:
        raise ValueError("Model bundle is missing optimizer revalidation")
    if verification is not None:
        checked = verification["context"]
        result = verification["report"]
        if (
            target != "simulator"
            or checked["target"] != "optimizer"
            or checked["deployment_context"]
            != {
                **expected_context,
                "simulator_source": value["components"]["simulator"]["source"],
                "phase_prediction_sha256": phase_reference["prediction_sha256"],
            }
            or result.get("valid") is not True
            or result["target"] != "optimizer"
            or result["candidate_sha256"] != value["components"]["optimizer"]["sha256"]
            or result["benchmark_sha256"] != checked["benchmark_sha256"]
        ):
            raise ValueError("Unchanged optimizer was not revalidated on this model")
    base = Path(__file__).resolve().parents[2]
    if not value["protected_files"]:
        raise ValueError("Policy bundle is missing its runtime code identity")
    for name, expected in value["protected_files"].items():
        path = (base / name).resolve()
        if (
            not path.is_relative_to(base)
            or not path.is_file()
            or source_digest(path.read_text()) != expected
        ):
            raise ValueError("Policy evaluation/runtime code changed")
    IdleTimeoutPolicy(value["idle_seconds"])


@dataclass(frozen=True)
class PolicyRequest:
    request_id: str
    bundle_id: str
    expected_generation: int
    expected_weights_version: int

    def __post_init__(self):
        if not isinstance(self.request_id, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", self.request_id
        ):
            raise ValueError("Invalid policy request identity")
        if not isinstance(self.bundle_id, str) or not re.fullmatch(
            r"[a-f0-9]{64}", self.bundle_id
        ):
            raise ValueError("Policy bundle identity must be a SHA256")
        if any(
            type(n) is not int or n < 0
            for n in (self.expected_generation, self.expected_weights_version)
        ):
            raise ValueError("Expected generations/versions must be nonnegative")


class PolicyBoundaryBusy(RuntimeError):
    """A valid policy application must wait for the next planning boundary."""


class PolicyApplication:
    """A loop owns admission; this store owns the immutable active-policy pointer."""

    def __init__(
        self,
        root: Path,
        *,
        compatibility: dict,
        execution_context: dict,
        initial_bundle_id: str = "constructor-policy",
    ):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.compatibility = json.loads(json.dumps(compatibility, allow_nan=False))
        self.execution_context = json.loads(
            json.dumps(execution_context, allow_nan=False)
        )
        if (
            not initial_bundle_id
            or not all(
                self.execution_context.get(k) for k in ("queue_id", "checkpoint_root")
            )
            or (
                self.execution_context.get("scope") == "serial-joint-timing-v1"
                and (
                    not self.execution_context.get("joint_policy_identity")
                    or type(self.execution_context.get("max_total_gpus")) is not int
                    or self.execution_context["max_total_gpus"] < 1
                    or not self.execution_context.get("joint_executor")
                )
                or self.execution_context.get("scope") != "serial-joint-timing-v1"
                and (
                    type(self.execution_context.get("inference_gpus")) is not int
                    or self.execution_context["inference_gpus"] < 1
                )
            )
        ):
            raise ValueError(
                "Policy activation requires queue, checkpoint and compatible execution identities"
            )
        self.initial_bundle_id = initial_bundle_id
        self.context = digest(
            {
                "compatibility": self.compatibility,
                "initial": initial_bundle_id,
                "execution": self.execution_context,
            }
        )
        with self._lock():
            if not (self.root / "active.json").exists():
                atomic_json(
                    self.root / "active.json",
                    {
                        "format_version": 1,
                        "context": self.context,
                        "generation": 0,
                        "bundle_id": initial_bundle_id,
                        "receipts": {},
                    },
                )
            self.state()

    def _lock(self):
        lock = (self.root / "policy.lock").open("a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            lock.close()
            raise
        return lock

    def state(self):
        value = json.loads((self.root / "active.json").read_text())
        if value["format_version"] != 1 or value["context"] != self.context:
            raise ValueError("Policy store belongs to another execution context")
        return value

    def preflight(self, request: PolicyRequest, version: int):
        state = self.state()
        receipt = state["receipts"].get(request.request_id)
        if receipt is not None:
            if receipt["request"] != asdict(request):
                raise ValueError("Policy request ID was reused")
            return receipt
        if (
            request.expected_generation != state["generation"]
            or request.expected_weights_version != version
        ):
            raise ValueError("Stale policy generation or weights version")
        return None

    def prepare(self, bundle: Path):
        from areal.autopilot.revision.execution import load_source

        value = json.loads(bundle.read_text())
        validate_bundle(value)
        joint = self.execution_context.get("scope") == "serial-joint-timing-v1"
        if joint != (value["kind"] == "serial-joint-policy-v1"):
            raise ValueError("Policy bundle has a different execution scope")
        if value["profile"].get("compatibility") != self.compatibility:
            raise ValueError("Policy bundle does not match the training contract")
        if joint:
            from .serial_policy import serial_policy_identity

            if (
                serial_policy_identity(
                    value["profile"],
                    value["settings"],
                    horizon=value["horizon"],
                    idle_seconds=value["idle_seconds"],
                )
                != self.execution_context["joint_policy_identity"]
            ):
                raise ValueError("Joint policy changes its immutable baseline contract")
        elif (
            value["profile"].get("inference_gpus", 1)
            != self.execution_context["inference_gpus"]
        ):
            raise ValueError("Policy bundle changes fixed Rollout residency")
        directory = self.root / "bundles" / value["bundle_id"]
        directory.mkdir(parents=True, exist_ok=True)
        copied = directory / "bundle.json"
        if copied.exists() and json.loads(copied.read_text()) != value:
            raise ValueError("Stored policy bundle changed")
        if not copied.exists():
            atomic_json(copied, value)
        if joint:
            from .serial_policy import build_serial_policy

            policy = build_serial_policy(value)
        else:
            simulator = build_phase_simulator(
                value["components"]["simulator"]["source"], value["profile"]
            )
            verify_phase_reference(simulator, value["phase_reference"])
            optimizer = load_source(value["components"]["optimizer"]["source"])
            policy = BasicPolicy(
                simulator,
                optimizer=optimizer.build_optimizer(dict(value["settings"])),
                fallback=IdleTimeoutPolicy(value["idle_seconds"]),
            )
            if not callable(
                getattr(policy.simulator, "evaluate", None)
            ) or not callable(getattr(policy.optimizer, "optimize", None)):
                raise ValueError("Policy factories do not implement their contracts")
        validate_bundle(value)
        return PreparedPolicy(json.dumps(value, sort_keys=True), self.context, policy)

    def commit(
        self,
        request: PolicyRequest,
        version: int,
        prepared,
        history,
        *,
        joint_state=None,
    ):
        with self._lock():
            receipt = self.preflight(request, version)
            if receipt is not None:
                return receipt
            state = self.state()
            if (
                prepared.context_id != self.context
                or prepared.bundle_id != request.bundle_id
                or not prepared.probed
            ):
                raise ValueError("Prepared policy belongs to another request/context")
            bundle = self.root / "bundles" / request.bundle_id / "bundle.json"
            value = json.loads(bundle.read_text())
            validate_bundle(value)
            if value["bundle_id"] != request.bundle_id or value != json.loads(
                prepared.manifest_json
            ):
                raise ValueError("Prepared policy bundle changed")
            if (
                value.get("parent_bundle_id") is not None
                and value["parent_bundle_id"] != state["bundle_id"]
            ):
                raise ValueError("Policy bundle was revised from another active parent")
            receipt = {
                "request": asdict(request),
                "parent_bundle_id": state["bundle_id"],
                "generation": state["generation"] + 1,
                "weights_version": version,
                "status": "applied",
            }
            atomic_json(
                self.root / "active.json",
                {
                    **state,
                    "generation": receipt["generation"],
                    "bundle_id": request.bundle_id,
                    "receipts": {**state["receipts"], request.request_id: receipt},
                    "history": asdict(history.snapshot()),
                    "joint_state": joint_state,
                },
            )
            return receipt

    def save_history(self, history: HistoryWindow, version: int, *, joint_state=None):
        with self._lock():
            state = self.state()
            atomic_json(
                self.root / "history.json",
                {
                    "context": self.context,
                    "generation": state["generation"],
                    "bundle_id": state["bundle_id"],
                    "weights_version": version,
                    "history": asdict(history.snapshot()),
                    "joint_state": joint_state,
                },
            )

    def history(self, version: int):
        state = self.state()
        if any(
            receipt["weights_version"] > version
            for receipt in state["receipts"].values()
        ):
            raise ValueError("Policy activation is newer than the recovered checkpoint")
        snapshot = state.get("history", {})
        path = self.root / "history.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["context"] != self.context or saved["weights_version"] > version:
                raise ValueError(
                    "Runtime history belongs to another committed execution"
                )
            if (
                saved["generation"] == state["generation"]
                and saved["bundle_id"] == state["bundle_id"]
            ):
                snapshot = saved["history"]
        result = HistoryWindow(
            max(
                32,
                len(snapshot.get("workloads", ())),
                (len(snapshot.get("measurements", ())) + 15) // 16,
            )
        )
        for row in snapshot.get("workloads", ()):
            result.add_workload(BatchWorkload(**row))
        for row in snapshot.get("measurements", ()):
            result.add_measurement(Measurement(**row))
        return result

    def joint_state(self, version):
        state = self.state()
        selected = state.get("joint_state")
        path = self.root / "history.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["context"] != self.context or saved["weights_version"] > version:
                raise ValueError(
                    "Joint runtime state belongs to another execution/version"
                )
            if (
                saved["generation"] == state["generation"]
                and saved["bundle_id"] == state["bundle_id"]
            ):
                selected = saved.get("joint_state", selected)
        if selected is not None and selected["through_version"] > version:
            raise ValueError("Joint runtime state is newer than recovered weights")
        return selected

    def recover(self):
        with self._lock():
            state = self.state()
            if state["generation"] == 0:
                return None
            return self.prepare(
                self.root / "bundles" / state["bundle_id"] / "bundle.json"
            )


@dataclass(frozen=True)
class PreparedPolicy:
    manifest_json: str
    context_id: str
    policy: object
    probed: bool = False

    @property
    def bundle_id(self):
        return json.loads(self.manifest_json)["bundle_id"]
