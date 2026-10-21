# SPDX-License-Identifier: Apache-2.0

"""Executable joint-policy bundles with separate timing/search acceptance receipts."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict
from pathlib import Path

from areal.autopilot.elastic.policy.basic.joint import JointSearchResult
from areal.autopilot.elastic.policy.basic.online import OnlineJointPolicy
from areal.autopilot.elastic.runtime.state.checkpoint import atomic_json
from areal.autopilot.revision.contracts import BASELINE_SOURCE, digest, source_digest
from areal.autopilot.revision.execution import load_source
from areal.autopilot.revision.serial import (
    SCOPE,
    build_serial_model,
    joint_settings,
    serial_window_from_dict,
    static_profile,
    verify_timing_reference,
)

COMPONENTS = ("serial_model", "joint_optimizer")


def serial_policy_identity(profile, settings, *, horizon, idle_seconds):
    return digest(
        {
            "profile": static_profile(profile),
            "settings": settings,
            "horizon": horizon,
            "idle_seconds": idle_seconds,
        }
    )


def _stage_evidence(stage):
    stage.check()
    evidence = {
        "context": json.loads((stage.root / "context.json").read_text()),
        "report": json.loads((stage.root / stage.record["report"]).read_text()),
    }
    if stage.record["accepted"]:
        evidence["acceptance"] = json.loads(
            (stage.source.parent / "acceptance.json").read_text()
        )
    return evidence


def export_serial_policy(
    model_evaluator,
    search,
    model,
    optimizer,
    destination,
    *,
    reference,
    parent,
    idle_seconds,
    horizon,
):
    parent = None if parent is None else json.loads(parent.read_text())
    if parent is not None:
        validate_serial_bundle(parent)
    stages = {"serial_model": model, "joint_optimizer": optimizer}
    evidence = {k: _stage_evidence(s) for k, s in stages.items()}
    accepted = [k for k, s in stages.items() if s.record["accepted"]]
    proofs = {}
    components = {}
    for name, stage in stages.items():
        source = stage.source.read_text()
        components[name] = {"source": source, "sha256": source_digest(source)}
        proofs[name] = (
            evidence[name]
            if name in accepted
            else (None if parent is None else parent["acceptances"].get(name))
        )
        if source != BASELINE_SOURCE[name] and proofs[name] is None:
            raise ValueError("Revised joint component needs its original acceptance")
    payload = {
        "format_version": 1,
        "kind": "serial-joint-policy-v1",
        "profile": model_evaluator.profile,
        "settings": model_evaluator.settings,
        "horizon": horizon,
        "idle_seconds": idle_seconds,
        "components": components,
        "evaluations": evidence,
        "acceptances": proofs,
        "accepted_components": accepted,
        "parent_bundle_id": None if parent is None else parent["bundle_id"],
        "timing_reference": reference,
        "search_reference": {
            "cases": json.loads(search._scenarios_json),
            "prediction_sha256": search.context["model_snapshot_sha256"],
            "model_version": search.context["model_version"],
        },
        "protected_files": {**model_evaluator._code, **search._code},
    }
    value = {**payload, "bundle_id": digest(payload)}
    validate_serial_bundle(value)
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / f"{value['bundle_id']}.json"
    if output.exists() and json.loads(output.read_text()) != value:
        raise ValueError("Immutable joint bundle changed")
    if not output.exists():
        atomic_json(output, value)
    return output


def _check_evidence(evidence, name, sha, *, acceptance=False):
    context, report = evidence["context"], evidence["report"]
    if (
        context["target"] != name
        or report["target"] != name
        or report.get("valid") is not True
        or report["candidate_sha256"] != sha
        or report["benchmark_sha256"] != context["benchmark_sha256"]
        or digest(context["deployment_context"]) != context["benchmark_sha256"]
    ):
        raise ValueError("Joint component has no matching evaluation")
    if acceptance:
        proof = evidence.get("acceptance", {})
        if (
            proof.get("accepted") is not True
            or proof.get("candidate_sha256") != sha
            or proof.get("benchmark_sha256") != context["benchmark_sha256"]
        ):
            raise ValueError("Joint component has no matching acceptance")
    return context["deployment_context"], report


def validate_serial_bundle(value):
    payload = {k: v for k, v in value.items() if k != "bundle_id"}
    if (
        value.get("format_version") != 1
        or value.get("kind") != "serial-joint-policy-v1"
        or digest(payload) != value.get("bundle_id")
        or not value.get("accepted_components")
        or not set(value["accepted_components"]) <= set(COMPONENTS)
    ):
        raise ValueError("Unsupported or changed joint policy bundle")
    settings = json.loads(json.dumps(asdict(joint_settings(value["settings"]))))
    if (
        settings != value["settings"]
        or static_profile(value["profile"]) != value["profile"]
    ):
        raise ValueError("Joint policy settings/profile changed")
    contexts, reports = {}, {}
    for name in COMPONENTS:
        component = value["components"][name]
        if source_digest(component["source"]) != component["sha256"]:
            raise ValueError("Joint component source changed")
        contexts[name], reports[name] = _check_evidence(
            value["evaluations"][name],
            name,
            component["sha256"],
            acceptance=name in value["accepted_components"],
        )
        proof = value["acceptances"].get(name)
        if (
            component["source"] != BASELINE_SOURCE[name]
            or name in value["accepted_components"]
        ):
            if proof is None:
                raise ValueError("Missing original joint acceptance")
            prior, _ = _check_evidence(
                proof, name, component["sha256"], acceptance=True
            )
            if prior.get("settings") != settings or prior.get("code") != contexts[
                name
            ].get("code"):
                raise ValueError(
                    "Retained acceptance has a different search/code contract"
                )
            if name == "serial_model" and prior.get("profile_sha256") != digest(
                value["profile"]
            ):
                raise ValueError("Retained model acceptance has a different profile")
            if (
                name in value["accepted_components"]
                and proof != value["evaluations"][name]
            ):
                raise ValueError("Current acceptance differs from current evaluation")
    reference, search = value["timing_reference"], value["search_reference"]
    model, optimizer = contexts["serial_model"], contexts["joint_optimizer"]
    if (
        not reference["cases"]
        or any("seconds" in c or "input" in c for c in reference["cases"])
        or model.get("model_scope") != SCOPE
        or model["profile_sha256"] != digest(value["profile"])
        or model["settings"] != settings
        or model["cases_sha256"] != digest(reference["cases"])
        or reports["serial_model"]["metrics"]["prediction_sha256"]
        != reference["prediction_sha256"]
    ):
        raise ValueError("Joint timing probes/profile differ from current revalidation")
    version = "serial-model:" + digest(
        {
            "source": value["components"]["serial_model"]["sha256"],
            "context": value["evaluations"]["serial_model"]["context"][
                "benchmark_sha256"
            ],
            "reference": reference,
            "scope": SCOPE,
            "horizon": value["horizon"],
            "idle_seconds": value["idle_seconds"],
        }
    )
    if (
        optimizer.get("api_version") != "joint-optimizer-v1"
        or optimizer["settings"] != settings
        or optimizer["scenarios_sha256"] != digest(search["cases"])
        or optimizer["model_snapshot_sha256"] != search["prediction_sha256"]
        or optimizer["model_version"] != version
        or search["model_version"] != version
    ):
        raise ValueError("Optimizer was not revalidated on this executable joint model")
    # These are part of the immutable baseline contract, not revision parameters.
    if type(value["horizon"]) is not int or value["horizon"] < 2:
        raise ValueError("Invalid joint policy horizon")
    base = Path(__file__).resolve().parents[2]
    expected_files = {**model["code"], **optimizer["code"]}
    if value["protected_files"] != expected_files or not expected_files:
        raise ValueError("Joint policy runtime code identity changed")
    for name, expected in expected_files.items():
        path = (base / name).resolve()
        if (
            not path.is_relative_to(base)
            or not path.is_file()
            or source_digest(path.read_text()) != expected
        ):
            raise ValueError("Joint evaluation/runtime code changed")


class VerifiedJointOptimizer:
    """Trusted candidate search inside a fixed allowlist/query/selection contract."""

    def __init__(self, source, settings, *, max_seconds=5):
        self.source, self.settings = source, copy.deepcopy(settings)
        self.fixed = joint_settings(settings)
        self.candidates, self.max_seconds = self.fixed.candidates, max_seconds

    def optimize(self, scenario, simulator):
        from areal.autopilot.elastic.policy.basic.joint import JointCandidateResult

        started, calls, predictions = time.monotonic(), 0, {}
        fingerprint = digest(asdict(scenario))
        violations = []
        allowed = tuple(dict.fromkeys((scenario.current, *self.candidates)))
        outer = self

        class Reference:
            def evaluate(self, work, config):
                nonlocal calls
                try:
                    calls += 1
                    if calls > outer.fixed.max_evaluations:
                        raise ValueError("Joint runtime query budget exceeded")
                    if digest(asdict(work)) != fingerprint or config not in allowed:
                        raise ValueError(
                            "Joint runtime search changed its workload/allowlist"
                        )
                    result = simulator.evaluate(copy.deepcopy(scenario), config)
                    predictions[config] = result
                    return copy.deepcopy(result)
                except Exception as error:
                    violations.append(str(error))
                    raise

        candidate = load_source(self.source).build_joint_optimizer(
            {
                **self.settings,
                "candidates": self.candidates,
            }
        )
        result = candidate.optimize(copy.deepcopy(scenario), Reference())
        if violations:
            raise ValueError(
                "Joint runtime search violated fixed contract: " + "; ".join(violations)
            )
        if time.monotonic() - started > self.max_seconds:
            raise ValueError("Joint runtime optimizer time budget exceeded")
        if not isinstance(result, JointSearchResult):
            raise ValueError("Joint optimizer must return JointSearchResult")
        # Ignore claimed costs, eligibility and diagnostics. Reconstruct everything.
        config = result.config
        if config is not None and config not in allowed:
            raise ValueError("Joint optimizer selected outside its fixed allowlist")
        selected = (
            None
            if config is None
            else simulator.evaluate(copy.deepcopy(scenario), config)
        )
        current = simulator.evaluate(copy.deepcopy(scenario), scenario.current)

        def eligible(prediction):
            return prediction.status == "complete" and (
                outer.fixed.max_seconds is None
                or prediction.seconds <= outer.fixed.max_seconds
            )

        if selected is not None:
            if not eligible(selected):
                raise ValueError(
                    "Joint selection is unsupported/infeasible or exceeds deadline"
                )
            if (
                config != scenario.current
                and eligible(current)
                and selected.total_gpu_seconds
                >= current.total_gpu_seconds * (1 - self.fixed.switch_margin)
            ):
                raise ValueError("Joint selection does not satisfy fixed switch margin")
        elif eligible(current) or any(eligible(p) for p in predictions.values()):
            raise ValueError("Joint optimizer omitted an eligible selection")
        return JointSearchResult(
            config,
            selected,
            tuple(
                JointCandidateResult(c, p, eligible(p), p.reason)
                for c, p in predictions.items()
            ),
            len(allowed) - len(predictions),
            time.monotonic() - started,
            "Accepted joint optimizer; selected prediction independently reconstructed",
        )


def build_serial_policy(value):
    validate_serial_bundle(value)
    model = build_serial_model(
        value["components"]["serial_model"]["source"],
        value["profile"],
        idle_seconds=value["idle_seconds"],
    )
    verify_timing_reference(model, value["timing_reference"])
    cases = value["search_reference"]["cases"]
    settings = joint_settings(value["settings"])

    def project():
        rows = []
        for row in cases:
            scenario = serial_window_from_dict(row)
            configs = tuple(dict.fromkeys((scenario.current, *settings.candidates)))
            rows.append([asdict(model.evaluate(scenario, c)) for c in configs])
        return rows

    if (
        digest(project()) != value["search_reference"]["prediction_sha256"]
        or project() != project()
    ):
        raise ValueError(
            "Executable joint model differs from optimizer's frozen projection"
        )
    policy = OnlineJointPolicy(
        model,
        VerifiedJointOptimizer(
            value["components"]["joint_optimizer"]["source"],
            value["settings"],
            max_seconds=value["evaluations"]["joint_optimizer"]["context"][
                "deployment_context"
            ]["max_optimizer_seconds"],
        ),
        horizon=value["horizon"],
    )
    validate_serial_bundle(value)
    return policy
