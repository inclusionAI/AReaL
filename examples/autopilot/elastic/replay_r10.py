# SPDX-License-Identifier: Apache-2.0

"""Freeze an external R10 reference; replay original Actor4 against real arrivals."""

from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path


def load_namespace(name, root):
    """Load numerical modules without importing optional GPU/engine packages."""
    for suffix, path in [
        ("", root),
        (".elastic", root / "elastic"),
        (".elastic.policy", root / "elastic/policy"),
    ]:
        spec = importlib.machinery.ModuleSpec(
            name + suffix, loader=None, is_package=True
        )
        module = importlib.util.module_from_spec(spec)
        module.__path__ = [str(path)]
        sys.modules[spec.name] = module


def freeze_reference(source, destination):
    """Copy numerical source/data, reject concurrent edits, and bind every file."""
    source = source.resolve()
    if not (source / "trials/r10/manifest.json").is_file():
        raise ValueError("Reference must be an Expert directory with the R10 archive")
    autopilot = source.parents[2]
    files = [
        autopilot / "contracts.py",
        *sorted(
            p
            for p in source.rglob("*")
            if p.is_file()
            and "__pycache__" not in p.parts
            and p.suffix in {".py", ".json", ".jsonl", ".gz"}
        ),
    ]
    hashes = {}
    for path in files:
        relative = path.relative_to(autopilot)
        payload = path.read_bytes()
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        hashes[str(relative)] = hashlib.sha256(payload).hexdigest()
    for path in files:
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != hashes[str(path.relative_to(autopilot))]
        ):
            raise ValueError(
                "Reference changed during snapshot; retry with stable inputs"
            )
    return hashes


def calibrate_advantage_rows(rows, profile_sha256, steps=(2, 3, 4, 5)):
    """Read labels only from the selected prefix; freeze the observed warm-up."""
    local_root = Path(__file__).resolve().parents[3] / "areal/autopilot"
    load_namespace("_r10_structural", local_root)
    from _r10_structural.elastic.advantage import (
        apply_advantage_scale,
        fit_advantage_scale,
    )

    by_step = {r["step"]: r for r in rows}
    if set(steps) - by_step.keys():
        raise ValueError("Missing calibration prefix steps")
    if any(by_step[s]["advantage_prediction"]["seconds"] is None for s in steps):
        receipt = {"status": "unsupported", "steps": list(steps)}
    else:
        samples = [
            {
                "step": s,
                "prediction": by_step[s]["advantage_prediction"],
                "observed_seconds": next(
                    p["seconds"]
                    for p in by_step[s]["phases"]
                    if p["name"] == "compute_advantage"
                ),
            }
            for s in steps
        ]
        receipt = fit_advantage_scale(samples, profile_sha256=profile_sha256)
    for row in rows:
        if receipt["status"] == "fitted":
            prediction = apply_advantage_scale(
                row["advantage_prediction"], receipt, profile_sha256=profile_sha256
            )
            row["advantage_prediction"] = prediction
            for key in ("predicted_phases", "strict_predicted_phases"):
                for phase in row[key]:
                    if phase["name"] == "compute_advantage":
                        phase.update(
                            seconds=prediction["seconds"],
                            evidence=prediction["evidence"],
                        )
            row["phase_notes"]["compute_advantage"] = [
                "Structural predictor with frozen whole-stage scale fitted only on steps 2-5",
                "Component rates remain priors; allocated times do not identify physical rates",
            ]
        row["simulation_role"] = (
            "observed_prefix" if row["step"] <= max(steps) else "prediction"
        )
        if row["simulation_role"] == "observed_prefix":
            for key in ("predicted_phases", "strict_predicted_phases"):
                row[key] = [dict(p) for p in row["phases"]]
    return receipt


def calibrate_publication_rows(rows, result, point_type, calibration_type):
    """Fit a whole-boundary scale on steps 2-5; component priors stay unchanged."""
    by_step = {row["step"]: row for row in rows}
    if result.duration.seconds is None:
        return {"status": "unsupported", "steps": [2, 3, 4, 5]}
    calibration = calibration_type.fit(
        "r10-publication-prefix-v1",
        tuple(
            point_type(
                f"r10-step-{step}",
                result.profile,
                "update_weights",
                result.duration.seconds,
                next(
                    p["seconds"]
                    for p in by_step[step]["phases"]
                    if p["name"] == "update_weights"
                ),
                result.features,
                "trainer.pause-through-version-confirmation",
            )
            for step in (2, 3, 4, 5)
        ),
    )
    calibrated = result.calibrated(calibration)
    for row in rows:
        row["publication_prediction"] = asdict(calibrated)
        for key in ("predicted_phases", "strict_predicted_phases"):
            for phase in row[key]:
                if phase["name"] == "update_weights":
                    phase.update(
                        seconds=calibrated.duration.seconds,
                        evidence=calibrated.duration.evidence,
                    )
        row["phase_notes"]["update_weights"] = [
            "Whole publication scale fitted on steps 2-5 only; raw component times remain uncalibrated priors",
        ]
        row["simulation_role"] = "observed_prefix" if row["step"] <= 5 else "prediction"
        if row["step"] <= 5:
            for key in ("predicted_phases", "strict_predicted_phases"):
                row[key] = [dict(p) for p in row["phases"]]
    return {"status": "fitted", **asdict(calibration)}


def reference_input(
    root,
    diagnostic,
    advantage_profile=None,
    advantage_calibration="prefix",
    actor_target_path=None,
    hardware=None,
    system=None,
    publication_profile=None,
    publication_calibration="none",
):
    load_namespace("_r10_reference", root)
    from _r10_reference.contracts import WorkloadStats
    from _r10_reference.elastic.policy.expert_designed import Catalog
    from _r10_reference.elastic.policy.expert_designed.catalog import (
        numerical_environment,
    )
    from _r10_reference.elastic.policy.expert_designed.core import code_revision
    from _r10_reference.elastic.policy.expert_designed.trial import DiagnosticActor

    structural_profile = None
    if advantage_profile is not None:
        local_root = Path(__file__).resolve().parents[3] / "areal/autopilot"
        load_namespace("_r10_structural", local_root)
        from _r10_structural.elastic.advantage import (
            AdvantageProfile,
            estimate_advantage,
        )

        structural_profile = AdvantageProfile.from_dict(
            json.loads(advantage_profile.read_text())
        )

    catalog = Catalog(root / "elastic/policy/expert_designed")
    trial = catalog.trial("r10")
    verified = trial.verify()
    assembled = trial.models()
    config = assembled.config
    if (
        config.actor.parallel.world_size != 4
        or config.actor.backend != "fsdp"
        or len(config.rollout_devices) != 4
    ):
        raise ValueError("This replay only accepts the original Actor4/Rollout4 recipe")
    target_document = None
    target_model = None
    target_actor = config.actor
    target_metadata = None
    if actor_target_path is not None or hardware is not None:
        if diagnostic or advantage_calibration != "none" or structural_profile is None:
            raise ValueError(
                "Counterfactual requires structural advantage without source calibration/fallback"
            )
        local_root = Path(__file__).resolve().parents[3] / "areal/autopilot"
        load_namespace("_r10_target", local_root)
        from _r10_target.elastic.actor_target import (
            actor_target,
            target_auxiliary,
            target_phases,
        )

        target_document = (
            json.loads(actor_target_path.read_text())
            if actor_target_path
            else {
                "schema_version": 1,
                "actor": asdict(config.actor),
                "runtime_revision": assembled.raw_actor.runtime_revision
                + ":hardware-prior",
            }
        )
        if hardware is not None:
            if "hardware" in target_document:
                raise ValueError("Choose catalog hardware or inline target hardware")
            target_document["hardware"] = {
                "values": asdict(catalog.hardware(hardware, system))
            }
        target_actor, target_model = actor_target(
            target_document, config.actor, assembled.raw_actor
        )
        target_metadata = {
            "kind": "counterfactual",
            "actor": asdict(target_actor),
            "hardware": asdict(target_model.hardware),
            "runtime_revision": target_model.runtime_revision,
            "source_calibration_reused": False,
            "target_sha256": hashlib.sha256(actor_target_path.read_bytes()).hexdigest()
            if actor_target_path
            else None,
            "catalog_hardware": hardware,
            "catalog_system": system,
        }
        structural_profile = replace(
            structural_profile,
            execution_id=target_model.runtime_revision
            + ":"
            + target_actor.backend
            + ":"
            + target_model.hardware.name,
            rates=replace(
                structural_profile.rates,
                device_bytes_per_second=target_model.hardware.hbm_gbps
                * 1e9
                * target_model.efficiency.hbm,
            ),
        )
    timeline = trial.read("observed_timeline")
    t0 = timeline["start_epoch"]
    exported = [g for g in trial.read("group_observations") if g["export"] is not None]
    arrivals = []
    for group in exported:
        if group["completed"] is None:
            raise ValueError("Exported group lacks its runner completion timestamp")
        arrivals.append(
            {
                "group_id": str(group["dispatch"]["task_id"]),
                "seconds": group["completed"]["wall_time"] - t0,
                "samples": len(group["export"]["samples"]),
            }
        )
    calibration = trial.read("calibration")
    actor = assembled.actor
    if diagnostic:
        fallback = calibration["diagnostic_fallback"]
        actor = DiagnosticActor(
            assembled.raw_actor,
            actor,
            fallback["advantage_seconds"],
            fallback["advantage_source_id"],
        )
    publish = next(
        t["tasks"][0]["measured"]
        for t in calibration["transitions"]
        if t["action"] == "publish"
    )
    publication_result = None
    if publication_profile is not None:
        from _r10_reference.elastic.policy.expert_designed import (
            CalibrationPoint,
            ExpertPublicationModel,
            PhaseCalibration,
        )

        publication_model, publication_work = ExpertPublicationModel.from_dict(
            json.loads(publication_profile.read_text()),
            (target_model or assembled.raw_actor).hardware,
        )
        publication_result = publication_model.estimate(
            target_model or assembled.raw_actor,
            target_actor,
            config.rollout,
            publication_work,
        )
    batches = trial.read("batch_observations")
    if len(batches) != 16 or len(timeline["steps"]) != 16:
        raise ValueError("R10 replay requires all sixteen original updates")
    rows, previous_end = [], 0
    for batch, observed in zip(batches, timeline["steps"], strict=True):
        if batch["step"] != observed["step"] or batch["step"] != len(rows) + 1:
            raise ValueError("R10 steps must be complete and ordered")
        lengths = tuple(batch["sequence_lengths"])
        if sum(lengths) != batch["sequence_tokens"]:
            raise ValueError("Logical sequence tokens disagree with batch workload")
        workload = WorkloadStats(
            n_seqs=len(lengths),
            sequence_tokens=sum(lengths),
            max_seq_len=max(lengths),
            min_seq_len=min(lengths),
            loss_tokens=batch["loss_tokens"],
            sequence_lengths=lengths,
            trajectory_group_sizes=tuple(batch["trajectory_group_sizes"]),
            trajectory_dispatch_weights=tuple(batch["trajectory_dispatch_weights"]),
        )
        strict_prediction = (target_model or assembled.actor).estimate(
            target_actor, workload
        )
        prediction = (
            actor.estimate(config.actor, workload) if diagnostic else strict_prediction
        )
        measured = batch["measured_seconds"]
        names = (
            "recompute_logp",
            "compute_advantage",
            "train_step",
            "update_weights",
            "save",
        )
        phases = [
            {"name": name, "seconds": measured[name], "evidence": "measured"}
            for name in names
        ]
        predictions = [
            {"name": name, "seconds": estimate.seconds, "evidence": estimate.evidence}
            for name, estimate in prediction.stages
        ]
        predictions.extend([{"name": "update_weights", **publish}, phases[-1]])
        strict_phases = [
            {"name": name, "seconds": estimate.seconds, "evidence": estimate.evidence}
            for name, estimate in strict_prediction.stages
        ]
        strict_phases.extend([{"name": "update_weights", **publish}, phases[-1]])
        backend_detail = None
        if target_model is not None:
            predictions, backend_detail = target_phases(strict_prediction)
            predictions.extend(
                target_auxiliary(target_document, final=batch["step"] == 16)
            )
            strict_phases = [dict(p) for p in predictions]
        if publication_result is not None:
            if target_document and "update_weights" in target_document.get(
                "auxiliary", {}
            ):
                raise ValueError(
                    "Choose structural publication or explicit auxiliary duration, not both"
                )
            for stage_list in (predictions, strict_phases):
                for stage in stage_list:
                    if stage["name"] == "update_weights":
                        stage.update(
                            seconds=publication_result.duration.seconds,
                            evidence=publication_result.duration.evidence,
                        )
        advantage = None
        phase_notes = {n: e.notes for n, e in prediction.stages}
        if publication_result is not None:
            phase_notes["update_weights"] = publication_result.duration.notes
        if structural_profile is not None:
            groups, offset = [], 0
            for size in batch["trajectory_group_sizes"]:
                groups.append(lengths[offset : offset + size])
                offset += size
            if offset != len(lengths):
                raise ValueError("Atomic groups must partition all sequence lengths")
            broadcast = (
                target_document.get("advantage_broadcast", {})
                if target_document
                else {}
            )
            try:
                advantage = estimate_advantage(
                    groups,
                    dp=target_actor.parallel.dp,
                    profile=structural_profile,
                    dispatch_weights=batch["trajectory_dispatch_weights"],
                    model_parallel_size=target_actor.parallel.tp
                    * target_actor.parallel.cp
                    * target_actor.parallel.pp,
                    broadcast_bytes_per_second=broadcast.get("bytes_per_second"),
                    broadcast_latency_seconds=broadcast.get("latency_seconds"),
                )
            except ValueError as exc:
                if target_model is None:
                    raise
                advantage = {
                    "seconds": None,
                    "evidence": "unsupported",
                    "reason": str(exc),
                }
            if backend_detail and backend_detail["feasibility"] == "infeasible":
                advantage = {
                    **advantage,
                    "seconds": None,
                    "evidence": "unsupported",
                    "reason": "Actor model marks target infeasible",
                }
            # Override only this stage: mutating the frozen Actor model would
            # invalidate the identity of its logp/train calibration profile.
            for stage_list in (predictions, strict_phases):
                for stage in stage_list:
                    if stage["name"] == "compute_advantage":
                        stage.update(
                            seconds=advantage["seconds"],
                            evidence=advantage["evidence"],
                        )
            phase_notes["compute_advantage"] = [
                "Structural prior; no affine prediction, timing fit or fallback",
                advantage.get("reason", "Operation counts and rates are revisable"),
            ]
        before = observed["start"] - previous_end
        after = observed["end"] - observed["start"] - sum(observed["phases"].values())
        if before < -1e-6 or after < -1e-6:
            raise ValueError("Invalid observed controller residual")
        group_ids = tuple(str(r["task_id"]) for r in batch["ranks"])
        if len(group_ids) != 4 or len(set(group_ids)) != 4 or len(lengths) != 16:
            raise ValueError("Original R10 batches require four groups of four samples")
        rows.append(
            {
                "step": batch["step"],
                "group_ids": group_ids,
                "sequence_tokens": sum(lengths),
                "effective_tokens": batch["loss_tokens"],
                "before_seconds": max(0, before),
                "after_seconds": max(0, after),
                "phases": phases,
                "predicted_phases": predictions,
                "strict_predicted_phases": strict_phases,
                "phase_notes": phase_notes,
                "advantage_prediction": advantage,
                "backend_prediction": backend_detail,
                "publication_prediction": asdict(publication_result)
                if publication_result
                else None,
                "observed": observed,
            }
        )
        previous_end = observed["end"]
    profile_sha256 = (
        hashlib.sha256(advantage_profile.read_bytes()).hexdigest()
        if advantage_profile
        else None
    )
    calibration_receipt = None
    publication_receipt = None
    if publication_result is not None and publication_calibration == "prefix":
        if target_metadata is not None:
            raise ValueError(
                "Source publication labels cannot calibrate a target configuration"
            )
        publication_receipt = calibrate_publication_rows(
            rows,
            publication_result,
            CalibrationPoint,
            PhaseCalibration,
        )
    if structural_profile is not None and advantage_calibration == "prefix":
        calibration_receipt = calibrate_advantage_rows(rows, profile_sha256)
    return {
        "trial": "r10",
        "actor_gpus": target_actor.parallel.world_size,
        "observed_actor_gpus": 4,
        "actor_target": target_metadata,
        "arrivals": arrivals,
        "steps": rows,
        "observed_seconds": timeline["duration"],
        "clock_alignment_seconds": timeline["clock_alignment_width"],
        "evaluation_split": {
            "cold_start": [1],
            "calibration": [2, 3, 4, 5],
            "validation": list(range(6, 17)),
        },
        "advantage_model": {
            "kind": "structural" if structural_profile is not None else "legacy-affine",
            "profile_sha256": profile_sha256,
            "evidence": "mixed_by_step"
            if calibration_receipt
            else ("assumed" if structural_profile is not None else "archive"),
            "calibration": calibration_receipt,
            "effective_profile": asdict(structural_profile)
            if target_metadata
            else None,
        },
        "publication_model": {
            "kind": "structural"
            if publication_result
            else ("target-auxiliary" if target_metadata else "archive-calibrated"),
            "profile_sha256": hashlib.sha256(
                publication_profile.read_bytes()
            ).hexdigest()
            if publication_profile
            else None,
            "calibration": publication_receipt,
            "raw_prediction": asdict(publication_result)
            if publication_result
            else None,
            "scope": "Dense separation AWEX only; complete target geometry required; component rates remain priors",
        },
        "observed_prefix_steps": [
            row["step"]
            for row in rows
            if row.get("simulation_role") == "observed_prefix"
        ],
        "reference": {
            "archive": verified,
            "code_revision": code_revision(),
            "numerical_environment": numerical_environment(),
            "manifest_sha256": hashlib.sha256(
                (trial.root / "manifest.json").read_bytes()
            ).hexdigest(),
        },
    }


def evaluate(data):
    root = Path(__file__).resolve().parents[3] / "areal/autopilot"
    load_namespace("_r10_replay", root)
    from _r10_replay.elastic.replay import GroupArrival, replay_actor, steps_from_dict

    arrivals = tuple(GroupArrival(**a) for a in data["arrivals"])
    observed_gpus = data.get("observed_actor_gpus", 4)
    predicted_gpus = data.get("actor_gpus", 4)
    measured = replay_actor(
        arrivals, steps_from_dict(data["steps"]), actor_gpus=observed_gpus
    )
    predicted = replay_actor(
        arrivals,
        steps_from_dict(data["steps"], "predicted_phases"),
        actor_gpus=predicted_gpus,
    )
    strict = replay_actor(
        arrivals,
        steps_from_dict(data["steps"], "strict_predicted_phases"),
        actor_gpus=predicted_gpus,
    )
    errors = []
    for row, actual in zip(measured["steps"], data["steps"], strict=True):
        reference = actual["observed"]
        errors.extend(
            [
                abs(row["start"] - reference["start"]),
                abs(row["end"] - reference["end"]),
                abs(
                    row["actor_start"]
                    - reference["start"]
                    - reference["phases"]["wait"]
                ),
            ]
        )
    alignment = {
        "max_boundary_error_seconds": max(errors),
        "finish_error_seconds": measured["seconds"] - data["observed_seconds"],
        "observed_actor_gpu_hours": observed_gpus * data["observed_seconds"] / 3600,
    }
    # One millisecond for rounded runner timestamps, plus the measured clock interval.
    tolerance = data["clock_alignment_seconds"] + 0.001
    alignment["tolerance_seconds"] = tolerance
    alignment["passed"] = max(errors) <= tolerance
    metrics = {}
    for phase in (
        "recompute_logp",
        "compute_advantage",
        "train_step",
        "update_weights",
    ):
        if any(
            not any(p["name"] == phase for p in r["strict_predicted_phases"])
            for r in data["steps"]
        ):
            continue
        available = [
            r
            for r in data["steps"]
            if r["step"] in data["evaluation_split"]["validation"]
            and next(p for p in r["strict_predicted_phases"] if p["name"] == phase)[
                "seconds"
            ]
            is not None
        ]
        pairs = [
            (
                next(p for p in r["strict_predicted_phases"] if p["name"] == phase)[
                    "seconds"
                ],
                next(p for p in r["phases"] if p["name"] == phase)["seconds"],
            )
            for r in available
        ]
        metrics[phase] = {
            "covered": len(pairs),
            "total": len(data["evaluation_split"]["validation"]),
            "evidence_counts": {
                evidence: sum(
                    next(
                        p for p in row["strict_predicted_phases"] if p["name"] == phase
                    )["evidence"]
                    == evidence
                    for row in available
                )
                for evidence in sorted(
                    {
                        next(
                            p
                            for p in row["strict_predicted_phases"]
                            if p["name"] == phase
                        )["evidence"]
                        for row in available
                    }
                )
            },
            "wape": sum(abs(p - a) for p, a in pairs) / sum(a for _, a in pairs)
            if pairs
            else None,
        }
    # Compare all three stages on the SAME rows. Stage-normalized absolute
    # errors add; the error of their per-step sum can cancel across stages.
    names = ("recompute_logp", "compute_advantage", "train_step")
    common = []
    for row in data["steps"]:
        if row["step"] not in data["evaluation_split"]["validation"]:
            continue
        prediction = {p["name"]: p["seconds"] for p in row["strict_predicted_phases"]}
        actual = {p["name"]: p["seconds"] for p in row["phases"]}
        if all(prediction[name] is not None for name in names):
            common.append((row["step"], prediction, actual))
    denominator = sum(a[n] for _, _, a in common for n in names)
    common_metrics = {}
    for name in names:
        numerator = sum(abs(p[name] - a[name]) for _, p, a in common)
        phase_total = sum(a[name] for _, _, a in common)
        common_metrics[name] = {
            "wape": numerator / phase_total if phase_total else None,
            "absolute_error_over_actor_total": numerator / denominator
            if denominator
            else None,
        }
    actor_error = sum(
        abs(sum(p[n] for n in names) - sum(a[n] for n in names)) for _, p, a in common
    )
    prefix_count = len(data.get("observed_prefix_steps", []))
    cutoff = measured["steps"][prefix_count - 1]["end"] if prefix_count else 0.0
    prediction_window = {
        "observed_prefix_steps": data.get("observed_prefix_steps", []),
        "start_seconds": cutoff,
        "first_predicted_step": prefix_count + 1,
        "measured_seconds": measured["seconds"] - cutoff,
        "predicted_seconds": predicted["seconds"] - cutoff
        if predicted["seconds"] is not None
        else None,
        "measured_actor_gpu_hours": observed_gpus
        * (measured["seconds"] - cutoff)
        / 3600,
        "predicted_actor_gpu_hours": predicted_gpus
        * (predicted["seconds"] - cutoff)
        / 3600
        if predicted["seconds"] is not None
        else None,
    }
    common_validation = {
        "steps": [step for step, _, _ in common],
        "actor_measured_seconds": denominator,
        "phases": common_metrics,
        "actor_total_wape": actor_error / denominator if denominator else None,
    }
    if data.get("actor_target") is not None:
        metrics = {
            "status": "not_scored",
            "reason": "No target backend/hardware measurements; source FSDP timings are not target labels",
        }
        common_validation = dict(metrics)
    return {
        "alignment": alignment,
        "measured_replay": measured,
        "expert_replay": predicted,
        "expert_strict_replay": strict,
        "development_validation": metrics,
        "prediction_window": prediction_window,
        "common_validation": common_validation,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-root",
        required=True,
        type=Path,
        help="External Expert directory containing the R10 archive",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference-python", default=sys.executable)
    parser.add_argument("--diagnostic-fallback", action="store_true")
    parser.add_argument(
        "--actor-target",
        type=Path,
        help="Explicit backend/mesh/optimizer recipe; target predictions are uncalibrated",
    )
    parser.add_argument(
        "--hardware", help="Catalog GPU profile, e.g. h200-sxm-reference"
    )
    parser.add_argument(
        "--system", help="Catalog machine profile; required with --hardware"
    )
    parser.add_argument(
        "--publication-profile",
        type=Path,
        help="Rank work and peer traffic/rates; default: bundled R10 geometry for original config",
    )
    parser.add_argument(
        "--publication-calibration",
        choices=("prefix", "none"),
        help="Original config default: fit whole publication on steps 2-5; targets: none",
    )
    parser.add_argument(
        "--legacy-publication",
        action="store_true",
        help="Reproduce archived constant publication instead of the structural model",
    )
    parser.add_argument(
        "--advantage-calibration",
        choices=("prefix", "none"),
        help="Default: fit structural stage scale on steps 2-5, observe steps 1-5, predict steps 6-16",
    )
    parser.add_argument(
        "--advantage-model",
        choices=("structural", "legacy-affine"),
        default="structural",
    )
    parser.add_argument(
        "--advantage-profile",
        type=Path,
        help="Explicit structural recipe/rates; default: bundled R10 v1 prior",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if bool(args.hardware) != bool(args.system):
        parser.error("--hardware and --system must be selected together")
    counterfactual = args.actor_target is not None or args.hardware is not None
    if args.legacy_publication and (
        counterfactual
        or args.publication_profile is not None
        or args.publication_calibration == "prefix"
    ):
        parser.error(
            "Archived publication is only for the original config without structural publication"
        )
    if args.publication_calibration is None:
        args.publication_calibration = (
            "none" if counterfactual or args.legacy_publication else "prefix"
        )
    if counterfactual and args.publication_calibration != "none":
        parser.error("Targets cannot fit publication using source timing labels")
    if (
        not counterfactual
        and not args.legacy_publication
        and args.publication_profile is None
    ):
        args.publication_profile = (
            Path(__file__).with_name("profiles") / "r10-publication-structural-v1.json"
        )
    if args.advantage_calibration is None:
        args.advantage_calibration = (
            "prefix"
            if args.advantage_model == "structural" and not counterfactual
            else "none"
        )
    if counterfactual and (
        args.advantage_model != "structural"
        or args.advantage_calibration != "none"
        or args.diagnostic_fallback
    ):
        parser.error(
            "Targets require structural advantage without source calibration or diagnostic fallback"
        )
    if args.advantage_model == "legacy-affine" and args.advantage_calibration != "none":
        parser.error("Prefix calibration requires --advantage-model structural")
    if args.advantage_model == "legacy-affine" and args.advantage_profile is not None:
        parser.error("--advantage-profile requires --advantage-model structural")
    if args.advantage_model == "structural" and args.advantage_profile is None:
        args.advantage_profile = (
            Path(__file__).with_name("profiles") / "r10-advantage-structural-v1.json"
        )
    if args.worker:
        print(
            json.dumps(
                reference_input(
                    args.reference_root,
                    args.diagnostic_fallback,
                    args.advantage_profile,
                    args.advantage_calibration,
                    args.actor_target,
                    args.hardware,
                    args.system,
                    args.publication_profile,
                    args.publication_calibration,
                ),
                allow_nan=False,
            )
        )
        return
    if args.output is None:
        parser.error("--output is required")
    source_paths = (
        Path(__file__).resolve(),
        Path(__file__).resolve().parents[3] / "areal/autopilot/elastic/replay.py",
        Path(__file__).resolve().parents[3] / "areal/autopilot/elastic/advantage.py",
        Path(__file__).resolve().parents[3] / "areal/autopilot/elastic/actor_target.py",
    )
    implementation = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths
    }
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output / "reference"
    hashes = freeze_reference(args.reference_root, snapshot)
    (args.output / "reference-manifest.json").write_text(
        json.dumps(hashes, indent=2) + "\n"
    )
    profile_bytes = None
    target_bytes = None
    publication_bytes = None
    if args.publication_profile is not None:
        publication_bytes = args.publication_profile.read_bytes()
        args.publication_profile = args.output / "publication-profile.json"
        args.publication_profile.write_bytes(publication_bytes)
    if args.actor_target is not None:
        target_bytes = args.actor_target.read_bytes()
        args.actor_target = args.output / "actor-target.json"
        args.actor_target.write_bytes(target_bytes)
    if args.advantage_profile is not None:
        profile_bytes = args.advantage_profile.read_bytes()
        args.advantage_profile = args.output / "advantage-profile.json"
        args.advantage_profile.write_bytes(profile_bytes)
    command = [
        args.reference_python,
        str(Path(__file__).resolve()),
        "--worker",
        "--reference-root",
        str(snapshot.resolve()),
        "--advantage-model",
        args.advantage_model,
        "--advantage-calibration",
        args.advantage_calibration,
    ]
    if args.advantage_profile is not None:
        command.extend(["--advantage-profile", str(args.advantage_profile.resolve())])
    if args.actor_target is not None:
        command.extend(["--actor-target", str(args.actor_target.resolve())])
    command.extend(["--publication-calibration", args.publication_calibration])
    if args.publication_profile is not None:
        command.extend(
            ["--publication-profile", str(args.publication_profile.resolve())]
        )
    if args.legacy_publication:
        command.append("--legacy-publication")
    if args.hardware is not None:
        command.extend(["--hardware", args.hardware, "--system", args.system])
    if args.diagnostic_fallback:
        command.append("--diagnostic-fallback")
    completed = subprocess.run(
        command, capture_output=True, text=True, timeout=60, check=True
    )
    data = json.loads(completed.stdout)
    if data["advantage_model"]["calibration"] is not None:
        (args.output / "advantage-calibration.json").write_text(
            json.dumps(
                data["advantage_model"]["calibration"], indent=2, allow_nan=False
            )
            + "\n"
        )
    if data["publication_model"]["calibration"] is not None:
        (args.output / "publication-calibration.json").write_text(
            json.dumps(
                data["publication_model"]["calibration"], indent=2, allow_nan=False
            )
            + "\n"
        )
    result = evaluate(data)
    if implementation != {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths
    }:
        raise ValueError("Replay implementation changed during execution")
    for relative, digest in hashes.items():
        if hashlib.sha256((snapshot / relative).read_bytes()).hexdigest() != digest:
            raise ValueError("Frozen reference changed during execution")
    if args.advantage_profile is not None and (
        args.advantage_profile.read_bytes() != profile_bytes
        or data["advantage_model"]["profile_sha256"]
        != hashlib.sha256(profile_bytes).hexdigest()
    ):
        raise ValueError("Structural profile changed during execution")
    if args.actor_target is not None and (
        args.actor_target.read_bytes() != target_bytes
        or data["actor_target"]["target_sha256"]
        != hashlib.sha256(target_bytes).hexdigest()
    ):
        raise ValueError("Actor target changed during execution")
    if args.publication_profile is not None and (
        args.publication_profile.read_bytes() != publication_bytes
        or data["publication_model"]["profile_sha256"]
        != hashlib.sha256(publication_bytes).hexdigest()
    ):
        raise ValueError("Publication profile changed during execution")
    report = {
        "format_version": "r10-actor-replay-v5",
        "implementation_sha256": implementation,
        "reference": data["reference"],
        "diagnostic_fallback": args.diagnostic_fallback,
        "advantage_model": data["advantage_model"],
        "actor_target": data["actor_target"],
        "publication_model": data["publication_model"],
        **result,
        "scope": [
            "Selected Actor GPUs held from main-loop start through final save; startup and teardown excluded",
            "Fixed original 16 batches/64 groups; five accepted prefetch groups never add training work",
            "Runner completion timestamps proxy availability; export/collector latency is not separately identified",
            "Measured relative controller overhead retained; target save requires explicit auxiliary profiles; publication accepts structural geometry/rates or an explicit auxiliary prior"
            if counterfactual
            else "Measured relative controller overhead and saver durations retained; no historical Actor waits/start times imposed",
            "Frozen Expert supplies target logp/train priors; structural advantage uses separately identified hardware/broadcast priors"
            if counterfactual
            else "Frozen Expert supplies logp/train; publication model and whole-stage prefix calibration are separately recorded",
            "Target predicts all sixteen steps without observing a source-config warmup or fitting source timing labels"
            if counterfactual
            else "Default observes steps 1-5, independently fits advantage/publication whole-stage scales on 2-5 and predicts 6-16; both calibration flags must be none for a full prior replay",
            "Whole-stage calibration does not identify component rates or prove cross-config accuracy; width extrapolation remains assumed",
            "Original staleness2 trace is exogenous; counterfactual admission, reward and quality not validated",
            "Strict unknown phases have no full-window cost; diagnostic fallback is assumed, not validated coverage",
            "Counterfactual backend/hardware uses fresh Expert priors, no source calibration; source timings are never target validation labels",
            "Target publication requires target-bound structural geometry/rates or an explicit auxiliary prior; missing publication/save or infeasible targets have no full-window cost",
            "Structural publication retains raw component priors; one whole-stage scale does not identify staging, link or control rates",
        ],
    }
    (args.output / "inputs.json").write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n"
    )
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "alignment": report["alignment"],
                "measured_seconds": report["measured_replay"]["seconds"],
                "measured_actor_gpu_hours": report["measured_replay"][
                    "actor_gpu_hours"
                ],
                "expert_status": report["expert_replay"]["status"],
                "expert_seconds": report["expert_replay"]["seconds"],
                "unsupported_steps": report["expert_replay"]["unsupported_steps"],
            },
            indent=2,
        )
    )
    if not report["alignment"]["passed"]:
        raise SystemExit(
            "Measured replay exceeds observed clock alignment; inspect report.json"
        )


if __name__ == "__main__":
    main()
