# SPDX-License-Identifier: Apache-2.0

"""Offline CLI. All inputs and outputs are JSON; no training dependencies."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from .actor import ActorCostModel
from .calibration import bundled_calibration, public_validation
from .catalog import hardware_profile, qwen3_30b_a3b, read_data, trial_config
from .config import ActorConfig, Hardware
from .observability import actor_trace_times, trace_events
from .workload import WorkloadStats


def _json(path: str) -> dict:
    return json.loads(Path(path).read_text())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("predict", "trial-suite"):
        sub = commands.add_parser(name)
        if name == "predict":
            configs = sub.add_mutually_exclusive_group(required=True)
            configs.add_argument("--trial")
            configs.add_argument(
                "--config", help="Resolved AReaL experiment/actor JSON"
            )
        workload = sub.add_mutually_exclusive_group(required=True)
        workload.add_argument(
            "--workload", help="WorkloadStats JSON, actual global actor sequence counts"
        )
        workload.add_argument("--metrics", help="One AReaL summary metrics row")
        hardware = sub.add_mutually_exclusive_group()
        hardware.add_argument("--hardware", default="h200_sxm_reference")
        hardware.add_argument(
            "--hardware-json",
            help="Explicit Hardware JSON, rates in one-direction GB/s",
        )
        sub.add_argument(
            "--actor-overrides",
            help="JSON ActorConfig field overrides, excluding backend/parallel",
        )
        sub.add_argument(
            "--group-gbps",
            help="JSON effective communication bandwidth overrides by group",
        )
        sub.add_argument(
            "--public-calibration",
            choices=("none", "matching", "transfer"),
            default="none",
        )
        sub.add_argument("--output")
    validation = commands.add_parser("public-validation")
    validation.add_argument("--output")
    trace = commands.add_parser("trace")
    trace.add_argument("path")
    trace.add_argument("--level", choices=("auto", "trainer", "worker"), default="auto")
    trace.add_argument("--global-step", type=int)
    trace.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        if args.command == "public-validation":
            result = public_validation()
        elif args.command == "trace":
            result = actor_trace_times(
                trace_events(Path(args.path).read_text()),
                level=args.level,
                global_step=args.global_step,
            )
        else:
            stats = (
                WorkloadStats.from_areal_metrics(_json(args.metrics))
                if args.metrics
                else WorkloadStats(**_json(args.workload))
            )
            hardware = (
                Hardware(**_json(args.hardware_json))
                if args.hardware_json
                else hardware_profile(args.hardware)
            )
            overrides = _json(args.actor_overrides) if args.actor_overrides else {}
            if "parallel" in overrides or "backend" in overrides:
                raise ValueError("Use resolved --config to change parallel/backend")
            group_gbps = _json(args.group_gbps) if args.group_gbps else {}

            def predict(config, template=False):
                config = replace(config, **overrides)
                model = ActorCostModel(qwen3_30b_a3b(), hardware, group_gbps=group_gbps)
                if args.public_calibration != "none":
                    model = bundled_calibration().apply(
                        model,
                        config,
                        allow_transfer=args.public_calibration == "transfer",
                    )
                report = model.estimate(config, stats)
                if template:
                    report["assumptions"].append(
                        "Screenshot template: unspecified checkpoint/PPO/logprob options use ActorConfig defaults; supply --actor-overrides or resolved --config"
                    )
                return report

            if args.command == "predict":
                config = (
                    trial_config(args.trial)
                    if args.trial
                    else ActorConfig.from_areal(_json(args.config))
                )
                result = predict(config, bool(args.trial))
            else:
                result = {
                    "scenario": "Same supplied actor workload for all trials, not inferred from prompt batch size",
                    "trials": {},
                }
                for trial in read_data("trials.json")["trials"]:
                    try:
                        result["trials"][trial["id"]] = predict(
                            trial_config(trial["id"]), True
                        )
                    except ValueError as exc:
                        result["trials"][trial["id"]] = {
                            "invalid_input_or_inferred_packing": str(exc)
                        }
        text = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            Path(args.output).write_text(text)
        else:
            sys.stdout.write(text)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.exit(2, f"actor-cost: {exc}\n")
    return 0
