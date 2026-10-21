# SPDX-License-Identifier: Apache-2.0

"""Reproducible, synthetic CPU validation; no GPU or LLM API calls."""

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def validate() -> dict:
    directory = ROOT / "areal/autopilot"
    spec = importlib.util.spec_from_file_location(
        "_areal_lifecycle_validation",
        directory / "__init__.py",
        submodule_search_locations=[str(directory)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    from _areal_lifecycle_validation.elastic.policy.cli import run_scenario

    scenario_path = ROOT / "examples/autopilot/elastic/lifecycle_scenario.json"
    base = json.loads(scenario_path.read_text())
    cases = {
        "cold_ready": ({}, "launch"),
        "long_idle": (
            {
                "observation": {"current_profile": "S029", "batch_ready": False},
                "history": [base["workload"]],
            },
            "release",
        ),
        "short_idle": (
            {
                "observation": {"current_profile": "S029", "batch_ready": False},
                "history": [{**base["workload"], "rollout_seconds": 1}],
            },
            "keep",
        ),
        "switch_declared": (
            {
                "observation": {"current_profile": "S026"},
                "workload": {**base["workload"], "rollout_seconds": 0},
                "compatible_transitions": [["S026", "S029"]],
            },
            "reconfigure",
        ),
        "switch_after_gap": (
            {
                "observation": {"current_profile": "S026"},
                "compatible_transitions": [["S026", "S029"]],
            },
            "keep",
        ),
        "switch_undeclared": ({"observation": {"current_profile": "S026"}}, "keep"),
        "time_bound": (
            {"optimizer": {**base["optimizer"], "max_seconds": 0.001}},
            "keep",
        ),
    }
    results, actors = {}, None
    for name, (changes, expected) in cases.items():
        result = run_scenario({**copy.deepcopy(base), **changes})
        if result["decision"]["action"] != expected:
            raise AssertionError(f"{name}: {result['decision']}")
        if actors is None:
            actors = [
                {
                    "config": row["config"],
                    "aliases": row["aliases"],
                    "stages_seconds": dict(row["stages"]),
                }
                for row in result["actor_predictions"]
            ]
        prediction = result["prediction"]
        results[name] = {
            "scenario_changes": changes,
            "decision": result["decision"],
            "selected_aliases": result["selected_aliases"],
            "evaluated": result["evaluated"],
            "rejected": result["rejected"],
            "plan": result["plan"],
            "prediction": None
            if prediction is None
            else {
                key: prediction[key]
                for key in (
                    "seconds",
                    "training_gpu_seconds",
                    "inference_gpu_seconds",
                    "total_gpu_seconds",
                    "samples",
                    "effective_tokens",
                )
            },
        }
    return {
        "kind": "synthetic_offline_validation",
        "scenario_sha256": hashlib.sha256(scenario_path.read_bytes()).hexdigest(),
        "assumptions": base["lifecycle_provenance"],
        "actor_profiles": actors,
        "cases": results,
        "limitations": result["limitations"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(validate(), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")
