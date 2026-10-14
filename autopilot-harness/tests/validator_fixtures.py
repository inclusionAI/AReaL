"""Synthetic trusted evidence for CPU tests, never runtime capacity evidence."""

from copy import deepcopy
from pathlib import Path

from autopilot_harness.validator.checker import configuration_id, read_document

PROJECT = Path(__file__).resolve().parents[1]


def context_for(baseline, candidate):
    value = read_document(PROJECT / "examples/validator/context.json")
    try:
        value["baseline_id"] = configuration_id(baseline)
        value["candidate_id"] = configuration_id(candidate)
    except (TypeError, ValueError):
        # Invalid JSON values are rejected before any evidence is consulted.
        return None

    def changes(a, b, path=()):
        if isinstance(a, dict) and isinstance(b, dict):
            result = []
            for key in a.keys() | b.keys():
                if key not in a or key not in b:
                    result.append(".".join((*path, key)))
                else:
                    result.extend(changes(a[key], b[key], (*path, key)))
            return result
        return [] if type(a) is type(b) and a == b else [".".join(path)]

    value["transition"]["changes"] = sorted(changes(baseline, candidate))
    fraction = candidate.get("sglang", {}).get("mem_fraction_static")
    if type(fraction) in (int, float) and 0 < fraction <= 1:
        value["memory"]["inference_memory_fraction"] = fraction
    return deepcopy(value)
