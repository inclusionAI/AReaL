# SPDX-License-Identifier: Apache-2.0

"""JSON decoding for the public Elastic planning inputs."""

from ...contracts import BatchWorkload, ElasticObservation, TrainingConfig
from .simulator import ForecastUpdate, HistorySnapshot, Measurement, PlanStep


def observation_from_dict(data: dict) -> ElasticObservation:
    data = dict(data)
    for key, cls in (
        ("training_config", TrainingConfig),
        ("checkpoint_config", TrainingConfig),
        ("next_batch", BatchWorkload),
    ):
        if data.get(key) is not None:
            data[key] = cls(**data[key])
    return ElasticObservation(**data)


def history_from_dict(data: dict) -> HistorySnapshot:
    return HistorySnapshot(
        tuple(BatchWorkload(**x) for x in data.get("workloads", ())),
        tuple(Measurement(**x) for x in data.get("measurements", ())),
    )


def forecast_from_dict(data: list[dict]) -> tuple[ForecastUpdate, ...]:
    return tuple(
        ForecastUpdate(BatchWorkload(**x["workload"]), x["wait_seconds"]) for x in data
    )


def plan_from_dict(data: list[dict]) -> tuple[PlanStep, ...]:
    result = []
    for step in data:
        if type(step.get("release_before", False)) is not bool:
            raise ValueError("release_before must be boolean")
        result.append(
            PlanStep(
                TrainingConfig(**step["config"]), step.get("release_before", False)
            )
        )
    return tuple(result)
