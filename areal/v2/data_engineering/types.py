# SPDX-License-Identifier: Apache-2.0

"""Execution contracts; domain payloads and policies belong to recipes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

JsonObject = dict[str, Any]


def positive_int(value: int, name: str, *, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class PublicTask:
    """The only task representation passed to a calibration backend."""

    task_id: str
    input: JsonObject
    metadata: JsonObject = field(default_factory=dict)


@dataclass(frozen=True)
class DataSample:
    """A public task and its private reference; not a generated trajectory."""

    task: PublicTask
    reference: Any


@dataclass(frozen=True)
class Attempt:
    """One backend response; execution failures are distinct from wrong answers."""

    task_id: str
    repeat: int
    response: Any = None
    failure_reason: str | None = None


@dataclass(frozen=True)
class CalibrationConfig:
    """Calibration factory, repetition count, and implementation-owned parameters."""

    implementation: str
    repeats: int = 2
    params: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("implementation",):
            if (
                not isinstance(getattr(self, name), str)
                or not getattr(self, name).strip()
            ):
                raise ValueError(f"calibration.{name} must be nonempty")
        positive_int(self.repeats, "calibration.repeats")
        if not isinstance(self.params, dict):
            raise ValueError("calibration.params must be an object")


@dataclass(frozen=True)
class DataJobConfig:
    """Bounded, foreground data job; recipe parameters define domain behavior.

    Input and exclusion paths are files. Exclusions are public-task JSONL files
    with task_id fields, including a prior snapshot's public.jsonl.
    """

    recipe: str
    target_count: int
    purpose: Literal["harness", "model", "joint"] = "joint"
    seed: int = 0
    params: JsonObject = field(default_factory=dict)
    inputs: dict[str, str] = field(default_factory=dict)
    exclude: tuple[str, ...] = ()
    calibration: CalibrationConfig | None = None
    max_batches: int = 1
    max_candidates: int = 8192

    def __post_init__(self) -> None:
        if not isinstance(self.recipe, str) or not self.recipe.strip():
            raise ValueError("recipe must be a nonempty module:callable")
        if self.purpose not in {"harness", "model", "joint"}:
            raise ValueError("purpose must be harness, model, or joint")
        positive_int(self.target_count, "target_count")
        positive_int(self.seed, "seed", minimum=0)
        positive_int(self.max_batches, "max_batches")
        positive_int(self.max_candidates, "max_candidates")
        if self.target_count > self.max_candidates:
            raise ValueError("target_count exceeds max_candidates")
        if not isinstance(self.params, dict) or not isinstance(self.inputs, dict):
            raise ValueError("params and inputs must be objects")
        if any(
            not isinstance(k, str) or not isinstance(v, str)
            for k, v in self.inputs.items()
        ):
            raise ValueError("inputs must map names to file paths")
        if not isinstance(self.exclude, (list, tuple)) or any(
            not isinstance(p, str) for p in self.exclude
        ):
            raise ValueError("exclude must be a sequence of file paths")
        if self.calibration is not None and not isinstance(
            self.calibration, CalibrationConfig
        ):
            raise ValueError("calibration must be a CalibrationConfig")

    @classmethod
    def from_dict(
        cls, data: JsonObject, *, base_dir: Path | None = None
    ) -> DataJobConfig:
        values = dict(data)
        if not isinstance(values.get("inputs", {}), dict):
            raise ValueError("inputs must be an object")
        if not isinstance(values.get("exclude", ()), (list, tuple)):
            raise ValueError("exclude must be a sequence of file paths")
        if values.get("calibration") is not None:
            values["calibration"] = CalibrationConfig(**values["calibration"])
        base = base_dir or Path.cwd()

        def resolve(path: str) -> str:
            value = Path(path).expanduser()
            return str(
                (base / value).resolve() if not value.is_absolute() else value.resolve()
            )

        values["inputs"] = {k: resolve(v) for k, v in values.get("inputs", {}).items()}
        values["exclude"] = tuple(resolve(p) for p in values.get("exclude", ()))
        return cls(**values)


@dataclass(frozen=True)
class RecipeContext:
    seed: int
    target_count: int
    purpose: str
    inputs: Mapping[str, Path]
    batch_index: int = 0


@dataclass(frozen=True)
class CalibrationContext:
    repeats: int
    batch_index: int


@dataclass(frozen=True)
class CalibrationBatch:
    """Responses and implementation-specific provenance, without required agent bindings."""

    attempts: tuple[Attempt, ...]
    provenance: JsonObject = field(default_factory=dict)


@dataclass(frozen=True)
class Selection:
    task_ids: tuple[str, ...]
    statistics: JsonObject = field(default_factory=dict)


class InsufficientDataError(ValueError):
    """Only this exception asks the executor to generate another batch."""


class DataRecipe(Protocol):
    """Domain policy executed by run_data_job; all payloads must serialize to JSON.

    Factories return an object implementing these methods. Imported helper code
    and additional files can be declared with an optional source_files() method.
    Selection and scoring must not mutate retained samples.
    """

    def generate(self, context: RecipeContext) -> Iterable[DataSample]: ...
    def clean(self, sample: DataSample) -> DataSample | None: ...
    def score(self, sample: DataSample, response: Any) -> float: ...
    def select(
        self,
        samples: Sequence[DataSample],
        scores: Mapping[str, tuple[float | None, ...]] | None,
        context: RecipeContext,
    ) -> Selection: ...
    def training_row(self, sample: DataSample) -> JsonObject: ...


class Calibrator(Protocol):
    """Process public tasks using an implementation-owned calibration method.

    References and scoring stay with the recipe executor. Return a response or
    explicit failure for every task/repeat pair, including failed executions.
    Backend-specific dependencies and version checks belong to the implementation.
    """

    async def __call__(
        self, tasks: Sequence[PublicTask], context: CalibrationContext
    ) -> CalibrationBatch: ...
