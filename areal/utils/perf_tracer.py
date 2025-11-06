from __future__ import annotations

import asyncio
import atexit
import functools
import getpass
import json
import os
import threading
import time
import warnings
from collections.abc import Callable
from contextlib import (
    AbstractAsyncContextManager,
    AbstractContextManager,
)
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, cast

from areal.api.cli_args import PerfTracerConfig, RequestTracerConfig
from areal.utils import logging

logger = logging.getLogger("PerfTracer")

# Context variable for storing request_id in async context
_current_request_id: ContextVar[int | None] = ContextVar("request_id", default=None)

# Suppress Pydantic warnings for standard dataclasses
# Pydantic may inspect all dataclasses even when not using pydantic.dataclasses
# and emit false warnings about field() parameters or frozen dataclasses
warnings.filterwarnings(
    "ignore",
    message=".*repr.*should be.*",
    category=UserWarning,
    module="pydantic",
)
warnings.filterwarnings(
    "ignore",
    message=".*frozen.*attribute.*provided to.*Field.*function.*",
    category=UserWarning,
    module="pydantic",
)


_THREAD_LOCAL = threading.local()


class PerfTraceCategory(str, Enum):
    COMPUTE = "compute"
    COMM = "comm"
    IO = "io"
    SYNC = "sync"
    SCHEDULER = "scheduler"
    INSTR = "instr"
    MISC = "misc"


Category = PerfTraceCategory


CategoryLike = PerfTraceCategory | str | None


_CATEGORY_ALIASES: dict[str, PerfTraceCategory] = {
    "compute": PerfTraceCategory.COMPUTE,
    "communication": PerfTraceCategory.COMM,
    "comm": PerfTraceCategory.COMM,
    "io": PerfTraceCategory.IO,
    "synchronization": PerfTraceCategory.SYNC,
    "sync": PerfTraceCategory.SYNC,
    "scheduling": PerfTraceCategory.SCHEDULER,
    "scheduler": PerfTraceCategory.SCHEDULER,
    "instrumentation": PerfTraceCategory.INSTR,
    "instr": PerfTraceCategory.INSTR,
    "misc": PerfTraceCategory.MISC,
}


_PERF_TRACE_FILENAME = "traces.jsonl"
_REQUEST_TRACE_FILENAME = "requests.jsonl"


class _NullContext(AbstractContextManager, AbstractAsyncContextManager):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, exc_tb):
        return False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, exc_tb):
        return False


def _rank_qualified_filename(filename: str, rank: int) -> str:
    root, ext = os.path.splitext(filename)
    return f"{root}-r{rank}{ext}"


def _maybe_duration(start: float | None, end: float | None) -> float | None:
    if start is None or end is None:
        return None
    return end - start


def _normalize_save_interval(config: PerfTracerConfig) -> int:
    return max(config.save_interval, 1)


def _normalize_flush_threshold(config: RequestTracerConfig) -> int:
    try:
        value = int(config.flush_threshold)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid flush_threshold=%r; defaulting to 1",
            getattr(config, "flush_threshold", None),
        )
        return 1
    return max(value, 1)


def _rollout_stats_to_dict(data: Any) -> dict[str, int] | None:
    if data is None:
        return None
    if isinstance(data, dict):
        converted: dict[str, int] = {}
        for key, value in data.items():
            try:
                converted[str(key)] = int(value)
            except (TypeError, ValueError):
                continue
        return converted or None
    attrs = ("accepted", "enqueued", "rejected", "running")
    converted: dict[str, int] = {}
    for attr in attrs:
        if hasattr(data, attr):
            try:
                converted[attr] = int(getattr(data, attr))
            except (TypeError, ValueError):
                continue
    if converted:
        return converted
    to_dict = getattr(data, "to_dict", None)
    if callable(to_dict):
        try:
            maybe = to_dict()
            return _rollout_stats_to_dict(maybe)
        except Exception:  # pragma: no cover - defensive
            return None
    return None


def _normalize_category(category: CategoryLike) -> str:
    if category is None:
        return PerfTraceCategory.MISC.value
    if isinstance(category, PerfTraceCategory):
        return category.value
    if isinstance(category, str) and category.strip():
        lowered = category.strip().lower()
        alias = _CATEGORY_ALIASES.get(lowered)
        if alias is not None:
            return alias.value
        return category
    return PerfTraceCategory.MISC.value


def _default_trace_path(
    config: PerfTracerConfig,
    *,
    rank: int,
    filename: str = _PERF_TRACE_FILENAME,
    subdir: str | None = None,
) -> str:
    base_dir = os.path.join(
        os.path.expanduser(os.path.expandvars(config.fileroot)),
        "logs",
        getpass.getuser(),
        config.experiment_name,
        config.trial_name,
    )
    if subdir:
        base_dir = os.path.join(base_dir, subdir)
    return os.path.join(base_dir, _rank_qualified_filename(filename, rank))


class RequestTraceEvent(str, Enum):
    ENQUEUED = "enqueued"
    EXECUTION_START = "execution_start"
    EXECUTION_END = "execution_end"
    CONSUMED = "consumed"
    GENERATE_START = "generate_start"
    GENERATE_END = "generate_end"
    REWARD_START = "reward_start"
    REWARD_END = "reward_end"
    DROPPED = "dropped"


@dataclass
class PhaseEntry:
    start_ts: float
    end_ts: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return {
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
        }


# NOTE: frozen=True is valid despite Pydantic warnings
@dataclass(frozen=True)
class PhaseConfig:
    name: str
    start_event: RequestTraceEvent
    end_event: RequestTraceEvent
    allow_multiple: bool = False
    ready_on_complete: bool = False
    on_end: Callable[[RequestRecord, dict[str, Any]], None] | None = None


# NOTE: frozen=True is valid despite Pydantic warnings
@dataclass(frozen=True)
class EventRule:
    timestamp_attr: str | None = None
    phase: str | None = None
    role: str | None = None
    allow_multiple: bool = False
    payload_handler: Callable[[RequestRecord, dict[str, Any]], None] | None = None
    mark_ready: bool = False


# NOTE: frozen=True is valid despite Pydantic warnings
@dataclass(frozen=True)
class RecordField:
    attr: str | None = None
    key: str | None = None
    compute: Callable[[RequestRecord], Any] | None = None
    omit_if_none: bool = True

    def resolve(self, record: RequestRecord) -> Any:
        if self.compute is not None:
            return self.compute(record)
        if self.attr is None:
            raise ValueError("RecordField requires either attr or compute")
        return getattr(record, self.attr)

    def key_name(self) -> str:
        if self.key is not None:
            return self.key
        if self.attr is None:
            raise ValueError("RecordField without attr must define key")
        return self.attr


@dataclass
class RequestRecord:
    request_id: int
    rank: int
    submit_ts: float
    status: str = "pending"
    rejection_reason: str | None = None
    rollout_stats: dict[str, int] | None = None
    enqueue_ts: float | None = None
    wait_return_ts: float | None = None
    phases: dict[str, list[PhaseEntry]] = field(init=False)
    counters: dict[str, int] = field(init=False)
    # NOTE: repr=False is valid for dataclasses.field() despite Pydantic warnings
    _active_phases: dict[str, PhaseEntry | None] = field(init=False, repr=False)

    PHASE_CONFIGS: ClassVar[tuple[PhaseConfig, ...]] = ()
    COUNTERS: ClassVar[tuple[str, ...]] = ()
    FIELD_SPECS: ClassVar[tuple[RecordField, ...]] = ()

    def __post_init__(self) -> None:
        self.phases = {cfg.name: [] for cfg in self.PHASE_CONFIGS}
        self._active_phases = {cfg.name: None for cfg in self.PHASE_CONFIGS}
        self.counters = {name: 0 for name in self.COUNTERS}

    @classmethod
    def default_phase_configs(cls) -> tuple[PhaseConfig, ...]:
        return (
            PhaseConfig(
                name="execution",
                start_event=RequestTraceEvent.EXECUTION_START,
                end_event=RequestTraceEvent.EXECUTION_END,
                on_end=cls._on_execution_end,
                ready_on_complete=True,
            ),
            PhaseConfig(
                name="generate",
                start_event=RequestTraceEvent.GENERATE_START,
                end_event=RequestTraceEvent.GENERATE_END,
                allow_multiple=True,
            ),
            PhaseConfig(
                name="reward",
                start_event=RequestTraceEvent.REWARD_START,
                end_event=RequestTraceEvent.REWARD_END,
                allow_multiple=True,
            ),
        )

    def is_ready_to_flush(self) -> bool:
        if self.status in {"rejected", "failed", "dropped"}:
            return True
        if self.status == "accepted" and self.wait_return_ts is not None:
            return True
        return False

    def increment_counter(self, name: str, value: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + value

    def apply_phase_event(
        self,
        phase: str,
        role: str,
        timestamp: float,
        *,
        allow_multiple: bool,
    ) -> None:
        entries = self.phases.setdefault(phase, [])
        current = self._active_phases.get(phase)
        if role == "start":
            if current is not None and current.end_ts is None and not allow_multiple:
                current.end_ts = timestamp
            entry = PhaseEntry(start_ts=timestamp)
            entries.append(entry)
            self._active_phases[phase] = entry
        elif role == "end":
            if current is None or current.end_ts is not None:
                entry = PhaseEntry(start_ts=timestamp)
                entries.append(entry)
            else:
                entry = current
            entry.end_ts = timestamp
            self._active_phases[phase] = None

    def _phase_first_start(self, phase: str) -> float | None:
        for entry in self.phases.get(phase, []):
            if entry.start_ts is not None:
                return entry.start_ts
        return None

    def _phase_first_end(self, phase: str) -> float | None:
        for entry in self.phases.get(phase, []):
            if entry.end_ts is not None:
                return entry.end_ts
        return None

    @staticmethod
    def _on_execution_end(record: RequestRecord, payload: dict[str, Any]) -> None:
        status = payload.get("status")
        if status is not None:
            record.status = status
        if "rejection_reason" in payload:
            record.rejection_reason = payload.get("rejection_reason")
        record.rollout_stats = _rollout_stats_to_dict(payload.get("rollout_stats"))

    @staticmethod
    def _on_drop(record: RequestRecord, payload: dict[str, Any]) -> None:
        record.status = "dropped"
        reason = payload.get("rejection_reason")
        if reason is not None:
            record.rejection_reason = reason
        record.rollout_stats = _rollout_stats_to_dict(payload.get("rollout_stats"))

    @classmethod
    def build_event_rules(cls) -> dict[RequestTraceEvent, EventRule]:
        rules: dict[RequestTraceEvent, EventRule] = {
            RequestTraceEvent.ENQUEUED: EventRule(timestamp_attr="enqueue_ts"),
            RequestTraceEvent.CONSUMED: EventRule(timestamp_attr="wait_return_ts"),
            RequestTraceEvent.DROPPED: EventRule(
                phase="execution",
                role="end",
                payload_handler=cls._on_drop,
                mark_ready=True,
            ),
        }
        for cfg in cls.PHASE_CONFIGS:
            rules[cfg.start_event] = EventRule(
                phase=cfg.name,
                role="start",
                allow_multiple=cfg.allow_multiple,
            )
            if cfg.end_event is not None:
                rules[cfg.end_event] = EventRule(
                    phase=cfg.name,
                    role="end",
                    payload_handler=cfg.on_end,
                    mark_ready=cfg.ready_on_complete,
                )
        return rules

    def _phase_total_duration(self, phase: str) -> float | None:
        durations = [
            entry.end_ts - entry.start_ts
            for entry in self.phases.get(phase, [])
            if entry.end_ts is not None
        ]
        if not durations:
            return None
        return sum(durations)

    @staticmethod
    def _compute_queue_wait(record: RequestRecord) -> float | None:
        return _maybe_duration(record.submit_ts, record.enqueue_ts)

    @staticmethod
    def _compute_runner_wait(record: RequestRecord) -> float | None:
        first_execution_start = record._phase_first_start("execution")
        return _maybe_duration(record.enqueue_ts, first_execution_start)

    @staticmethod
    def _compute_execution_time(record: RequestRecord) -> float | None:
        return record._phase_total_duration("execution")

    @staticmethod
    def _compute_post_wait(record: RequestRecord) -> float | None:
        first_execution_end = record._phase_first_end("execution")
        return _maybe_duration(first_execution_end, record.wait_return_ts)

    @staticmethod
    def _compute_total_time(record: RequestRecord) -> float | None:
        return _maybe_duration(record.submit_ts, record.wait_return_ts)

    @staticmethod
    def _compute_generate_time(record: RequestRecord) -> float | None:
        return record._phase_total_duration("generate")

    @staticmethod
    def _compute_reward_time(record: RequestRecord) -> float | None:
        return record._phase_total_duration("reward")

    @classmethod
    def default_field_specs(cls) -> tuple[RecordField, ...]:
        return (
            RecordField("request_id"),
            RecordField("rank"),
            RecordField("status"),
            RecordField("rejection_reason", omit_if_none=True),
            RecordField("submit_ts"),
            RecordField("enqueue_ts", omit_if_none=True),
            RecordField("wait_return_ts", omit_if_none=True),
            RecordField("rollout_stats", omit_if_none=True),
            RecordField(
                compute=cls._compute_queue_wait,
                key="queue_wait_s",
                omit_if_none=True,
            ),
            RecordField(
                compute=cls._compute_runner_wait,
                key="runner_wait_s",
                omit_if_none=True,
            ),
            RecordField(
                compute=cls._compute_execution_time,
                key="execution_s",
                omit_if_none=True,
            ),
            RecordField(
                compute=cls._compute_post_wait,
                key="post_wait_s",
                omit_if_none=True,
            ),
            RecordField(
                compute=cls._compute_total_time,
                key="total_s",
                omit_if_none=True,
            ),
            RecordField(
                compute=cls._compute_generate_time,
                key="generate_s",
                omit_if_none=True,
            ),
            RecordField(
                compute=cls._compute_reward_time,
                key="reward_calc_s",
                omit_if_none=True,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        for field_spec in self.FIELD_SPECS:
            value = field_spec.resolve(self)
            if field_spec.omit_if_none and value is None:
                continue
            data[field_spec.key_name()] = value
        if any(self.phases.values()):
            data["phases"] = {
                name: [entry.to_dict() for entry in entries]
                for name, entries in self.phases.items()
                if entries
            }
        if any(self.counters.values()):
            data["counters"] = {k: v for k, v in self.counters.items() if v}
        return data


RequestRecord.PHASE_CONFIGS = RequestRecord.default_phase_configs()
RequestRecord.FIELD_SPECS = RequestRecord.default_field_specs()
_REQUEST_EVENT_RULES = RequestRecord.build_event_rules()

_REQUEST_TRACE_METHOD_TO_EVENT: dict[str, RequestTraceEvent] = {
    "mark_enqueued": RequestTraceEvent.ENQUEUED,
    "mark_execution_start": RequestTraceEvent.EXECUTION_START,
    "mark_execution_end": RequestTraceEvent.EXECUTION_END,
    "mark_consumed": RequestTraceEvent.CONSUMED,
    "mark_dropped": RequestTraceEvent.DROPPED,
    "mark_generate_start": RequestTraceEvent.GENERATE_START,
    "mark_generate_end": RequestTraceEvent.GENERATE_END,
    "mark_reward_start": RequestTraceEvent.REWARD_START,
    "mark_reward_end": RequestTraceEvent.REWARD_END,
}


def trace_request_event(
    request_id: int | None,
    method: str,
    **payload: Any,
) -> None:
    if request_id is None:
        return
    tracer = get_request_tracer()
    if tracer is None:
        return
    if method == "increment_counter":
        name = payload.get("name")
        if not name:
            return
        tracer.increment_counter(request_id, name, payload.get("value", 1))
        return
    event = _REQUEST_TRACE_METHOD_TO_EVENT.get(method)
    if event is None:
        return
    tracer.record_event(request_id, event, **payload)


class _SyncRequestPhaseScope(AbstractContextManager[Any]):
    """Sync context manager for tracing request phases.

    Automatically calls mark_{phase}_start on enter and mark_{phase}_end on exit,
    ensuring proper pairing even if exceptions occur.
    """

    def __init__(
        self,
        request_id: int | None,
        phase: str,
        *,
        start_payload: dict[str, Any] | None = None,
        end_payload: dict[str, Any] | None = None,
    ) -> None:
        self._request_id = request_id
        self._phase = phase
        self._start_method = f"mark_{phase}_start"
        self._end_method = f"mark_{phase}_end"
        self._start_payload = start_payload or {}
        self._end_payload = end_payload or {}

    def __enter__(self) -> _SyncRequestPhaseScope:
        trace_request_event(self._request_id, self._start_method, **self._start_payload)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Always call end event, even if exception occurred
        trace_request_event(self._request_id, self._end_method, **self._end_payload)
        return False  # Don't suppress exceptions


class _AsyncRequestPhaseScope(AbstractAsyncContextManager[Any]):
    """Async context manager for tracing request phases.

    Automatically calls mark_{phase}_start on enter and mark_{phase}_end on exit,
    ensuring proper pairing even if exceptions occur.
    """

    def __init__(
        self,
        request_id: int | None,
        phase: str,
        *,
        start_payload: dict[str, Any] | None = None,
        end_payload: dict[str, Any] | None = None,
    ) -> None:
        self._request_id = request_id
        self._phase = phase
        self._start_method = f"mark_{phase}_start"
        self._end_method = f"mark_{phase}_end"
        self._start_payload = start_payload or {}
        self._end_payload = end_payload or {}

    async def __aenter__(self) -> _AsyncRequestPhaseScope:
        trace_request_event(self._request_id, self._start_method, **self._start_payload)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Always call end event, even if exception occurred
        trace_request_event(self._request_id, self._end_method, **self._end_payload)
        return False  # Don't suppress exceptions


def trace_request_phase(
    request_id: int | None,
    phase: str,
    *,
    start_payload: dict[str, Any] | None = None,
    end_payload: dict[str, Any] | None = None,
) -> AbstractContextManager[Any]:
    """Sync context manager for tracing request phases.

    Automatically pairs mark_{phase}_start and mark_{phase}_end events,
    ensuring they are always called together even if exceptions occur.

    Parameters
    ----------
    request_id : int | None
        The request ID to trace. If None, no tracing occurs.
    phase : str
        The phase name (e.g., "generate", "reward", "execution").
        Will call mark_{phase}_start and mark_{phase}_end.
    start_payload : dict[str, Any] | None
        Optional payload to pass to the start event.
    end_payload : dict[str, Any] | None
        Optional payload to pass to the end event.

    Returns
    -------
    AbstractContextManager
        A sync context manager for the phase tracing.

    Examples
    --------
    >>> with trace_request_phase(request_id, "generate"):
    ...     result = engine.generate(req)

    >>> with trace_request_phase(request_id, "reward"):
    ...     reward = reward_fn(prompt, completion)

    >>> with trace_request_phase(
    ...     request_id,
    ...     "execution",
    ...     end_payload={"status": "accepted"}
    ... ):
    ...     result = process_request()
    """
    if request_id is None:
        return _NullContext()
    return _SyncRequestPhaseScope(
        request_id,
        phase,
        start_payload=start_payload,
        end_payload=end_payload,
    )


def atrace_request_phase(
    request_id: int | None,
    phase: str,
    *,
    start_payload: dict[str, Any] | None = None,
    end_payload: dict[str, Any] | None = None,
) -> AbstractAsyncContextManager[Any]:
    """Async context manager for tracing request phases.

    Automatically pairs mark_{phase}_start and mark_{phase}_end events,
    ensuring they are always called together even if exceptions occur.

    Parameters
    ----------
    request_id : int | None
        The request ID to trace. If None, no tracing occurs.
    phase : str
        The phase name (e.g., "generate", "reward", "execution").
        Will call mark_{phase}_start and mark_{phase}_end.
    start_payload : dict[str, Any] | None
        Optional payload to pass to the start event.
    end_payload : dict[str, Any] | None
        Optional payload to pass to the end event.

    Returns
    -------
    AbstractAsyncContextManager
        An async context manager for the phase tracing.

    Examples
    --------
    >>> async with atrace_request_phase(request_id, "generate"):
    ...     result = await engine.agenerate(req)

    >>> async with atrace_request_phase(request_id, "reward"):
    ...     reward = await reward_fn(prompt, completion)

    >>> async with atrace_request_phase(
    ...     request_id,
    ...     "execution",
    ...     end_payload={"status": "accepted"}
    ... ):
    ...     result = await process_request()
    """
    if request_id is None:
        return _NullContext()
    return _AsyncRequestPhaseScope(
        request_id,
        phase,
        start_payload=start_payload,
        end_payload=end_payload,
    )


class RequestTracer:
    def __init__(
        self,
        config: RequestTracerConfig,
        *,
        output_path: str,
        rank: int,
    ) -> None:
        self._config = config
        self._rank = rank
        self._lock = threading.Lock()
        self._next_id = 0
        self._records: dict[int, RequestRecord] = {}
        self._ready: set[int] = set()
        self._flush_threshold = _normalize_flush_threshold(config)
        self._output_path = output_path
        self._event_rules = _REQUEST_EVENT_RULES

    def register_submission(self) -> int:
        now = time.perf_counter()
        with self._lock:
            request_id = self._next_id
            self._next_id += 1
            self._records[request_id] = RequestRecord(
                request_id=request_id,
                rank=self._rank,
                submit_ts=now,
            )
        return request_id

    def _apply_event(
        self,
        request_id: int,
        event: RequestTraceEvent,
        payload: dict[str, Any] | None = None,
    ) -> bool:
        rule = self._event_rules.get(event)
        if rule is None:
            return False
        data = dict(payload or {})
        with self._lock:
            record = self._records.get(request_id)
            if record is None:
                return False
            timestamp = time.perf_counter()
            if rule.timestamp_attr is not None:
                setattr(record, rule.timestamp_attr, timestamp)
            if rule.phase is not None and rule.role is not None:
                record.apply_phase_event(
                    rule.phase,
                    rule.role,
                    timestamp,
                    allow_multiple=rule.allow_multiple,
                )
            if rule.payload_handler is not None:
                rule.payload_handler(record, data)
            ready = rule.mark_ready or record.is_ready_to_flush()
            if ready:
                self._ready.add(request_id)
                if len(self._ready) >= self._flush_threshold:
                    return True
            return False

    def record_event(
        self,
        request_id: int,
        event: RequestTraceEvent,
        **payload: Any,
    ) -> None:
        should_flush = self._apply_event(request_id, event, payload)
        if should_flush:
            self.flush()

    def increment_counter(self, request_id: int, name: str, value: int = 1) -> None:
        with self._lock:
            record = self._records.get(request_id)
            if record is None:
                return
            record.increment_counter(name, value)
            if record.is_ready_to_flush():
                self._ready.add(request_id)

    def flush(self, force: bool = False) -> None:
        with self._lock:
            if force:
                candidate_ids = list(self._records.keys())
            else:
                candidate_ids = list(self._ready)
            if not candidate_ids:
                return

            to_flush: list[tuple[int, RequestRecord, bool]] = []
            for request_id in candidate_ids:
                record = self._records.get(request_id)
                if record is None:
                    self._ready.discard(request_id)
                    continue
                if not force and not record.is_ready_to_flush():
                    continue
                was_ready = request_id in self._ready
                to_flush.append((request_id, record, was_ready))

            if not to_flush:
                return

            for request_id, _, _ in to_flush:
                self._records.pop(request_id, None)
                self._ready.discard(request_id)

        try:
            payload = [record.to_dict() for (_, record, _) in to_flush]
            lines = [json.dumps(item, ensure_ascii=False) for item in payload]

            parent = os.path.dirname(self._output_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(self._output_path, "a", encoding="utf-8") as fout:
                for line in lines:
                    fout.write(f"{line}\n")
                fout.flush()
                os.fsync(fout.fileno())
        except (OSError, TypeError) as exc:  # pragma: no cover - depends on filesystem
            logger.error(
                "Failed to append request trace to %s: %s",
                self._output_path,
                exc,
            )
            with self._lock:
                for request_id, record, was_ready in to_flush:
                    self._records[request_id] = record
                    if was_ready:
                        self._ready.add(request_id)

    def reset(self) -> None:
        self.flush(force=True)
        with self._lock:
            self._records.clear()
            self._ready.clear()
            self._next_id = 0
            self._flush_threshold = _normalize_flush_threshold(self._config)


class _Scope(AbstractContextManager[Any]):
    def __init__(
        self,
        tracer: PerfTracer,
        name: str,
        *,
        category: str,
        args: dict[str, Any] | None,
    ) -> None:
        self._tracer = tracer
        self._name = name
        self._category = category
        self._args = args
        self._start_ns: int | None = None

    def __enter__(self) -> _Scope:
        self._start_ns = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc, exc_tb):
        if self._start_ns is None:
            return False
        duration_ns = time.perf_counter_ns() - self._start_ns
        args = dict(self._args or {})
        if exc_type is not None:
            args.setdefault("exception", exc_type.__name__)
        self._tracer._record_complete(
            self._name,
            self._start_ns,
            duration_ns,
            category=self._category,
            args=args,
        )
        return False


class _AsyncScope(AbstractAsyncContextManager[Any]):
    def __init__(
        self,
        tracer: PerfTracer,
        name: str,
        *,
        category: str,
        args: dict[str, Any] | None,
    ) -> None:
        self._scope = _Scope(tracer, name, category=category, args=args)

    async def __aenter__(self) -> _AsyncScope:
        self._scope.__enter__()
        return self

    async def __aexit__(self, exc_type, exc, exc_tb):
        return self._scope.__exit__(exc_type, exc, exc_tb)


def _thread_id() -> int:
    cached = getattr(_THREAD_LOCAL, "tid", None)
    if cached is not None:
        return cached
    try:
        tid = threading.get_native_id()
    except AttributeError:  # pragma: no cover - Python <3.8 fallback
        tid = threading.get_ident()
    _THREAD_LOCAL.tid = tid
    return tid


class PerfTracer:
    """A lightweight tracer that emits Chrome Trace compatible JSON."""

    def __init__(self, config: PerfTracerConfig, *, rank: int) -> None:
        if rank < 0:
            raise ValueError("rank must be a non-negative integer")
        self._config = config
        self._enabled = config.enabled
        self._rank = rank
        self._events: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._pid = os.getpid()
        self._origin_ns = time.perf_counter_ns()
        self._thread_meta_emitted: set[int] = set()
        self._process_meta_emitted: set[int] = set()
        self._output_path = _default_trace_path(
            config,
            rank=rank,
            subdir="perf_tracer",
        )
        self._save_interval = _normalize_save_interval(config)
        self._request_tracer: RequestTracer | None = None
        self._configure_request_tracer(config, rank=rank)

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, flag: bool) -> None:
        self._enabled = flag

    def _configure_request_tracer(self, config: PerfTracerConfig, *, rank: int) -> None:
        request_cfg = getattr(config, "request_tracer", None)
        enabled = bool(request_cfg and getattr(request_cfg, "enabled", False))
        if enabled:
            request_cfg = cast(RequestTracerConfig, request_cfg)
            output_path = _default_trace_path(
                config,
                filename=_REQUEST_TRACE_FILENAME,
                rank=rank,
                subdir="request_tracer",
            )
            if self._request_tracer is None:
                self._request_tracer = RequestTracer(
                    request_cfg,
                    output_path=output_path,
                    rank=rank,
                )
            else:
                raise RuntimeError("Request tracer is already configured")
        else:
            if self._request_tracer is not None:
                self._request_tracer.flush(force=True)
            self._request_tracer = None

    @property
    def request_tracer(self) -> RequestTracer | None:
        return self._request_tracer

    # ------------------------------------------------------------------
    # Core recording API
    # ------------------------------------------------------------------
    def trace_scope(
        self,
        name: str,
        *,
        category: CategoryLike = None,
        args: dict[str, Any] | None = None,
    ) -> AbstractContextManager[Any]:
        if not self._enabled:
            return _NullContext()
        return _Scope(
            self,
            name,
            category=_normalize_category(category),
            args=args,
        )

    def atrace_scope(
        self,
        name: str,
        *,
        category: CategoryLike = None,
        args: dict[str, Any] | None = None,
    ) -> AbstractAsyncContextManager[Any]:
        if not self._enabled:
            return _NullContext()
        return _AsyncScope(
            self,
            name,
            category=_normalize_category(category),
            args=args,
        )

    def instant(
        self,
        name: str,
        *,
        category: CategoryLike = None,
        args: dict[str, Any] | None = None,
    ) -> None:
        if not self._enabled:
            return
        self._record_event(
            {
                "name": name,
                "ph": "i",
                "ts": self._now_us(),
                "pid": self._pid,
                "tid": _thread_id(),
                "cat": _normalize_category(category),
                "args": args or {},
                "s": "t",
            }
        )

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save(self, *, step: int | None = None, force: bool = False) -> None:
        if self._request_tracer is not None and force:
            self._request_tracer.flush(force=True)

        if not self._enabled:
            return

        # Save only on the last step of each interval (0-indexed).
        # For example, if save_interval=3, saves at steps 2, 5, 8, ...
        interval = self._save_interval
        if (
            not force
            and step is not None
            and interval > 1
            and ((step + 1) % interval) != 0
        ):
            return

        with self._lock:
            if not self._events:
                return
            events_to_write: list[dict[str, Any]] = self._events
            self._events = []

        try:
            serialized_events = [
                json.dumps(event, ensure_ascii=False) for event in events_to_write
            ]
            output_path = self._output_path

            parent = os.path.dirname(output_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(output_path, "a", encoding="utf-8") as fout:
                for line in serialized_events:
                    fout.write(f"{line}\n")
                fout.flush()
                os.fsync(fout.fileno())
        except (OSError, TypeError) as exc:  # pragma: no cover - depends on filesystem
            logger.error("Failed to append perf trace to %s: %s", output_path, exc)
            with self._lock:
                self._events[0:0] = events_to_write

    def reset(self) -> None:
        if self._request_tracer is not None:
            self._request_tracer.reset()
        with self._lock:
            self._events = []
            self._thread_meta_emitted = set()
            self._process_meta_emitted = set()
            self._origin_ns = time.perf_counter_ns()
            self._enabled = self._config.enabled
            self._save_interval = _normalize_save_interval(self._config)

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _record_complete(
        self,
        name: str,
        start_ns: int,
        duration_ns: int,
        *,
        category: str,
        args: dict[str, Any] | None,
    ) -> None:
        event = {
            "name": name,
            "ph": "X",
            "ts": self._relative_us(start_ns),
            # Chrome trace viewers drop complete events whose duration rounds to 0 µs,
            # so clamp to 1 µs to keep sub-microsecond spans visible.
            "dur": max(duration_ns // 1000, 1),
            "pid": self._pid,
            "tid": _thread_id(),
            "cat": category,
            "args": args or {},
        }
        self._record_event(event)

    def _record_event(self, event: dict[str, Any]) -> None:
        if not self._enabled:
            return
        tid = event.get("tid")
        if isinstance(tid, int):
            self._ensure_thread_metadata(tid)
        event["pid"] = self._pid
        self._ensure_process_metadata(self._pid)
        if event.get("ph") != "M":
            args = event.setdefault("args", {})
            args.setdefault("rank", self._rank)
        with self._lock:
            self._events.append(event)

    def _ensure_thread_metadata(self, tid: int) -> None:
        if tid in self._thread_meta_emitted:
            return
        self._thread_meta_emitted.add(tid)
        thread_name = threading.current_thread().name
        meta_event = {
            "name": "thread_name",
            "ph": "M",
            "pid": self._pid,
            "tid": tid,
            "args": {"name": thread_name},
        }
        with self._lock:
            self._events.append(meta_event)

    def _ensure_process_metadata(self, pid: int) -> None:
        if pid in self._process_meta_emitted:
            return
        self._process_meta_emitted.add(pid)
        rank_label = f"Rank {self._rank}, Process"
        process_name_event = {
            "name": "process_name",
            "ph": "M",
            "pid": pid,
            "args": {"name": rank_label},
        }
        sort_event = {
            "name": "process_sort_index",
            "ph": "M",
            "pid": pid,
            "args": {"sort_index": self._rank},
        }
        with self._lock:
            self._events.extend([process_name_event, sort_event])

    def _now_us(self) -> int:
        return self._relative_us(time.perf_counter_ns())

    def _relative_us(self, ts_ns: int) -> int:
        return max((ts_ns - self._origin_ns) // 1000, 0)


GLOBAL_TRACER: PerfTracer | None = None
_GLOBAL_TRACER_LOCK = threading.Lock()


def _save_at_exit() -> None:
    tracer = GLOBAL_TRACER
    if tracer is None or not tracer.enabled:
        return
    try:
        tracer.save(force=True)
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to flush perf trace on exit: %s", exc, exc_info=True)


atexit.register(_save_at_exit)


# ----------------------------------------------------------------------
# Module-level convenience functions
# ----------------------------------------------------------------------
def _require_configured_tracer() -> PerfTracer:
    tracer = GLOBAL_TRACER
    if tracer is None:
        raise RuntimeError(
            "PerfTracer is not configured. Call perf_tracer.configure(...) first."
        )
    return tracer


def get_tracer() -> PerfTracer:
    return _require_configured_tracer()


def get_request_tracer() -> RequestTracer | None:
    tracer = GLOBAL_TRACER
    if tracer is None:
        return None
    return tracer.request_tracer


def configure(
    config: PerfTracerConfig,
    *,
    rank: int,
) -> PerfTracer:
    global GLOBAL_TRACER
    with _GLOBAL_TRACER_LOCK:
        if GLOBAL_TRACER is not None:
            raise RuntimeError(
                "PerfTracer has already been configured. Call perf_tracer.reset() "
                "before configuring again."
            )
        GLOBAL_TRACER = PerfTracer(config, rank=rank)
        logger.info(
            "Configured global PerfTracer: enabled=%s, request_tracing=%s, rank=%s",
            GLOBAL_TRACER.enabled,
            GLOBAL_TRACER.request_tracer is not None,
            rank,
        )
        return GLOBAL_TRACER


def reset() -> None:
    """Clear the global tracer so the next configure() call reinitializes it."""
    global GLOBAL_TRACER
    with _GLOBAL_TRACER_LOCK:
        tracer = GLOBAL_TRACER
        GLOBAL_TRACER = None
    if tracer is not None:
        tracer.reset()


def trace_scope(
    name: str,
    *,
    category: CategoryLike = None,
    args: dict[str, Any] | None = None,
):
    tracer = GLOBAL_TRACER
    if tracer is None:
        return _NullContext()
    return tracer.trace_scope(name, category=category, args=args)


def atrace_scope(
    name: str,
    *,
    category: CategoryLike = None,
    args: dict[str, Any] | None = None,
):
    tracer = GLOBAL_TRACER
    if tracer is None:
        return _NullContext()
    return tracer.atrace_scope(name, category=category, args=args)


def instant(
    name: str,
    *,
    category: CategoryLike = None,
    args: dict[str, Any] | None = None,
) -> None:
    tracer = GLOBAL_TRACER
    if tracer is None:
        return
    tracer.instant(name, category=category, args=args)


def save(*, step: int | None = None, force: bool = False) -> None:
    tracer = GLOBAL_TRACER
    if tracer is None:
        return
    tracer.save(step=step, force=force)


def trace_perf(name: str, *, category: CategoryLike = None):
    """
    Decorator for tracing function performance with PerfTracer.

    Automatically creates a trace scope around the entire function execution.
    Works with both sync and async functions.

    Parameters
    ----------
    name : str
        Trace name to display in the trace viewer.
    category : CategoryLike, optional
        Trace category (compute, io, comm, sync, scheduler, etc.).

    Examples
    --------
    >>> @trace_perf("ppo_update", category="compute")
    ... async def update_model(self, batch):
    ...     loss = compute_loss(batch)
    ...     loss.backward()
    ...     return loss

    >>> @trace_perf("save_checkpoint", category="io")
    ... def save(self, path):
    ...     torch.save(self.state_dict(), path)
    """

    def decorator(func):
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                async with atrace_scope(name, category=category):
                    return await func(*args, **kwargs)

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                with trace_scope(name, category=category):
                    return func(*args, **kwargs)

            return sync_wrapper

    return decorator


def set_request_id(request_id: int | None) -> None:
    """
    Set the request_id for the current async context.

    This should be called once at the entry point of each request/task.
    All subsequent async calls within the same context will have access
    to this request_id.

    Parameters
    ----------
    request_id : int | None
        The request ID to set in the current context.

    Examples
    --------
    >>> async def execute_request(data):
    ...     request_id = submit_request(data)
    ...     perf_tracer.set_request_id(request_id)  # Set once
    ...     result = await workflow.arun_episode(...)  # All children inherit
    ...     return result
    """
    _current_request_id.set(request_id)


def get_request_id() -> int | None:
    """
    Get the request_id from the current async context.

    Returns the request_id that was set via set_request_id(),
    or None if no request_id has been set.

    Returns
    -------
    int | None
        The request ID from the current context, or None.

    Examples
    --------
    >>> request_id = perf_tracer.get_request_id()
    >>> if request_id is not None:
    ...     print(f"Processing request {request_id}")
    """
    return _current_request_id.get()


def trace_request(phase: str):
    """
    Decorator for tracing request phases using contextvars.

    Automatically reads request_id from the async context (set via
    set_request_id) and traces the phase execution.

    Parameters
    ----------
    phase : str
        Phase name (e.g., "generate", "reward", "execution").
        Will call mark_{phase}_start and mark_{phase}_end.

    Examples
    --------
    >>> # Set context at entry point
    >>> async def arun_episode(self, engine, task_input):
    ...     perf_tracer.set_request_id(task_input.request_id)
    ...     resps = await self._do_generate(engine, req, n_samples)
    ...     results = await self._compute_rewards(resps)
    ...     return results

    >>> # Use decorator on methods - no need to pass request_id!
    >>> @trace_request("generate")
    ... async def _do_generate(self, engine, req, n_samples):
    ...     return await asyncio.gather(...)

    >>> @trace_request("reward")
    ... async def _compute_rewards(self, resps):
    ...     for resp in resps:
    ...         reward = await self.async_reward_fn(...)
    ...     return results
    """

    def decorator(func):
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                request_id = get_request_id()
                trace_request_event(request_id, f"mark_{phase}_start")
                try:
                    return await func(*args, **kwargs)
                finally:
                    trace_request_event(request_id, f"mark_{phase}_end")

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                request_id = get_request_id()
                trace_request_event(request_id, f"mark_{phase}_start")
                try:
                    return func(*args, **kwargs)
                finally:
                    trace_request_event(request_id, f"mark_{phase}_end")

            return sync_wrapper

    return decorator


__all__ = [
    "PerfTracer",
    "RequestTracer",
    "PerfTraceCategory",
    "Category",
    "RequestTraceEvent",
    "trace_request_event",
    "trace_request_phase",
    "atrace_request_phase",
    "trace_perf",
    "trace_request",
    "set_request_id",
    "get_request_id",
    "GLOBAL_TRACER",
    "get_tracer",
    "get_request_tracer",
    "configure",
    "reset",
    "trace_scope",
    "atrace_scope",
    "instant",
    "save",
]
