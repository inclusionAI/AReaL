from __future__ import annotations

import atexit
import getpass
import json
import os
import threading
import time
from contextlib import (
    AbstractAsyncContextManager,
    AbstractContextManager,
)
from dataclasses import dataclass
from enum import Enum
from typing import Any

from areal.api.cli_args import PerfTracerConfig, RequestTracerConfig
from areal.utils import logging

logger = logging.getLogger("PerfTracer")


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


def _rank_qualified_filename(filename: str, rank: int) -> str:
    root, ext = os.path.splitext(filename)
    return f"{root}-r{rank}{ext}"


def _maybe_duration(start: float | None, end: float | None) -> float | None:
    if start is None or end is None:
        return None
    return end - start


def _normalize_save_interval(config: PerfTracerConfig) -> int:
    return max(config.save_interval, 1)


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


def _normalize_flush_threshold(config: RequestTracerConfig) -> int:
    return max(config.flush_threshold, 1)


def _staleness_to_dict(stats: Any) -> dict[str, int] | None:
    if stats is None:
        return None
    if isinstance(stats, dict):
        return {
            "submitted": int(stats.get("submitted", 0)),
            "running": int(stats.get("running", 0)),
            "accepted": int(stats.get("accepted", 0)),
        }
    submitted = getattr(stats, "submitted", None)
    running = getattr(stats, "running", None)
    accepted = getattr(stats, "accepted", None)
    if submitted is None and running is None and accepted is None:
        return None
    return {
        "submitted": int(submitted or 0),
        "running": int(running or 0),
        "accepted": int(accepted or 0),
    }


@dataclass
class RequestRecord:
    request_id: int
    rank: int
    submit_ts: float
    enqueue_ts: float | None = None
    execute_start_ts: float | None = None
    execute_end_ts: float | None = None
    wait_return_ts: float | None = None
    status: str = "pending"
    should_accept: bool | None = None
    rejection_reason: str | None = None
    staleness: dict[str, int] | None = None

    def is_ready_to_flush(self) -> bool:
        if self.status in {"rejected", "failed", "dropped"}:
            return True
        if self.status == "accepted" and self.wait_return_ts is not None:
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "request_id": self.request_id,
            "rank": self.rank,
            "status": self.status,
            "should_accept": self.should_accept,
            "rejection_reason": self.rejection_reason,
            "submit_ts": self.submit_ts,
            "enqueue_ts": self.enqueue_ts,
            "execute_start_ts": self.execute_start_ts,
            "execute_end_ts": self.execute_end_ts,
            "wait_return_ts": self.wait_return_ts,
            "staleness": self.staleness,
        }
        data["queue_wait_s"] = _maybe_duration(self.submit_ts, self.enqueue_ts)
        data["runner_wait_s"] = _maybe_duration(self.enqueue_ts, self.execute_start_ts)
        data["execution_s"] = _maybe_duration(
            self.execute_start_ts, self.execute_end_ts
        )
        data["post_wait_s"] = _maybe_duration(self.execute_end_ts, self.wait_return_ts)
        data["total_s"] = _maybe_duration(self.submit_ts, self.wait_return_ts)
        return data


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

    def mark_enqueued(self, request_id: int) -> None:
        with self._lock:
            record = self._records.get(request_id)
            if record is None:
                return
            record.enqueue_ts = time.perf_counter()

    def mark_execution_start(self, request_id: int) -> None:
        with self._lock:
            record = self._records.get(request_id)
            if record is None:
                return
            record.execute_start_ts = time.perf_counter()

    def mark_execution_end(
        self,
        request_id: int,
        *,
        status: str,
        should_accept: bool | None,
        rejection_reason: str | None = None,
        staleness: Any = None,
    ) -> None:
        now = time.perf_counter()
        should_flush = False
        with self._lock:
            record = self._records.get(request_id)
            if record is None:
                return
            record.execute_end_ts = now
            record.status = status
            record.should_accept = should_accept
            record.rejection_reason = rejection_reason
            record.staleness = _staleness_to_dict(staleness)
            if record.is_ready_to_flush():
                self._ready.add(request_id)
                if len(self._ready) >= self._flush_threshold:
                    should_flush = True
        if should_flush:
            self.flush()

    def mark_consumed(self, request_id: int) -> None:
        should_flush = False
        with self._lock:
            record = self._records.get(request_id)
            if record is None:
                return
            record.wait_return_ts = time.perf_counter()
            if record.is_ready_to_flush():
                self._ready.add(request_id)
                if len(self._ready) >= self._flush_threshold:
                    should_flush = True
        if should_flush:
            self.flush()

    def mark_dropped(self, request_id: int, reason: str) -> None:
        should_flush = False
        with self._lock:
            record = self._records.get(request_id)
            if record is None:
                return
            record.execute_end_ts = time.perf_counter()
            record.status = "dropped"
            record.rejection_reason = reason
            self._ready.add(request_id)
            if len(self._ready) >= self._flush_threshold:
                should_flush = True
        if should_flush:
            self.flush()

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

        payload = [record.to_dict() for (_, record, _) in to_flush]
        lines = [json.dumps(item, ensure_ascii=False) for item in payload]

        try:
            parent = os.path.dirname(self._output_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(self._output_path, "a", encoding="utf-8") as fout:
                for line in lines:
                    fout.write(f"{line}\n")
                fout.flush()
                os.fsync(fout.fileno())
        except OSError as exc:  # pragma: no cover - depends on filesystem
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


class _NullContext(AbstractContextManager, AbstractAsyncContextManager):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, exc_tb):
        return False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, exc_tb):
        return False


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

        serialized_events = [
            json.dumps(event, ensure_ascii=False) for event in events_to_write
        ]
        output_path = self._output_path

        try:
            parent = os.path.dirname(output_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(output_path, "a", encoding="utf-8") as fout:
                for line in serialized_events:
                    fout.write(f"{line}\n")
                fout.flush()
                os.fsync(fout.fileno())
        except OSError as exc:  # pragma: no cover - depends on filesystem
            logger.error("Failed to append perf trace to %s: %s", output_path, exc)
            with self._lock:
                self._events = events_to_write + self._events

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


__all__ = [
    "PerfTracer",
    "RequestTracer",
    "PerfTraceCategory",
    "Category",
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
