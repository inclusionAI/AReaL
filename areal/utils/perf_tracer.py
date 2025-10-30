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
    contextmanager,
)
from enum import Enum
from typing import Any

from areal.api.cli_args import PerfTracerConfig
from areal.utils import logging

logger = logging.getLogger("PerfTracer")

try:  # pragma: no cover - platform specific
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


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


@contextmanager
def _acquire_file_lock(path: str):
    """Best-effort advisory lock for cross-rank aggregation."""

    if fcntl is None:
        yield
        return

    lock_path = f"{path}.lock"
    parent = os.path.dirname(lock_path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)

    with open(lock_path, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


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


def _normalize_save_interval(value: int | None) -> int:
    try:
        interval = int(value) if value is not None else 1
    except (TypeError, ValueError):  # pragma: no cover - defensive
        interval = 1
    if interval <= 0:
        return 1
    return interval


class _NullContext(AbstractContextManager, AbstractAsyncContextManager):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, exc_tb):
        return False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, exc_tb):
        return False


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
        self._output_path = _default_trace_path(config)
        self._save_interval_steps = _normalize_save_interval(config.save_interval_steps)

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, flag: bool) -> None:
        self._enabled = flag

    def set_rank(self, rank: int) -> None:
        if rank < 0:
            raise ValueError("rank must be a non-negative integer")
        self._rank = rank

    def apply_config(self, config: PerfTracerConfig, *, rank: int) -> None:
        self._config = config
        self.set_rank(rank)
        self._output_path = _default_trace_path(config)
        self.set_enabled(config.enabled)
        self._save_interval_steps = _normalize_save_interval(config.save_interval_steps)

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
        if not self._enabled:
            return

        # Save only on the last step of each interval (0-indexed).
        # For example, if save_interval_steps=3, saves at steps 2, 5, 8, ...
        if (
            not force
            and step is not None
            and self._save_interval_steps > 1
            and ((step + 1) % self._save_interval_steps) != 0
        ):
            return

        with self._lock:
            if not self._events:
                return

            events = self._events
            serialized_events = [
                json.dumps(event, ensure_ascii=False) for event in events
            ]
            output_path = self._output_path

            try:
                with _acquire_file_lock(output_path):
                    with open(output_path, "a", encoding="utf-8") as fout:
                        for line in serialized_events:
                            fout.write(line)
                            fout.write("\n")
                        fout.flush()
                        os.fsync(fout.fileno())
                self._events = []
            except OSError as exc:  # pragma: no cover - depends on filesystem
                logger.error("Failed to append perf trace to %s: %s", output_path, exc)

    def reset(self) -> None:
        with self._lock:
            self._events = []
            self._thread_meta_emitted = set()
            self._process_meta_emitted = set()
            self._origin_ns = time.perf_counter_ns()
            self._enabled = self._config.enabled
            self._save_interval_steps = _normalize_save_interval(
                self._config.save_interval_steps
            )

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


def _default_trace_path(
    config: PerfTracerConfig,
    *,
    filename: str = "traces.jsonl",
) -> str:
    base_dir = os.path.join(
        os.path.expanduser(os.path.expandvars(config.fileroot)),
        "logs",
        getpass.getuser(),
        config.experiment_name,
        config.trial_name,
    )
    return os.path.join(base_dir, filename)


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


def configure(
    config: PerfTracerConfig,
    *,
    rank: int,
) -> PerfTracer:
    global GLOBAL_TRACER
    with _GLOBAL_TRACER_LOCK:
        if GLOBAL_TRACER is None:
            GLOBAL_TRACER = PerfTracer(config, rank=rank)
        else:
            GLOBAL_TRACER.apply_config(config, rank=rank)
        tracer = GLOBAL_TRACER
    return tracer


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
    "PerfTraceCategory",
    "Category",
    "GLOBAL_TRACER",
    "get_tracer",
    "configure",
    "reset",
    "trace_scope",
    "atrace_scope",
    "instant",
    "save",
]
