from __future__ import annotations

import atexit
import json
import os
import threading
import time
from contextlib import (
    AbstractAsyncContextManager,
    AbstractContextManager,
    contextmanager,
)
from typing import Any

from areal.utils import logging

logger = logging.getLogger(__name__)

try:  # pragma: no cover - platform specific
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


_THREAD_LOCAL = threading.local()


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


def _load_existing_trace(path: str) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as fin:
            return json.load(fin)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        logger.warning("Existing perf trace at %s is invalid; overwriting", path)
        return {}


def _strtobool(value: str | None) -> bool:
    if value is None:
        return False
    return value.lower() in {"1", "true", "t", "yes", "y", "on"}


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

    def __init__(
        self,
        *,
        enabled: bool = False,
        output_path: str | None = None,
        rank: int | None = None,
        aggregate: bool = False,
    ) -> None:
        self._enabled = enabled
        self._output_path: str | None = None
        self._rank = rank
        self._aggregate = aggregate
        self._events: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._pid = os.getpid()
        self._origin_ns = time.perf_counter_ns()
        self._thread_meta_emitted: set[int] = set()
        self._process_meta_emitted: set[int] = set()
        self._user_output_path: str | None = None
        if output_path:
            self.set_output(output_path, rank=rank)

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, flag: bool) -> None:
        self._enabled = bool(flag)

    def set_aggregate(self, flag: bool) -> None:
        aggregate = bool(flag)
        if self._aggregate == aggregate:
            return
        self._aggregate = aggregate
        if self._user_output_path is not None:
            # Re-resolve the output path to adjust rank suffix usage
            self.set_output(self._user_output_path, rank=self._rank)

    def set_output(
        self,
        path: str,
        *,
        rank: int | None = None,
        make_parent: bool = True,
    ) -> str:
        if not path:
            raise ValueError("path must be a non-empty string")

        self._user_output_path = path
        path = os.path.expanduser(path)

        if os.path.isdir(path):
            base_path = os.path.join(path, "perf_trace.json")
        else:
            base_path = path

        if rank is not None:
            self._rank = rank

        rank_to_use = self._rank

        use_rank_suffix = rank_to_use is not None and not self._aggregate
        final_path = (
            self._apply_rank_suffix(base_path, rank_to_use)
            if use_rank_suffix
            else base_path
        )
        parent = os.path.dirname(final_path)
        if parent and make_parent:
            os.makedirs(parent, exist_ok=True)
        self._output_path = final_path
        return final_path

    @staticmethod
    def _apply_rank_suffix(path: str, rank: int | None) -> str:
        if rank is None:
            return path
        root, ext = os.path.splitext(path)
        suffix = f".rank{rank}"
        return f"{root}{suffix}{ext}" if ext else f"{root}{suffix}"

    # ------------------------------------------------------------------
    # Core recording API
    # ------------------------------------------------------------------
    def trace_scope(
        self,
        name: str,
        *,
        category: str | None = None,
        args: dict[str, Any] | None = None,
    ) -> AbstractContextManager[Any]:
        if not self._enabled:
            return _NullContext()
        return _Scope(self, name, category=category, args=args)

    def atrace_scope(
        self,
        name: str,
        *,
        category: str | None = None,
        args: dict[str, Any] | None = None,
    ) -> AbstractAsyncContextManager[Any]:
        if not self._enabled:
            return _NullContext()
        return _AsyncScope(self, name, category=category, args=args)

    def instant(
        self,
        name: str,
        *,
        category: str | None = None,
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
                "cat": category or "general",
                "args": args or {},
                "s": "t",
            }
        )

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save(self, path: str | None = None, *, reset: bool = True) -> str | None:
        if not self._enabled:
            return None

        with self._lock:
            events = list(self._events)
            if not events:
                return None

        output_path = path
        if output_path is not None:
            output_path = self.set_output(output_path)
        elif self._output_path is None:
            logger.warning(
                "PerfTracer.save called without an output path; skipping write"
            )
            return None
        else:
            output_path = self._output_path

        payload = {
            "traceEvents": events,
            "displayTimeUnit": "ms",
        }

        try:
            if self._aggregate:
                with _acquire_file_lock(output_path):
                    existing = _load_existing_trace(output_path)
                    combined = dict(existing) if existing else {}
                    existing_events = list(combined.get("traceEvents", []))
                    existing_events.extend(events)
                    combined["traceEvents"] = existing_events
                    combined.setdefault("displayTimeUnit", "ms")
                    with open(output_path, "w", encoding="utf-8") as fout:
                        json.dump(combined, fout, ensure_ascii=False)
            else:
                with open(output_path, "w", encoding="utf-8") as fout:
                    json.dump(payload, fout, ensure_ascii=False)
        except OSError as exc:  # pragma: no cover - depends on filesystem
            logger.error("Failed to write perf trace to %s: %s", output_path, exc)
            return None

        if reset:
            self.reset()
        return output_path

    def reset(self) -> None:
        with self._lock:
            self._events = []
            self._thread_meta_emitted = set()
            self._process_meta_emitted = set()
            self._origin_ns = time.perf_counter_ns()

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _record_complete(
        self,
        name: str,
        start_ns: int,
        duration_ns: int,
        *,
        category: str | None,
        args: dict[str, Any] | None,
    ) -> None:
        event = {
            "name": name,
            "ph": "X",
            "ts": self._relative_us(start_ns),
            "dur": max(duration_ns // 1000, 1),
            "pid": self._pid,
            "tid": _thread_id(),
            "cat": category or "general",
            "args": args or {},
        }
        self._record_event(event)

    def _record_event(self, event: dict[str, Any]) -> None:
        if not self._enabled:
            return
        tid = event.get("tid")
        if isinstance(tid, int):
            self._ensure_thread_metadata(tid)
        if self._aggregate and self._rank is not None:
            pid = self._pid
            event["pid"] = pid
            self._ensure_process_metadata(pid)
            if event.get("ph") != "M":
                args = event.setdefault("args", {})
                if "rank" not in args:
                    try:
                        args["rank"] = int(self._rank)
                    except (TypeError, ValueError):
                        args["rank"] = self._rank
        elif self._aggregate and self._rank is None:
            logger.warning(
                "PerfTracer aggregation enabled but rank is not set; "
                "call configure(..., rank=<rank>) to avoid merged lanes"
            )
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
        if self._rank is not None:
            rank_label = f"Rank {self._rank}, Process"
        else:
            rank_label = "Process"
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
            "args": {"sort_index": self._rank if self._rank is not None else pid},
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
        category: str | None,
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

    def __exit__(self, exc_type, exc, exc_tb) -> None:
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
        category: str | None,
        args: dict[str, Any] | None,
    ) -> None:
        self._scope = _Scope(tracer, name, category=category, args=args)

    async def __aenter__(self) -> _AsyncScope:
        self._scope.__enter__()
        return self

    async def __aexit__(self, exc_type, exc, exc_tb) -> None:
        self._scope.__exit__(exc_type, exc, exc_tb)
        return False


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


def _init_tracer_from_env() -> PerfTracer:
    path = os.getenv("AREAL_PERF_TRACE_PATH")
    enabled = _strtobool(os.getenv("AREAL_PERF_TRACE_ENABLED")) or bool(path)
    aggregate = _strtobool(os.getenv("AREAL_PERF_TRACE_AGGREGATE"))
    tracer = PerfTracer(enabled=enabled, aggregate=aggregate)
    if path:
        try:
            tracer.set_output(path)
        except ValueError:
            logger.warning(
                "AREAL_PERF_TRACE_PATH is set but invalid (%s); tracing disabled",
                path,
            )
            tracer.set_enabled(False)
    return tracer


GLOBAL_TRACER = _init_tracer_from_env()


def _save_at_exit() -> None:
    if not GLOBAL_TRACER.enabled:
        return
    try:
        GLOBAL_TRACER.save(reset=False)
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to flush perf trace on exit: %s", exc, exc_info=True)


atexit.register(_save_at_exit)


# ----------------------------------------------------------------------
# Module-level convenience functions
# ----------------------------------------------------------------------
def get_tracer() -> PerfTracer:
    return GLOBAL_TRACER


def configure(
    *,
    enabled: bool | None = None,
    output_path: str | None = None,
    rank: int | None = None,
    aggregate: bool | None = None,
) -> PerfTracer:
    tracer = get_tracer()
    if aggregate is not None:
        tracer.set_aggregate(aggregate)
    if rank is not None:
        tracer._rank = rank
    if output_path is not None:
        tracer.set_output(output_path)
    if enabled is not None:
        tracer.set_enabled(enabled)
    return tracer


def trace_scope(
    name: str,
    *,
    category: str | None = None,
    args: dict[str, Any] | None = None,
):
    return GLOBAL_TRACER.trace_scope(name, category=category, args=args)


def atrace_scope(
    name: str,
    *,
    category: str | None = None,
    args: dict[str, Any] | None = None,
):
    return GLOBAL_TRACER.atrace_scope(name, category=category, args=args)


def instant(
    name: str,
    *,
    category: str | None = None,
    args: dict[str, Any] | None = None,
) -> None:
    GLOBAL_TRACER.instant(name, category=category, args=args)


def save(path: str | None = None, *, reset: bool = True) -> str | None:
    return GLOBAL_TRACER.save(path, reset=reset)


__all__ = [
    "PerfTracer",
    "GLOBAL_TRACER",
    "get_tracer",
    "configure",
    "trace_scope",
    "atrace_scope",
    "instant",
    "save",
]
