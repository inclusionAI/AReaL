# SPDX-License-Identifier: Apache-2.0

"""AgentServiceController — orchestrates agent service micro-services via Guards.

Mirrors the architecture of
:class:`~areal.experimental.inference_service.controller.controller.GatewayInferenceController`:
Guard workers are created via the Scheduler, then the controller forks
Router, Worker+DataProxy pairs, and Gateway onto them via HTTP API.

Lifecycle::

    from areal.infra.scheduler.local import LocalScheduler

    scheduler = LocalScheduler(...)
    controller = AgentServiceController(config, scheduler)
    controller.initialize()
    # ... run traffic ...
    controller.scale_up(2)     # add 2 Worker+DataProxy pairs
    controller.scale_down(1)   # drain + remove 1 pair
    controller.destroy()
"""

from __future__ import annotations

import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import requests

from areal.experimental.agent_service.controller.config import (
    AgentServiceControllerConfig,
)
from areal.utils import logging
from areal.utils.network import format_hostport

if TYPE_CHECKING:
    from areal.api.scheduler_api import Scheduler, Worker

logger = logging.getLogger("AgentServiceController")

_GUARD_ROLE = "agent-guard"
_UNREGISTER_RETRIES = 3
_HEALTH_CHECK_WORKERS = 4


@dataclass
class _WorkerPair:
    pair_index: int
    guard_addr: str
    worker_host: str
    worker_port: int
    proxy_host: str
    proxy_port: int
    proxy_addr: str
    worker_addr: str


class AgentServiceController:
    """Orchestrator for the Agent Service micro-service stack.

    Parameters
    ----------
    config:
        Controller configuration.
    scheduler:
        Scheduler instance used to create and manage Guard workers.
    """

    def __init__(
        self,
        config: AgentServiceControllerConfig,
        scheduler: Scheduler | None = None,
    ) -> None:
        self.config = config
        self.scheduler = scheduler

        self._guard_addrs: list[str] = []
        self._workers: list[Worker] = []
        self._service_roles: list[str] = []

        self._router_addr: str = ""
        self._gateway_addr: str = ""

        self._pairs: dict[int, _WorkerPair] = {}
        self._pairs_lock = threading.Lock()
        self._next_pair_index: int = 0

        self._forked_services: list[tuple[str, str, int]] = []

        self._sessions: dict[str, dict[str, Any]] = {}
        self._sessions_lock = threading.Lock()

        self._health_stop = threading.Event()
        self._health_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Launch the full micro-service stack.

        Order: Guards (via scheduler) → Router → Worker+DataProxy pairs →
        register → Gateway → health monitor.
        On failure, already-forked services are cleaned up via destroy().

        When ``num_pairs`` is 0 and no scheduler is provided, the stack
        is skipped entirely — only the data-collection APIs
        (``new_session``, ``set_reward``) are available.
        """
        if self.config.num_pairs == 0 and self.scheduler is None:
            logger.info(
                "num_pairs=0 with no scheduler; "
                "skipping micro-service stack (data-collection-only mode)"
            )
            return
        if self.scheduler is None:
            raise ValueError("A scheduler is required when num_pairs > 0")
        try:
            self._do_initialize()
        except Exception:
            logger.error("initialize() failed, rolling back...")
            self.destroy()
            raise

    def _do_initialize(self) -> None:
        from areal.api.cli_args import SchedulingSpec, SchedulingStrategy
        from areal.api.scheduler_api import Job

        cfg = self.config

        # Step 1: Create Guard workers via scheduler
        guard_spec = SchedulingSpec(
            gpu=0, cmd=f"{sys.executable} -m areal.experimental.agent_service.guard"
        )
        num_guards = max(cfg.num_pairs, 1)
        guard_job = Job(
            role=_GUARD_ROLE,
            replicas=num_guards,
            tasks=[guard_spec for _ in range(num_guards)],
            scheduling_strategy=SchedulingStrategy(),
        )
        self.scheduler.create_workers(job=guard_job)
        self._service_roles.append(_GUARD_ROLE)

        self._workers = self.scheduler.get_workers(role=_GUARD_ROLE)
        self._guard_addrs = [
            f"http://{format_hostport(w.ip, int(w.worker_ports[0]))}"
            for w in self._workers
        ]
        logger.info("Guards ready: %s", self._guard_addrs)

        # Step 2: Fork Router on guard[0]
        guard_0 = self._guard_addrs[0]
        router_cmd = [
            sys.executable,
            "-m",
            "areal.experimental.agent_service.router",
            "--admin-api-key",
            cfg.admin_api_key,
        ]
        router_host, router_port = self._fork_on_guard(
            guard_addr=guard_0,
            role="agent-router",
            worker_index=0,
            raw_cmd=router_cmd,
        )
        self._router_addr = f"http://{format_hostport(router_host, router_port)}"
        logger.info("Router: %s", self._router_addr)

        # Step 3: Fork Worker+DataProxy pairs
        self.scale_up(cfg.num_pairs)

        # Step 4: Fork Gateway on guard[0]
        gw_cmd = [
            sys.executable,
            "-m",
            "areal.experimental.agent_service.gateway",
            "--router-addr",
            self._router_addr,
            "--admin-api-key",
            cfg.admin_api_key,
        ]
        gw_host, gw_port = self._fork_on_guard(
            guard_addr=guard_0,
            role="agent-gateway",
            worker_index=0,
            raw_cmd=gw_cmd,
        )
        self._gateway_addr = f"http://{format_hostport(gw_host, gw_port)}"
        logger.info("Gateway: %s", self._gateway_addr)

        # Step 5: Start health monitor
        if cfg.health_poll_interval > 0:
            self._health_stop.clear()
            self._health_thread = threading.Thread(
                target=self._health_monitor_loop, daemon=True
            )
            self._health_thread.start()

    def destroy(self) -> None:
        """Tear down all services in reverse order."""
        self._stop_health_monitor()

        for guard_addr, role, worker_index in reversed(self._forked_services):
            try:
                self._kill_forked_service(guard_addr, role, worker_index)
            except requests.RequestException:
                logger.error(
                    "Error killing forked service %s/%d: %s",
                    role,
                    worker_index,
                    traceback.format_exc(),
                )
        self._forked_services.clear()

        if self.scheduler is not None:
            for role in reversed(self._service_roles):
                try:
                    self.scheduler.delete_workers(role=role)
                    logger.info("Workers deleted for role: %s", role)
                except Exception:
                    logger.error(
                        "Error deleting workers for role %s: %s",
                        role,
                        traceback.format_exc(),
                    )
        self._service_roles.clear()
        self._workers.clear()
        self._guard_addrs.clear()
        with self._pairs_lock:
            self._pairs.clear()
        self._router_addr = ""
        self._gateway_addr = ""

    def scale_up(self, count: int) -> list[int]:
        """Add *count* Worker+DataProxy pairs.

        Pairs are distributed across guards round-robin.
        Returns the pair indices that were created.
        """
        cfg = self.config
        created: list[int] = []

        for _ in range(count):
            pair_index = self._next_pair_index
            self._next_pair_index += 1

            guard_addr = self._guard_addrs[pair_index % len(self._guard_addrs)]

            worker_cmd = [
                sys.executable,
                "-m",
                "areal.experimental.agent_service.worker",
                "--agent",
                cfg.agent_cls_path,
                "--log-level",
                cfg.log_level,
            ]
            worker_host, worker_port = self._fork_on_guard(
                guard_addr=guard_addr,
                role=f"agent-worker-{pair_index}",
                worker_index=pair_index,
                raw_cmd=worker_cmd,
            )
            worker_addr = f"http://{format_hostport(worker_host, worker_port)}"

            proxy_cmd = [
                sys.executable,
                "-m",
                "areal.experimental.agent_service.data_proxy",
                "--worker-addr",
                worker_addr,
            ]
            proxy_host, proxy_port = self._fork_on_guard(
                guard_addr=guard_addr,
                role=f"agent-proxy-{pair_index}",
                worker_index=pair_index,
                raw_cmd=proxy_cmd,
            )
            proxy_addr = f"http://{format_hostport(proxy_host, proxy_port)}"

            pair = _WorkerPair(
                pair_index=pair_index,
                guard_addr=guard_addr,
                worker_host=worker_host,
                worker_port=worker_port,
                proxy_host=proxy_host,
                proxy_port=proxy_port,
                proxy_addr=proxy_addr,
                worker_addr=worker_addr,
            )

            try:
                self._register_proxy(proxy_addr)
            except Exception:
                logger.error(
                    "Failed to register pair %d, cleaning up forked processes",
                    pair_index,
                )
                self._cleanup_pair_forks(pair_index, guard_addr)
                raise

            with self._pairs_lock:
                self._pairs[pair_index] = pair
            created.append(pair_index)

            logger.info(
                "Pair %d: worker=%s proxy=%s", pair_index, worker_addr, proxy_addr
            )

        return created

    def scale_down(self, count: int) -> list[int]:
        """Remove *count* pairs (LIFO order).

        For each pair: unregister from Router (with retry) → drain active
        sessions → kill DataProxy → kill Worker.
        Returns the pair indices that were removed.
        """
        removed: list[int] = []

        with self._pairs_lock:
            indices = sorted(self._pairs.keys(), reverse=True)[:count]

        for pair_index in indices:
            with self._pairs_lock:
                pair = self._pairs.get(pair_index)
            if pair is None:
                continue

            try:
                self._unregister_proxy(pair.proxy_addr)
            except requests.RequestException:
                logger.error(
                    "Unregister failed for pair %d after retries, skipping",
                    pair_index,
                )
                continue

            self._drain_proxy(pair.proxy_addr)

            with self._pairs_lock:
                self._pairs.pop(pair_index, None)

            proxy_key = (pair.guard_addr, f"agent-proxy-{pair_index}", pair_index)
            worker_key = (pair.guard_addr, f"agent-worker-{pair_index}", pair_index)

            for guard_addr, role, wi in [proxy_key, worker_key]:
                try:
                    self._kill_forked_service(guard_addr, role, wi)
                    entry = (guard_addr, role, wi)
                    if entry in self._forked_services:
                        self._forked_services.remove(entry)
                except requests.RequestException:
                    logger.warning(
                        "Failed to kill %s/%d: %s",
                        role,
                        wi,
                        traceback.format_exc(),
                    )

            removed.append(pair_index)
            logger.info("Removed pair %d", pair_index)

        return removed

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def router_addr(self) -> str:
        return self._router_addr

    @property
    def gateway_addr(self) -> str:
        return self._gateway_addr

    @property
    def pairs(self) -> dict[int, _WorkerPair]:
        with self._pairs_lock:
            return dict(self._pairs)

    # ------------------------------------------------------------------
    # Data-collection APIs (inference service integration)
    # ------------------------------------------------------------------

    def new_session(self, task_id: str = "") -> dict[str, str]:
        """Create a new session for data collection.

        Generates a session ID for the agent service and starts a
        corresponding session on the inference service via
        ``/rl/start_session``.

        Parameters
        ----------
        task_id:
            Task identifier forwarded to the inference service.  Defaults
            to the generated session ID when empty.

        Returns
        -------
        dict with keys:

        * ``session_id`` — agent-service session ID (use as ``user``
          field in ``/v1/responses`` requests).
        * ``inference_session_id`` — inference-service session ID
          (for trajectory export).
        * ``inference_api_key`` — session-scoped API key for the
          inference gateway.
        """
        cfg = self.config
        if not cfg.inference_addr:
            raise RuntimeError(
                "inference_addr must be set in AgentServiceControllerConfig "
                "to use data-collection APIs"
            )

        session_id = f"agent-sess-{uuid.uuid4().hex[:12]}"
        if not task_id:
            task_id = session_id

        inf_addr = cfg.inference_addr.rstrip("/")
        resp = requests.post(
            f"{inf_addr}/rl/start_session",
            json={"task_id": task_id},
            headers={"Authorization": f"Bearer {cfg.inference_api_key}"},
            timeout=cfg.request_timeout,
        )
        resp.raise_for_status()
        inf_data = resp.json()

        session_info: dict[str, str] = {
            "session_id": session_id,
            "inference_session_id": inf_data["session_id"],
            "inference_api_key": inf_data["api_key"],
        }

        with self._sessions_lock:
            self._sessions[session_id] = session_info

        logger.info(
            "New session: %s (inference session: %s)",
            session_id,
            inf_data["session_id"],
        )
        return session_info

    def step(
        self,
        input: str | list[dict[str, Any]],
        session_id: str,
    ) -> dict[str, Any]:
        """Send a message to the agent service and return the response.

        Parameters
        ----------
        input:
            A plain string or an OpenResponses-style input list
            (e.g. ``[{"type": "message", "content": "hello"}]``).
        session_id:
            Agent-service session ID returned by :meth:`new_session`.

        Returns
        -------
        dict
            The JSON response from the agent service gateway
            ``POST /v1/responses``.
        """
        session_info = self._resolve_session(session_id)
        sid = session_info["session_id"]

        if not self._gateway_addr:
            raise RuntimeError(
                "step() requires the agent-service gateway to be running. "
                "It is not available in data-collection-only mode "
                "(num_pairs=0 with no scheduler)."
            )

        if isinstance(input, str):
            input_items: list[dict[str, Any]] = [{"type": "message", "content": input}]
        else:
            input_items = input

        cfg = self.config
        metadata: dict[str, Any] = {}
        if cfg.inference_addr:
            metadata["inference_base_url"] = cfg.inference_addr.rstrip("/")
        if cfg.inference_model:
            metadata["inference_model"] = cfg.inference_model
        inf_api_key = session_info.get("inference_api_key", "")
        if inf_api_key:
            metadata["inference_api_key"] = inf_api_key

        body: dict[str, Any] = {
            "input": input_items,
            "model": (cfg.inference_model or "default").replace("/", "--"),
            "user": sid,
        }
        if metadata:
            body["metadata"] = metadata

        resp = requests.post(
            f"{self._gateway_addr}/v1/responses",
            json=body,
            headers={"Authorization": f"Bearer {cfg.admin_api_key}"},
            timeout=cfg.request_timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def set_reward(
        self,
        reward: float,
        session_id: str,
    ) -> dict[str, Any]:
        """Set a reward on the inference service for the current session.

        Parameters
        ----------
        reward:
            Scalar reward value.
        session_id:
            Agent-service session ID returned by :meth:`new_session`.

        Returns
        -------
        dict
            The JSON response from the inference gateway
            ``POST /rl/set_reward``.
        """
        session_info = self._resolve_session(session_id)
        inf_api_key = session_info["inference_api_key"]

        cfg = self.config
        inf_addr = cfg.inference_addr.rstrip("/")
        resp = requests.post(
            f"{inf_addr}/rl/set_reward",
            json={"interaction_id": None, "reward": reward},
            headers={"Authorization": f"Bearer {inf_api_key}"},
            timeout=cfg.request_timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def _resolve_session(self, session_id: str) -> dict[str, Any]:
        with self._sessions_lock:
            session_info = self._sessions.get(session_id)
        if session_info is None:
            raise KeyError(f"Unknown session_id: {session_id!r}")
        return session_info

    # ------------------------------------------------------------------
    # Guard interaction helpers
    # ------------------------------------------------------------------

    def _fork_on_guard(
        self,
        guard_addr: str,
        role: str,
        worker_index: int,
        raw_cmd: list[str],
        health_path: str = "/health",
        env: dict[str, str] | None = None,
    ) -> tuple[str, int]:
        resp = requests.post(
            f"{guard_addr}/alloc_ports",
            json={"count": 1},
            timeout=30,
        )
        resp.raise_for_status()
        port_data = resp.json()
        host = port_data["host"]
        port = port_data["ports"][0]

        cmd = list(raw_cmd) + ["--host", host, "--port", str(port)]

        merged_env = {**self.config.env, **(env or {})}

        fork_payload: dict[str, Any] = {
            "role": role,
            "worker_index": worker_index,
            "raw_cmd": cmd,
        }
        if merged_env:
            fork_payload["env"] = merged_env

        resp = requests.post(
            f"{guard_addr}/fork",
            json=fork_payload,
            timeout=30,
        )
        resp.raise_for_status()

        self._forked_services.append((guard_addr, role, worker_index))

        addr = f"http://{format_hostport(host, port)}"
        self._wait_for_service(f"{addr}{health_path}", role)

        return host, port

    def _cleanup_pair_forks(self, pair_index: int, guard_addr: str) -> None:
        for role_prefix in ("agent-proxy-", "agent-worker-"):
            role = f"{role_prefix}{pair_index}"
            entry = (guard_addr, role, pair_index)
            if entry in self._forked_services:
                try:
                    self._kill_forked_service(guard_addr, role, pair_index)
                except requests.RequestException:
                    pass
                self._forked_services.remove(entry)

    def _kill_forked_service(
        self, guard_addr: str, role: str, worker_index: int
    ) -> None:
        try:
            resp = requests.post(
                f"{guard_addr}/kill_forked_worker",
                json={"role": role, "worker_index": worker_index},
                timeout=10,
            )
            if resp.status_code == 200:
                logger.info("Killed forked service %s/%d", role, worker_index)
            else:
                logger.warning(
                    "Failed to kill forked service %s/%d: %s",
                    role,
                    worker_index,
                    resp.text,
                )
        except requests.RequestException as exc:
            logger.error(
                "Error killing forked service %s/%d: %s", role, worker_index, exc
            )

    def _wait_for_service(
        self, url: str, name: str, timeout: float | None = None
    ) -> None:
        timeout = timeout or self.config.setup_timeout
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                resp = requests.get(url, timeout=2)
                if resp.status_code == 200:
                    logger.info("%s healthy at %s", name, url)
                    return
            except requests.RequestException:
                pass
            time.sleep(0.5)
        raise TimeoutError(f"{name} did not become healthy at {url} within {timeout}s")

    def _register_proxy(self, proxy_addr: str) -> None:
        """Raises on failure so that ``scale_up`` callers know the pair is
        non-functional.
        """
        if not self._router_addr:
            return
        resp = requests.post(
            f"{self._router_addr}/register",
            json={"addr": proxy_addr},
            headers={"Authorization": f"Bearer {self.config.admin_api_key}"},
            timeout=10,
        )
        resp.raise_for_status()
        logger.info("Registered proxy %s with Router", proxy_addr)

    def _drain_proxy(self, proxy_addr: str) -> None:
        timeout = self.config.drain_timeout
        if timeout <= 0:
            return
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                resp = requests.get(f"{proxy_addr}/health", timeout=2)
                if resp.status_code == 200:
                    active = resp.json().get("active_sessions", 0)
                    if active == 0:
                        logger.info("Proxy %s drained", proxy_addr)
                        return
                    logger.debug(
                        "Proxy %s draining: %d active sessions", proxy_addr, active
                    )
            except requests.RequestException:
                break
            time.sleep(1.0)
        logger.warning(
            "Proxy %s drain timed out after %.0fs, force-killing", proxy_addr, timeout
        )

    def _check_pair_health(self, pair_index: int, proxy_addr: str) -> None:
        try:
            resp = requests.get(f"{proxy_addr}/health", timeout=2)
            if resp.status_code != 200:
                logger.warning(
                    "Pair %d proxy %s returned %d",
                    pair_index,
                    proxy_addr,
                    resp.status_code,
                )
        except requests.RequestException:
            logger.warning("Pair %d proxy %s unreachable", pair_index, proxy_addr)

    def _health_monitor_loop(self) -> None:
        interval = self.config.health_poll_interval
        while not self._health_stop.wait(timeout=interval):
            with self._pairs_lock:
                snapshot = list(self._pairs.items())
            if not snapshot:
                continue
            with ThreadPoolExecutor(
                max_workers=min(_HEALTH_CHECK_WORKERS, len(snapshot))
            ) as pool:
                futures = {
                    pool.submit(self._check_pair_health, idx, pair.proxy_addr): idx
                    for idx, pair in snapshot
                }
                for future in as_completed(futures, timeout=10):
                    try:
                        future.result()
                    except Exception:
                        pass

    def _stop_health_monitor(self) -> None:
        self._health_stop.set()
        if self._health_thread is not None:
            self._health_thread.join(timeout=5)
            self._health_thread = None

    def _unregister_proxy(self, proxy_addr: str) -> None:
        """Unregister with retry. Raises after all retries exhausted."""
        if not self._router_addr:
            return
        last_exc: Exception | None = None
        for attempt in range(_UNREGISTER_RETRIES):
            try:
                resp = requests.post(
                    f"{self._router_addr}/unregister",
                    json={"addr": proxy_addr},
                    headers={"Authorization": f"Bearer {self.config.admin_api_key}"},
                    timeout=5,
                )
                resp.raise_for_status()
                logger.info("Unregistered proxy %s", proxy_addr)
                return
            except requests.RequestException as exc:
                last_exc = exc
                logger.warning(
                    "Unregister proxy %s attempt %d/%d failed: %s",
                    proxy_addr,
                    attempt + 1,
                    _UNREGISTER_RETRIES,
                    exc,
                )
                if attempt < _UNREGISTER_RETRIES - 1:
                    time.sleep(1.0)
        raise last_exc  # type: ignore[misc]
