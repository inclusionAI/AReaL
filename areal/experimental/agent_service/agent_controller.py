"""AgentController: lifecycle manager for Gateway + Agent Worker processes.

This module provides the :class:`AgentController` class, which manages the
Gateway and Agent Worker processes exclusively via the Scheduler API — the same
pattern used by :class:`~areal.infra.controller.rollout_controller.RolloutController`.

Usage::

    from areal.experimental.agent_service.agent_controller import AgentController
    from areal.experimental.agent_service.config import GatewayConfig
    from areal.api.cli_args import AgentServiceSpec

    controller = AgentController(config=GatewayConfig(), scheduler=scheduler)
    gateway_addr = controller.start(
        AgentServiceSpec(agent_import_path="mymodule.MyAgent", workers=4)
    )
    # ... training loop ...
    controller.stop()
"""

from __future__ import annotations

import json
import time

from areal.api.cli_args import AgentServiceSpec, SchedulingSpec
from areal.api.scheduler_api import Job, Scheduler, Worker
from areal.infra.scheduler.exceptions import WorkerTimeoutError
from areal.utils import logging

from .config import GatewayConfig

logger = logging.getLogger("AgentController")

# Role names used for scheduler registration
_GATEWAY_ROLE = "agent_gateway"
_WORKER_ROLE = "agent_worker"


class AgentController:
    """Manages the lifecycle of the Gateway and Agent Worker processes.

    The controller creates processes via the scheduler (same pattern as
    :class:`~areal.infra.controller.rollout_controller.RolloutController`).
    It exposes a simple API for starting, stopping, and scaling workers.

    Attributes:
        config: Gateway configuration.
        scheduler: Scheduler used to create worker processes.
    """

    _GATEWAY_ROLE = _GATEWAY_ROLE
    _WORKER_ROLE = _WORKER_ROLE

    def __init__(self, config: GatewayConfig, scheduler: Scheduler) -> None:
        """Initialize the AgentController.

        Args:
            config: Gateway configuration (host, port, queue_size, etc.).
            scheduler: Scheduler instance for creating worker processes.
        """
        if scheduler is None:
            raise TypeError(
                "scheduler must be a Scheduler instance, not None. "
                "Pass a LocalScheduler, RayScheduler, or SlurmScheduler."
            )
        self._config = config
        self._scheduler = scheduler

        self._agent_workers: list[Worker] = []
        self._gateway_addr: str | None = None
        self._gateway_started: bool = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def gateway_addr(self) -> str | None:
        """The Gateway HTTP address, or None if not started."""
        return self._gateway_addr

    @property
    def gateway_started(self) -> bool:
        """True if the Gateway process has been started."""
        return self._gateway_started

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, spec: AgentServiceSpec) -> str:
        """Start Gateway + Workers. Returns gateway_addr.

        Convenience method that calls :meth:`start_gateway` then
        :meth:`start_workers`.

        Args:
            spec: Agent service specification (import path, reuse, workers, etc.).

        Returns:
            HTTP address of the Gateway (e.g., ``'http://host:8300'``).
        """
        self.start_gateway()
        self.start_workers(spec)
        return self.gateway_addr

    def start_gateway(self) -> str:
        """Start the Gateway process via the scheduler.

        Returns:
            HTTP address of the Gateway (e.g., ``'http://host:8300'``).
        """
        if self._gateway_started:
            logger.warning("Gateway already started, returning existing address")
            assert self._gateway_addr is not None
            return self._gateway_addr

        cmd = (
            f"python -m areal.experimental.agent_service.gateway "
            f"--queue-size {self._config.queue_size}"
        )

        spec = SchedulingSpec(
            cpu=self._config.gateway_cpu,
            gpu=0,
            mem=self._config.gateway_mem,
            port_count=1,
            cmd=cmd,
            env_vars={},
        )
        job = Job(
            role=self._GATEWAY_ROLE,
            replicas=1,
            tasks=[spec],
        )

        worker_ids = self._scheduler.create_workers(job=job)
        logger.info("Gateway worker created: %s", worker_ids)

        try:
            gateway_workers = self._scheduler.get_workers(
                role=self._GATEWAY_ROLE, timeout=30
            )
        except WorkerTimeoutError as e:
            raise RuntimeError(f"Gateway failed to become ready: {e}") from e

        gw = gateway_workers[0]
        self._gateway_addr = f"http://{gw.ip}:{gw.worker_ports[0]}"
        self._gateway_started = True

        logger.info("Gateway started at %s", self._gateway_addr)
        return self._gateway_addr

    def start_workers(self, spec: AgentServiceSpec) -> list[str]:
        """Start N Agent Worker processes via the scheduler.

        Each worker is configured to register with the Gateway on startup.

        Args:
            spec: Agent service specification containing import path,
                reuse flag, init kwargs, and worker count.

        Returns:
            List of worker addresses (e.g., ``['http://host:8301', ...]``).

        Raises:
            RuntimeError: If the Gateway has not been started yet.
        """
        if not self._gateway_started or self._gateway_addr is None:
            raise RuntimeError(
                "Gateway must be started before starting workers. "
                "Call start_gateway() first."
            )

        agent_init_kwargs_str = (
            json.dumps(spec.agent_init_kwargs) if spec.agent_init_kwargs else ""
        )

        worker_spec = SchedulingSpec(
            cpu=self._config.worker_cpu,
            gpu=0,
            mem=self._config.worker_mem,
            port_count=1,
            cmd="python -m areal.experimental.agent_service.worker_server",
            env_vars={
                "AREAL_AGENT_GATEWAY_ADDR": self._gateway_addr,
                "AREAL_AGENT_IMPORT_PATH_INTERNAL": spec.agent_import_path or "",
                "AREAL_AGENT_REUSE_INTERNAL": "true" if spec.agent_reuse else "false",
                "AREAL_AGENT_INIT_KWARGS_INTERNAL": agent_init_kwargs_str,
            },
        )
        job = Job(
            role=self._WORKER_ROLE,
            replicas=spec.workers,
            tasks=[worker_spec],
        )

        worker_ids = self._scheduler.create_workers(job=job)
        logger.info("Agent Worker processes created: %s", worker_ids)

        try:
            self._agent_workers = self._scheduler.get_workers(
                role=self._WORKER_ROLE, timeout=60
            )
        except WorkerTimeoutError as e:
            raise RuntimeError(f"Agent workers failed to start: {e}") from e

        # P2 mitigation: Workers register with Gateway asynchronously on
        # startup (POST /register). The scheduler's get_workers() only
        # confirms the worker HTTP server is up (/health), but Gateway
        # registration may not have completed yet. Brief sleep reduces
        # the race window for the first request.
        time.sleep(0.5)

        addrs = [f"http://{w.ip}:{w.worker_ports[0]}" for w in self._agent_workers]
        logger.info("Agent Workers started at: %s", addrs)
        return addrs

    def stop(self) -> None:
        """Stop all Gateway and Worker processes. Idempotent.

        Deletes workers first (so they can unregister from the Gateway),
        then deletes the Gateway. Each deletion is wrapped in a try/except
        so that calling ``stop()`` twice does not raise.
        """
        # Delete workers first (before gateway)
        try:
            self._scheduler.delete_workers(self._WORKER_ROLE)
        except Exception as e:
            logger.warning("Failed to delete agent workers: %s", e)

        try:
            self._scheduler.delete_workers(self._GATEWAY_ROLE)
        except Exception as e:
            logger.warning("Failed to delete gateway: %s", e)

        self._gateway_started = False
        self._gateway_addr = None
        self._agent_workers = []

    def scale_workers(
        self,
        target: int,
        spec: AgentServiceSpec | None = None,
    ) -> None:
        """Scale the number of Agent Workers to the target count.

        Currently only supports **scale-up** (adding workers). Scaling down
        requires stopping individual workers, which is scheduler-dependent.

        .. note::
            Scale-up creates ``target - current`` additional workers. The
            *spec* argument is required when scaling up.

        Args:
            target: Target number of Agent Workers.
            spec: Agent service specification for new workers (required for
                scale-up).

        # TODO(agent): Implement scale-down via selective worker termination
        #   once the Scheduler API supports deleting individual replicas.
        """
        current = len(self._agent_workers)
        if target > current:
            if spec is None:
                raise ValueError("spec is required when scaling up (target > current)")
            delta = target - current
            logger.info("Scaling up: adding %d workers (current=%d)", delta, current)
            # Create a modified spec with the delta count
            scale_spec = AgentServiceSpec(
                agent_import_path=spec.agent_import_path,
                agent_reuse=spec.agent_reuse,
                agent_init_kwargs=spec.agent_init_kwargs,
                workers=delta,
            )
            self.start_workers(scale_spec)
        elif target < current:
            logger.warning(
                "Scaling down from %d to %d workers is not yet supported "
                "(requires scheduler-level worker termination). "
                "Current count unchanged.",
                current,
                target,
            )
        else:
            logger.info("Worker count already at target %d, no change needed", target)

    def get_worker_count(self) -> int:
        """Return the current number of Agent Worker processes.

        Returns:
            Number of registered Agent Worker processes.
        """
        return len(self._agent_workers)

    def get_gateway_addr(self) -> str:
        """Return the Gateway HTTP address.

        Returns:
            HTTP address of the Gateway (e.g., ``'http://host:8300'``).

        Raises:
            RuntimeError: If the Gateway has not been started.
        """
        if self._gateway_addr is None:
            raise RuntimeError("Gateway not started. Call start_gateway() first.")
        return self._gateway_addr
