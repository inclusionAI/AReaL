# SPDX-License-Identifier: Apache-2.0

"""Configuration for the AgentServiceController."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..auth import DEFAULT_ADMIN_API_KEY


@dataclass
class AgentServiceControllerConfig:
    """Unified configuration for AgentServiceController.

    Consolidates settings for the guard, router, gateway, worker, and
    data proxy micro-services launched by the controller.
    """

    # -- Agent class -------------------------------------------------------
    agent_cls_path: str = ""
    """Fully-qualified import path for the ``AgentRunnable`` implementation
    (e.g. ``examples.agent_service.agent.Tau2Agent``)."""

    # -- Authentication ----------------------------------------------------
    admin_api_key: str = DEFAULT_ADMIN_API_KEY
    """Shared admin API key for inter-service Bearer auth."""

    # -- Scaling -----------------------------------------------------------
    num_pairs: int = 1
    """Number of Worker+DataProxy pairs to launch on initialize."""

    # -- Timeouts ----------------------------------------------------------
    setup_timeout: float = 120.0
    """Timeout (seconds) waiting for each service to become healthy."""

    health_poll_interval: float = 5.0
    """Seconds between health polls for crash detection (0 = disabled)."""

    drain_timeout: float = 30.0
    """Seconds to wait for active sessions to drain before force-killing a pair."""

    # -- Log level ---------------------------------------------------------
    log_level: str = "warning"
    """Log level for spawned micro-services."""

    # -- Environment -------------------------------------------------------
    env: dict[str, str] = field(default_factory=dict)
    """Extra environment variables to pass to all forked child processes."""

    def __post_init__(self) -> None:
        if not self.agent_cls_path:
            raise ValueError("agent_cls_path must be a non-empty import path")
        if self.num_pairs < 0:
            raise ValueError(f"num_pairs must be non-negative, got {self.num_pairs}")
        if self.setup_timeout <= 0:
            raise ValueError(
                f"setup_timeout must be positive, got {self.setup_timeout}"
            )
        if self.drain_timeout < 0:
            raise ValueError(
                f"drain_timeout must be non-negative, got {self.drain_timeout}"
            )
