"""Configuration for the GatewayRolloutController."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GatewayControllerConfig:
    """Unified configuration for GatewayRolloutController.

    Consolidates settings for the gateway, router, data proxy services,
    and the WorkflowExecutor / staleness management.
    """

    # -- Gateway service ---------------------------------------------------
    gateway_host: str = "0.0.0.0"
    gateway_port: int = 8080

    # -- Router service ----------------------------------------------------
    router_host: str = "0.0.0.0"
    router_port: int = 8081

    # -- Data proxy service ------------------------------------------------
    data_proxy_host: str = "0.0.0.0"
    data_proxy_base_port: int = 8082  # incremented per worker

    # -- Shared credentials ------------------------------------------------
    admin_api_key: str = "areal-admin-key"

    # -- Model / tokenizer -------------------------------------------------
    tokenizer_path: str = ""
    model_path: str = ""

    # -- Routing -----------------------------------------------------------
    routing_strategy: str = "round_robin"
    poll_interval: float = 5.0  # router health-poll interval (seconds)

    # -- HTTP timeouts -----------------------------------------------------
    request_timeout: float = 120.0  # per-request timeout (seconds)
    setup_timeout: float = 300.0  # timeout waiting for services to start

    # -- SGLang backend tuning ---------------------------------------------
    max_resubmit_retries: int = 20
    resubmit_wait: float = 0.5  # seconds between is_paused polls

    # -- Log level for gateway micro-services ------------------------------
    log_level: str = "info"

    # -- WorkflowExecutor / staleness --------------------------------------
    consumer_batch_size: int = 16
    max_concurrent_rollouts: int | None = None
    max_head_offpolicyness: int = 0
    queue_size: int | None = None
    enable_rollout_tracing: bool = False

    # -- Trajectory dump ---------------------------------------------------
    fileroot: str | None = None
    experiment_name: str | None = None
    trial_name: str | None = None
    check_trajectory_format: bool = False

    # -- Scheduler / allocation (passed through from trainer) --------------
    scheduling_spec: tuple = field(default_factory=tuple)
    scheduling_strategy: str | None = None
    pause_grace_period: float = 0.5
