# SPDX-License-Identifier: Apache-2.0

"""Configuration for the GatewayInferenceController."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GatewayControllerConfig:
    """Unified configuration for GatewayInferenceController.

    Consolidates settings for the gateway, router, data proxy services,
    and the WorkflowExecutor / staleness management.
    """

    # -- Model / tokenizer -------------------------------------------------
    tokenizer_path: str = ""
    model_path: str = ""
    model: str = "default"

    # -- Routing -----------------------------------------------------------
    routing_strategy: str = "round_robin"
    poll_interval: float = 5.0  # router health-poll interval (seconds)

    # -- HTTP timeouts -----------------------------------------------------
    request_timeout: float = 120.0  # per-request timeout (seconds)
    setup_timeout: float = 300.0  # timeout waiting for services to start
    set_reward_finish_timeout: float = 0.0

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
    dump_to_file: bool = False

    # -- Scheduler / allocation (passed through from trainer) --------------
    backend: str = "sglang:d1"
    scheduling_spec: tuple = field(default_factory=tuple)
    pause_grace_period: float = 0.5
    n_gpus_per_node: int | None = None  # GPUs per physical node; None = single-node

    # -- Admin / workflow --------------------------------------------------
    admin_api_key: str | None = None
    turn_discount: float = 1.0
    export_style: str = "individual"
    tool_call_parser: str = "qwen"
    reasoning_parser: str = "qwen3"
    engine_max_tokens: int | None = None
    chat_template_type: str = "hf"

    # -- External model API ------------------------------------------------
    api_url: str | None = None
    provider_api_key: str | None = None
