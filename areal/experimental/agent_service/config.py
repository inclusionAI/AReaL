"""Configuration for Agent Service.

This module defines the AgentServiceConfig dataclass for configuring
the Agent Service.
"""

from dataclasses import dataclass, field


@dataclass
class AgentServiceConfig:
    """Configuration for Agent Service.

    The Agent Service runs agent.run() in an independent process,
    accepting requests from OpenAIProxyWorkflow (mode="service").

    Attributes:
        host: Host address to bind the service.
        port: Port number for the service.
    """

    host: str = field(
        default="0.0.0.0",
        metadata={"help": "Host address to bind the Agent Service."},
    )
    port: int = field(
        default=8300,
        metadata={"help": "Port number for the Agent Service."},
    )

    def __post_init__(self):
        """Validate configuration values."""
        if self.port <= 0 or self.port > 65535:
            raise ValueError(f"port must be between 1 and 65535, got {self.port}")


@dataclass
class GatewayConfig:
    """Configuration for the Agent Service Gateway.

    The Gateway receives /run_episode requests, queues them internally,
    and dispatches to Agent Worker processes via HTTP.

    Attributes:
        host: Host address to bind the Gateway.
        port: Port number for the Gateway HTTP server.
        queue_size: Maximum number of queued requests before returning 503.
        worker_timeout: Timeout in seconds for Worker HTTP responses.
        health_check_interval: Interval in seconds for passive health checks.
        gateway_cpu: CPU cores for the Gateway process.
        gateway_mem: Memory (GB) for the Gateway process.
        worker_cpu: CPU cores per Agent Worker process.
        worker_mem: Memory (GB) per Agent Worker process.
    """

    host: str = field(
        default="0.0.0.0",
        metadata={"help": "Host address to bind the Gateway."},
    )
    port: int = field(
        default=8300,
        metadata={"help": "Port number for the Gateway."},
    )
    queue_size: int = field(
        default=1000,
        metadata={"help": "Maximum queued requests before returning 503."},
    )
    worker_timeout: float = field(
        default=300.0,
        metadata={"help": "Timeout in seconds for Worker HTTP responses."},
    )
    health_check_interval: float = field(
        default=30.0,
        metadata={"help": "Interval in seconds between passive health checks."},
    )
    gateway_cpu: int = field(
        default=2,
        metadata={"help": "CPU cores for the Gateway process."},
    )
    gateway_mem: int = field(
        default=4,
        metadata={"help": "Memory (GB) for the Gateway process."},
    )
    worker_cpu: int = field(
        default=4,
        metadata={"help": "CPU cores per Agent Worker process."},
    )
    worker_mem: int = field(
        default=4,
        metadata={"help": "Memory (GB) per Agent Worker process."},
    )

    def __post_init__(self):
        """Validate configuration values."""
        if self.port <= 0 or self.port > 65535:
            raise ValueError(f"port must be between 1 and 65535, got {self.port}")
        if self.queue_size < 1:
            raise ValueError(f"queue_size must be at least 1, got {self.queue_size}")
        if self.worker_timeout <= 0:
            raise ValueError(
                f"worker_timeout must be positive, got {self.worker_timeout}"
            )
