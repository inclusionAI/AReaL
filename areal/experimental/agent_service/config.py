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
    task_timeout: float = field(
        default=300.0,
        metadata={
            "help": "Timeout in seconds for a single run_episode() call. Workers send an error result if exceeded."
        },
    )

    def __post_init__(self):
        """Validate configuration values."""
        if self.port <= 0 or self.port > 65535:
            raise ValueError(f"port must be between 1 and 65535, got {self.port}")
        if self.task_timeout <= 0:
            raise ValueError(f"task_timeout must be positive, got {self.task_timeout}")


@dataclass
class GatewayConfig:
    """Configuration for the Agent Service Gateway with ZMQ bridge architecture.

    The Gateway acts as a transparent reverse proxy with ZMQ-based request/response
    routing. It distributes requests from clients to Agent Worker processes via
    a ZMQ Router (DEALER pattern) and collects results via another ZMQ Router.
    """

    task_timeout: float = field(
        default=300.0,
        metadata={"help": "Timeout in seconds for Worker tasks to complete."},
    )
    result_ttl: float = field(
        default=300.0,
        metadata={
            "help": "Time-to-live in seconds for completed results before cleanup."
        },
    )
    zmq_req_frontend_addr: str = field(
        default="tcp://*:0",
        metadata={
            "help": "ZMQ Router PULL socket address for incoming requests from Gateway."
        },
    )
    zmq_req_backend_addr: str = field(
        default="tcp://*:0",
        metadata={
            "help": "ZMQ Router ROUTER socket address for Workers to connect via DEALER."
        },
    )
    zmq_res_frontend_addr: str = field(
        default="tcp://*:0",
        metadata={
            "help": "ZMQ Router PULL socket address for incoming results from Workers."
        },
    )
    zmq_res_backend_addr: str = field(
        default="tcp://*:0",
        metadata={
            "help": "ZMQ Router PUSH socket address for Gateway to pull results."
        },
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
        if self.task_timeout <= 0:
            raise ValueError(f"task_timeout must be positive, got {self.task_timeout}")
        if self.result_ttl <= 0:
            raise ValueError(f"result_ttl must be positive, got {self.result_ttl}")
