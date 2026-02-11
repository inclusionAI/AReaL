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
        workers: Number of worker processes for handling concurrent requests.
    """

    host: str = field(
        default="0.0.0.0",
        metadata={"help": "Host address to bind the Agent Service."},
    )
    port: int = field(
        default=8300,
        metadata={"help": "Port number for the Agent Service."},
    )
    workers: int = field(
        default=1,
        metadata={"help": "Number of worker processes (default: 1)."},
    )

    def __post_init__(self):
        """Validate configuration values."""
        if self.port <= 0 or self.port > 65535:
            raise ValueError(f"port must be between 1 and 65535, got {self.port}")
        if self.workers < 1:
            raise ValueError(f"workers must be at least 1, got {self.workers}")
