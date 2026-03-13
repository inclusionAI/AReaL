"""Configuration for AEnvironment integration."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class AenvConfig:
    """Configuration for aenvironment-based tool execution.

    Attributes:
        aenv_url: AEnvironment control endpoint (without port suffixes).
        env_name: Environment name in AEnvironment.
        datasource: Optional datasource mounted by the environment runtime.
        ttl: Environment instance time-to-live (e.g., "30m").
        environment_variables: Environment variables passed to the environment instance.
        arguments: Extra command-line arguments for the environment instance.
        timeout: General request timeout in seconds.
        startup_timeout: Environment startup timeout in seconds.
        tool_call_timeout: Tool call timeout in seconds.
        max_retries: Maximum retry attempts for retriable tool call failures.
        retry_delay: Initial delay between retries in seconds.
        auto_release: Whether to release environment resources on adapter cleanup.
        turn_discount: Discount factor used for multi-turn reward propagation.
        tool_error_policy: Tool failure policy: raise immediately or append an error message.
    """

    aenv_url: str = field(
        default="http://localhost",
        metadata={"help": "AEnvironment base URL"},
    )
    env_name: str = field(
        default="default",
        metadata={"help": "AEnvironment environment name"},
    )
    datasource: str = field(
        default="",
        metadata={"help": "Optional datasource mounted by the environment"},
    )
    ttl: str = field(
        default="30m",
        metadata={"help": "Environment instance TTL"},
    )
    environment_variables: dict[str, str] = field(
        default_factory=dict,
        metadata={"help": "Environment variables passed to environment instance"},
    )
    arguments: list[str] = field(
        default_factory=list,
        metadata={"help": "Extra arguments passed to environment entrypoint"},
    )

    timeout: float = field(
        default=30.0,
        metadata={"help": "General request timeout in seconds"},
    )
    startup_timeout: float = field(
        default=120.0,
        metadata={"help": "Environment startup timeout in seconds"},
    )
    tool_call_timeout: float = field(
        default=30.0,
        metadata={"help": "Tool call timeout in seconds"},
    )
    max_retries: int = field(
        default=2,
        metadata={"help": "Maximum retries for retriable tool failures"},
    )
    retry_delay: float = field(
        default=0.5,
        metadata={"help": "Initial retry delay in seconds"},
    )
    auto_release: bool = field(
        default=True,
        metadata={"help": "Release environment resources on adapter cleanup"},
    )

    turn_discount: float = field(
        default=0.9,
        metadata={"help": "Reward discount factor for multi-turn interactions"},
    )
    tool_error_policy: Literal["raise", "append_error"] = field(
        default="append_error",
        metadata={"help": "Tool failure policy"},
    )

    def __post_init__(self):
        """Validate configuration values."""
        if not self.aenv_url:
            raise ValueError("aenv_url must be non-empty")
        if not self.env_name:
            raise ValueError("env_name must be non-empty")

        if self.timeout <= 0:
            raise ValueError(f"timeout must be positive, got {self.timeout}")
        if self.startup_timeout <= 0:
            raise ValueError(
                f"startup_timeout must be positive, got {self.startup_timeout}"
            )
        if self.tool_call_timeout <= 0:
            raise ValueError(
                f"tool_call_timeout must be positive, got {self.tool_call_timeout}"
            )

        if self.max_retries < 0:
            raise ValueError(
                f"max_retries must be non-negative, got {self.max_retries}"
            )
        if self.retry_delay < 0:
            raise ValueError(
                f"retry_delay must be non-negative, got {self.retry_delay}"
            )

        if not 0.0 <= self.turn_discount <= 1.0:
            raise ValueError(
                f"turn_discount must be within [0, 1], got {self.turn_discount}"
            )
