"""Configuration for the Agent Service.

Follows the same conventions as :class:`OpenAIProxyConfig`
(``areal/api/cli_args.py:1552``): field ordering, metadata format, and
``__post_init__`` validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentServiceConfig:
    """Configuration for the Agent Service.

    Attributes
    ----------
    agent_import_path : str
        Import path for the agent class (e.g.
        ``myproject.agents.WebSearchAgent``).  The class must implement the
        :class:`AgentRunnable` protocol.
    num_workers : int
        Number of Agent Workers to launch.
    max_concurrent_sessions : int
        Maximum concurrent sessions per worker.
    session_timeout_seconds : int
        Session timeout in seconds.
    admin_api_key : str
        Admin API key for the Agent Gateway.
    """

    # Required: import path for the agent class
    agent_import_path: str = field(
        default="",
        metadata={
            "help": (
                "Import path for the agent class, e.g., "
                "'myproject.agents.WebSearchAgent'. "
                "The class must implement the AgentRunnable protocol "
                "(async def run(request, *, llm, emitter))."
            ),
        },
    )

    # Worker pool settings
    num_workers: int = field(
        default=1,
        metadata={"help": "Number of Agent Workers to launch."},
    )
    max_concurrent_sessions: int = field(
        default=8,
        metadata={
            "help": "Maximum concurrent sessions per worker.",
        },
    )

    # Session settings
    session_timeout_seconds: int = field(
        default=3600,
        metadata={
            "help": ("Session timeout in seconds. Matches OpenAIProxyConfig default."),
        },
    )

    # Admin
    admin_api_key: str = field(
        default="areal-admin-key",
        metadata={
            "help": (
                "Admin API key for the Agent Gateway. "
                "WARNING: Change from default for non-local deployments."
            ),
        },
    )

    def __post_init__(self) -> None:
        if not self.agent_import_path:
            raise ValueError(
                "agent_import_path must be set to a valid import path, "
                "e.g., 'myproject.agents.WebSearchAgent'"
            )
        if self.num_workers <= 0:
            raise ValueError(f"num_workers must be positive, got {self.num_workers}")
        if self.max_concurrent_sessions <= 0:
            raise ValueError(
                f"max_concurrent_sessions must be positive, "
                f"got {self.max_concurrent_sessions}"
            )
        if not self.admin_api_key or not self.admin_api_key.strip():
            raise ValueError("admin_api_key must not be empty")
