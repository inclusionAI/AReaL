"""Agent Service for running agent code in a standalone process.

This module provides the AgentService class, which executes agent.run() in an
independent process.

Architecture Overview
---------------------
The Agent Service is part of a five-process architecture:

1. **Controller**: Orchestrates the training loop
2. **Rollout Worker**: Runs OpenAIProxyWorkflow
3. **Agent Service** (this module): Runs agent.run(), CPU-bound
4. **Proxy Server**: Manages sessions/rewards/trajectories, IO-bound
5. **SGLang Server**: LLM inference, GPU-bound

Communication flow::

    Rollout Worker  --HTTP-->  Agent Service  --HTTP-->  Proxy Server  --HTTP-->  SGLang

This separation enables:
- Independent scaling of agent workloads
- Hot-updating of agent strategies
- Easier testing of agent logic in isolation

Agent Instantiation Modes
-------------------------
The service supports two modes to match RemoteInfEngine._resolve_workflow() behavior:

1. **per-request** (agent_reuse=False, default):
   - Creates a new agent instance for each request
   - Matches behavior when passing agent CLASS to workflow
   - agent_kwargs from request are used for instantiation

2. **shared** (agent_reuse=True):
   - Reuses a single agent instance across all requests
   - Matches behavior when passing agent INSTANCE to workflow
   - agent_init_kwargs from service init are used for instantiation
   - Agents should be stateless (no locking, consistent with inline mode)

Concurrency
-----------
No semaphore or concurrency limiting is used, consistent with inline mode:
- agent.run() is async I/O bound (HTTP calls to Proxy Server)
- Downstream services (Proxy Server, SGLang) have their own flow control
- HTTP server (uvicorn) manages connection limits
"""

from __future__ import annotations

from typing import Any

from areal.utils import logging
from areal.utils.dynamic_import import import_from_string

from .config import AgentServiceConfig

logger = logging.getLogger("AgentService")


class AgentService:
    """Standalone Agent Service process.

    The Agent Service executes agent.run() in an independent process.

    Responsibilities:
    1. Accept /run_episode requests from OpenAIProxyWorkflow (mode="service")
    2. Instantiate agent and execute agent.run()
    3. Return results (rewards) back to the workflow

    Parameters
    ----------
    agent_import_path : str | None
        Import path for the agent class (e.g., "mymodule.MyAgent").
        Required when agent_reuse=True (shared mode).
        Can be None for dynamic mode where each request specifies its own agent.
    config : AgentServiceConfig
        Configuration for the Agent Service.
    agent_reuse : bool
        If True, reuse a single agent instance across all requests (shared mode).
        If False, create a new agent instance for each request (per-request mode).
        Default is False.
    agent_init_kwargs : dict[str, Any] | None
        Keyword arguments for agent instantiation in shared mode.
        Only used when agent_reuse=True.

    Example
    -------
    ```python
    from areal.experimental.agent_service import AgentService, AgentServiceConfig

    config = AgentServiceConfig(port=8300)

    # Per-request mode (default): new agent for each request
    service = AgentService(
        agent_import_path="mymodule.MyAgent",
        config=config,
        agent_reuse=False,
    )

    # Shared mode: reuse single agent instance
    service = AgentService(
        agent_import_path="mymodule.MyAgent",
        config=config,
        agent_reuse=True,
        agent_init_kwargs={"model": "gpt-4"},
    )

    # Dynamic mode: agent_import_path specified per-request
    service = AgentService(
        agent_import_path=None,  # Dynamic mode
        config=config,
        agent_reuse=False,
    )

    # Run a single episode
    async with service:
        result = await service.run_episode(
            data={"prompt": "What is 2+2?"},
            session_url="http://proxy:8000/session/abc123",
            agent_kwargs={"model": "gpt-4"},  # Only used in per-request mode
            agent_import_path="mymodule.MyAgent",  # Optional override
        )
    ```
    """

    def __init__(
        self,
        agent_import_path: str | None,
        config: AgentServiceConfig,
        agent_reuse: bool = False,
        agent_init_kwargs: dict[str, Any] | None = None,
    ):
        # Shared mode requires agent_import_path at init time
        if agent_reuse and not agent_import_path:
            raise ValueError(
                "agent_import_path is required when agent_reuse=True (shared mode)"
            )

        self.agent_import_path = agent_import_path
        self.config = config
        self.agent_reuse = agent_reuse
        self.agent_init_kwargs = agent_init_kwargs or {}
        # Cache multiple agent classes by import path
        self._agent_classes: dict[str, type] = {}
        self._shared_agent: Any | None = None
        self._running = False

    def _get_agent_class(self, import_path: str | None = None) -> type:
        """Get or import the agent class.

        Parameters
        ----------
        import_path : str | None
            Import path to use. If None, uses self.agent_import_path.

        Returns
        -------
        type
            The imported agent class.

        Raises
        ------
        ValueError
            If no import path is available (neither provided nor set at init).
        """
        path = import_path or self.agent_import_path
        if not path:
            raise ValueError(
                "agent_import_path required (either at init or per-request)"
            )

        if path not in self._agent_classes:
            logger.info(f"Importing agent class from: {path}")
            self._agent_classes[path] = import_from_string(path)

        return self._agent_classes[path]

    def _get_or_create_agent(
        self,
        agent_kwargs: dict[str, Any] | None,
        agent_import_path: str | None = None,
    ) -> Any:
        """Get or create an agent instance based on agent_reuse mode.

        Parameters
        ----------
        agent_kwargs : dict[str, Any] | None
            Keyword arguments for agent instantiation (only used in per-request mode).
        agent_import_path : str | None
            Agent class import path for this request.
            Only used in per-request mode (ignored in shared mode).

        Returns
        -------
        Any
            The agent instance.
        """
        if self.agent_reuse:
            # Shared mode: reuse single agent instance (ignores per-request import_path)
            if self._shared_agent is None:
                agent_class = self._get_agent_class()  # Uses init-time config
                self._shared_agent = agent_class(**self.agent_init_kwargs)
                logger.info(
                    f"Created shared agent instance with kwargs: "
                    f"{list(self.agent_init_kwargs.keys())}"
                )
            return self._shared_agent
        else:
            # Per-request mode: supports dynamic import_path
            agent_class = self._get_agent_class(agent_import_path)
            return agent_class(**(agent_kwargs or {}))

    async def run_episode(
        self,
        data: dict[str, Any],
        session_url: str,
        agent_kwargs: dict[str, Any] | None = None,
        agent_import_path: str | None = None,
    ) -> Any:
        """Execute a single agent episode.

        In per-request mode (agent_reuse=False), creates a new agent for each request.
        In shared mode (agent_reuse=True), reuses the same agent instance.

        Parameters
        ----------
        data : dict[str, Any]
            Input data for the agent (e.g., prompt, metadata).
        session_url : str
            The Proxy Server session URL for this episode.
        agent_kwargs : dict[str, Any] | None
            Keyword arguments for agent instantiation (only used in per-request mode).
        agent_import_path : str | None
            Agent class import path for this request.
            Takes priority over init-time agent_import_path.
            Only used in per-request mode (ignored in shared mode).

        Returns
        -------
        Any
            The result from agent.run() (typically rewards dict or float).

        Raises
        ------
        RuntimeError
            If the service has not been started.
        """
        if not self._running:
            raise RuntimeError("AgentService is not running. Call start() first.")

        # Get or create agent based on agent_reuse mode
        agent = self._get_or_create_agent(agent_kwargs, agent_import_path)

        # Execute agent.run()
        # The agent should use session_url as base_url for OpenAI API calls
        return await agent.run(data, base_url=session_url)

    async def start(self) -> None:
        """Start the Agent Service.

        Initializes resources and marks the service as running.
        In shared mode (agent_reuse=True), also pre-creates the shared agent instance.
        """
        if self._running:
            logger.warning("AgentService is already running")
            return

        mode_str = "shared" if self.agent_reuse else "per-request"
        path_str = self.agent_import_path or "(dynamic)"
        logger.info(
            f"Starting AgentService with agent_import_path={path_str}, mode={mode_str}"
        )

        # Only pre-import if agent_import_path is specified
        if self.agent_import_path:
            self._get_agent_class()

        # In shared mode, pre-create the agent instance
        if self.agent_reuse:
            self._get_or_create_agent(None)

        self._running = True
        logger.info("AgentService started successfully")

    async def stop(self) -> None:
        """Stop the Agent Service.

        Cleans up resources and marks the service as stopped.
        """
        if not self._running:
            logger.warning("AgentService is not running")
            return

        logger.info("Stopping AgentService...")
        self._running = False
        self._shared_agent = None  # Allow GC of shared agent instance
        self._agent_classes.clear()  # Clear cached agent classes
        logger.info("AgentService stopped")

    @property
    def is_running(self) -> bool:
        """Check if the service is running."""
        return self._running

    async def health_check(self) -> dict[str, Any]:
        """Perform a health check.

        Returns
        -------
        dict[str, Any]
            Health status including running state and resource info.
        """
        return {
            "status": "ok" if self._running else "stopped",
            "running": self._running,
            "agent_import_path": self.agent_import_path or "(dynamic)",
            "agent_reuse": self.agent_reuse,
        }

    async def __aenter__(self) -> AgentService:
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()
