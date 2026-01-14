from __future__ import annotations

import asyncio
import atexit
import os
import threading
import traceback
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING, Any

import aiohttp

from areal.api.workflow_api import AgentWorkflow, RolloutWorkflow
from areal.core import workflow_context
from areal.utils import logging, stats_tracker
from areal.utils.perf_tracer import session_context, trace_session

from .client_session import OpenAIProxyClient

if TYPE_CHECKING:
    from ..client import TRolloutEngine
    from ..types import InteractionWithTokenLogpReward

logger = logging.getLogger("OpenAIProxyWorkflow")


# Lazy-initialized process pool for running agent tasks
_executor: ProcessPoolExecutor | None = None
_executor_lock = threading.Lock()


def _get_executor() -> ProcessPoolExecutor:
    """Get or create the shared process pool executor."""
    global _executor
    if _executor is None:
        with _executor_lock:
            if _executor is None:
                _executor = ProcessPoolExecutor(max_workers=4)
                # Register cleanup on process exit
                atexit.register(_shutdown_executor)
    return _executor


def _shutdown_executor() -> None:
    """Shutdown the shared thread pool executor if it exists.

    Called via atexit at process exit, when no other threads should be
    accessing the executor.
    """
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=False)
        _executor = None


def _wrap_run(agent: AgentWorkflow, data: dict[str, Any], extra_envs: dict[str, str]):
    for key, value in extra_envs.items():
        os.environ[key] = value
    return asyncio.run(agent.run(data))


class OpenAIProxyWorkflow(RolloutWorkflow):
    def __init__(
        self,
        mode: str,
        agent: AgentWorkflow,
        proxy_addr: str,
        discount: float = 1.0,
        export_style: str = "individual",
    ):
        self.mode = mode
        self.agent = agent
        self.proxy_addr = proxy_addr
        self.discount = discount
        self.export_style = export_style

    @trace_session("run_agent")
    async def _run_agent(self, base_url: str, data: dict):
        if self.mode == "inline":
            http_client = await workflow_context.get_httpx_client()
            extra_kwargs = {
                "base_url": base_url,
                "http_client": http_client,
            }
            return await self.agent.run(data, **extra_kwargs)
        if self.mode == "subproc":
            extra_envs = {
                "OPENAI_BASE_URL": base_url,
                "OPENAI_API_KEY": "DUMMY",
            }
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                _get_executor(),
                _wrap_run,
                self.agent,
                data,
                extra_envs,
            )
        raise ValueError(f"Unsupported mode: {self.mode}")

    async def _grant_capacity(self, session: aiohttp.ClientSession) -> None:
        """Grant capacity via HTTP."""
        url = f"{self.proxy_addr}/grant_capacity"
        async with session.post(url) as resp:
            resp.raise_for_status()

    @session_context()
    async def arun_episode(
        self, engine: TRolloutEngine, data: dict[str, Any]
    ) -> dict[str, InteractionWithTokenLogpReward] | None:
        task_id = workflow_context.get().task_id

        http_session = await workflow_context.get_aiohttp_session()

        # Grant capacity for clients, otherwise agent sessions are rejected.
        # Designed for online mode. Users' requests do not have any staleness
        # control, which may be detrimental to RL training. We use a hacky way
        # to control the staleness. The staleness is always explicitly controlled
        # by the rollout controller and staleness manager. If the code runs
        # to this point, it means that we are within the allowed staleness window,
        # so we can grant capacity to let the agent session proceed.
        await self._grant_capacity(http_session)

        proxy_client: OpenAIProxyClient | None = None

        # Create proxy client to manage the lifecycle of an RL session
        # An RL session creates a unique URL for storing interactions
        # for this agentic trajectory.
        proxy_client = OpenAIProxyClient(
            session=http_session,
            base_url=self.proxy_addr,
            task_id=str(task_id),
        )
        async with proxy_client:
            # Run the user code.
            try:
                rewards = await self._run_agent(proxy_client.session_url, data)
            except Exception:
                logger.warning(
                    f"Agent task failed. This trajectory will be rejected. Error: {traceback.format_exc()}"
                )
                raise

            # Assign rewards back according to user code output
            if isinstance(rewards, dict):
                for completion_id, reward in rewards.items():
                    await proxy_client.set_reward(completion_id, reward)
            elif isinstance(rewards, float):
                await proxy_client.set_last_reward(rewards)
            else:
                raise ValueError(f"Invalid reward type: {type(rewards)}")

        # Apply turn-level discount and export interactions
        interactions = await proxy_client.export_interactions(
            discount=self.discount,
            style=self.export_style,
        )

        # Record stats
        last_id = list(interactions.keys())[-1] if interactions else None
        if last_id and interactions:
            last_reward = interactions[last_id].reward
            stats_tracker.get(workflow_context.stat_scope()).scalar(reward=last_reward)

        return interactions
