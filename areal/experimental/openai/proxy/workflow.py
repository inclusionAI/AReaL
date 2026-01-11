from __future__ import annotations

import asyncio
import atexit
import os
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING, Any

import aiohttp

from areal.api.workflow_api import AgentWorkflow, RolloutWorkflow
from areal.core import workflow_context
from areal.utils import logging, stats_tracker
from areal.utils.perf_tracer import session_context, trace_session

from .client_session import OpenAIProxyClientSession

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


def _wrap_run(agent, data, extra_envs):
    for key, value in extra_envs.items():
        os.environ[key] = value

    try:
        return asyncio.run(agent.run(None, data))
    except Exception:
        logger.error(f"Agent task failed: {traceback.format_exc()}")
        raise


class OpenAIProxyWorkflow(RolloutWorkflow):
    """Workflow that wraps an AgentWorkflow and manages OpenAI proxy interaction.

    This workflow uses HTTP-only mode where all operations go through HTTP
    to a remote ProxyRolloutServer.

    Parameters
    ----------
    mode : str
        Must be "offline" (agent runs locally in subprocess). Online mode
        is not currently supported.
    agent : AgentWorkflow
        The agent workflow to run
    proxy_addr : str
        HTTP address of remote proxy server (required)
    discount : float
        Discount factor for reward propagation
    export_style : str
        Style for exporting interactions ("individual" or "merged")
    """

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
        # extra_envs = {
        #     "OPENAI_BASE_URL": base_url,
        # }
        # executor = _get_executor()
        # fut = executor.submit(_wrap_run, self.agent, data, extra_envs)
        try:
            return await self.agent.run(base_url, data)
            # return await asyncio.wrap_future(fut)
        except Exception:
            logger.error(f"Agent task failed: {traceback.format_exc()}")
            raise

    async def _grant_capacity(self, session: aiohttp.ClientSession) -> None:
        """Grant capacity via HTTP."""
        url = f"{self.proxy_addr}/grant_capacity"
        async with session.post(url) as resp:
            resp.raise_for_status()

    @session_context()
    async def arun_episode(
        self, engine: TRolloutEngine, data: dict[str, Any]
    ) -> dict[str, InteractionWithTokenLogpReward] | None:
        """Run an episode of the agent workflow.

        Parameters
        ----------
        engine : TRolloutEngine
            The inference engine (not used in HTTP-only mode)
        data : dict[str, Any]
            Input data for the episode

        Returns
        -------
        dict[str, InteractionWithTokenLogpReward] | None
            Dictionary of interactions with rewards, or None if failed
        """
        task_id = workflow_context.get().task_id

        tik = time.time()
        async with aiohttp.ClientSession() as http_session:
            t1 = time.time()
            # Grant capacity using the shared session
            await self._grant_capacity(http_session)
            t2 = time.time()

            mode = self.mode
            assert mode == "offline"
            proxy_client: OpenAIProxyClientSession | None = None

            # Create proxy client (start RL session)
            proxy_client = OpenAIProxyClientSession(
                session=http_session,
                base_url=self.proxy_addr,
                task_id=str(task_id),
            )
            async with proxy_client:
                t4 = time.time()
                rewards = await self._run_agent(proxy_client.session_url, data)
                t5 = time.time()

                if isinstance(rewards, dict):
                    for completion_id, reward in rewards.items():
                        await proxy_client.set_reward(completion_id, reward)
                elif isinstance(rewards, float):
                    await proxy_client.set_last_reward(rewards)
                else:
                    raise ValueError(f"Invalid reward type: {type(rewards)}")
                t6 = time.time()
            t7 = time.time()

            interactions = await proxy_client.export_interactions(
                discount=self.discount,
                style=self.export_style,
            )
            t8 = time.time()

        # Record stats
        last_id = list(interactions.keys())[-1] if interactions else None
        if last_id and interactions:
            last_reward = interactions[last_id].reward
            stats_tracker.get(workflow_context.stat_scope()).scalar(reward=last_reward)
        t9 = time.time()

        f"{self.proxy_addr}, aiohttp session: {t1 - tik:.2f}, "
        f"grant cap: {t2 - t1:.2f}, "
        f"start session: {t4 - t2:.2f}, "
        f"run agent: {t5 - t4:.2f}, "
        f"set reward: {t6 - t5:.2f}, end_session: {t7 - t6:.2f}, export: {t8 - t7:.2f}, "
        f"log reward: {t9 - t8:.2f}"

        return interactions
