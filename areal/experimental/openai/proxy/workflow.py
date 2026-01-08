from __future__ import annotations

import asyncio
import atexit
import os
import threading
import traceback
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING

from areal.api.workflow_api import AgentWorkflow, RolloutWorkflow
from areal.core import workflow_context
from areal.utils import logging, stats_tracker
from areal.utils.perf_tracer import session_context, trace_session

from .client_session import OpenAIProxyClientSession
from .server import OpenAIProxyServer

if TYPE_CHECKING:
    from ..client import TRolloutEngine

logger = logging.getLogger("OpenAIProxyWorkflow")


# Lazy-initialized thread pool for async HTTP requests
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
    def __init__(
        self,
        mode: str,
        agent: AgentWorkflow,
        proxy_server: OpenAIProxyServer,
        discount: float = 1.0,
        export_style: str = "individual",
    ):
        self.mode = mode
        self.proxy_server = proxy_server
        self.agent = agent

        self.discount = discount
        self.export_style = export_style

    @trace_session("run_agent")
    async def _run_agent(self, base_url: str, data: dict):
        extra_envs = {
            "OPENAI_BASE_URL": base_url,
        }
        executor = _get_executor()
        fut = executor.submit(_wrap_run, self.agent, data, extra_envs)
        try:
            return await asyncio.wrap_future(fut)
        except Exception:
            logger.error(f"Agent task failed: {traceback.format_exc()}")
            raise

    @session_context()
    async def arun_episode(self, engine: TRolloutEngine, data):
        # Ensure that we own the same engine instance
        task_id = workflow_context.get().task_id

        # Grant capacity for staleness control
        await self.proxy_server.grant_capacity()

        mode = self.mode
        if mode == "online":
            # `engine` and `data` are not used here
            session_id = await self.proxy_server.fetch_next_session()

        if mode == "offline":
            async with OpenAIProxyClientSession(
                base_url=self.proxy_server.public_addr, task_id=str(task_id)
            ) as session:
                rewards = await self._run_agent(session.session_url, data)

                session_id = session.session_id
                if isinstance(rewards, dict):
                    for completion_id, reward in rewards.items():
                        self.proxy_server.set_reward(session_id, completion_id, reward)
                elif isinstance(rewards, float):
                    self.proxy_server.set_last_reward(session_id, rewards)
                else:
                    raise ValueError(f"Invalid reward type: {type(rewards)}")

            # Pop a session id from the server queue and ignore it.
            _ = await self.proxy_server.fetch_next_session()

        session_data = await self.proxy_server.wait_for_session(session_id)
        last_id = session_data.completions.last_interaction_id
        interactions = session_data.completions.export_interactions(
            reward_discount=self.discount, style=self.export_style
        )

        # Record the last reward in wandb/tensorboard
        last_reward = interactions[last_id].reward
        stats_tracker.get(workflow_context.stat_scope()).scalar(reward=last_reward)

        return interactions
