from __future__ import annotations

import asyncio
import os
import traceback
from typing import TYPE_CHECKING

from areal.api.workflow_api import AgentWorkflow, RolloutWorkflow
from areal.core import workflow_context
from areal.utils import logging

from .client_session import OpenAIProxyClientSession
from .server import OpenAIProxyServer

if TYPE_CHECKING:
    from ..client import TRolloutEngine

logger = logging.getLogger("OpenAIProxyWorkflow")


def _wrap_run_agent(agent, extra_envs, data):
    # Save original environment
    original_env = {key: os.environ.get(key) for key in extra_envs.keys()}

    try:
        # Set new environment variables
        for key, value in extra_envs.items():
            os.environ[key] = value

        return asyncio.run(agent.run(data))
    except Exception:
        logger.error(traceback.format_exc())
        raise
    finally:
        # Restore original environment (optional)
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


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
                try:
                    rewards = await self.agent.run(session.session_url, data)
                except Exception:
                    logger.error(f"Agent task failed: {traceback.format_exc()}")
                    raise
                session_id = session.session_id
                if isinstance(rewards, dict):
                    for completion_id, reward in rewards.items():
                        self.proxy_server.set_reward(session_id, completion_id, reward)
                elif isinstance(rewards, float):
                    self.proxy_server.set_last_reward(session_id, rewards)
                else:
                    raise ValueError(f"Invalid reward type: {type(rewards)}")

        return await self.proxy_server.wait_for_session(
            session_id, discount=self.discount, style=self.export_style
        )
