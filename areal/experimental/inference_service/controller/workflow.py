# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import aiohttp

from areal.api.workflow_api import RolloutWorkflow
from areal.experimental.openai.proxy.server import deserialize_interactions
from areal.infra import workflow_context
from areal.utils import logging, stats_tracker

if TYPE_CHECKING:
    from areal.api.engine_api import InferenceEngine
    from areal.experimental.inference_service.controller.controller import (
        GatewayInferenceController,
    )
    from areal.experimental.openai.types import InteractionWithTokenLogpReward

logger = logging.getLogger("InferenceServiceWorkflow")

_GRANT_CAPACITY_PATHNAME = "grant_capacity"
_RL_START_SESSION_PATHNAME = "rl/start_session"
_RL_SET_REWARD_PATHNAME = "rl/set_reward"
_EXPORT_TRAJECTORIES_PATHNAME = "export_trajectories"


class InferenceServiceWorkflow(RolloutWorkflow):
    def __init__(
        self,
        controller: GatewayInferenceController,
        agent: Any | None = None,
        gateway_addr: str = "",
        admin_api_key: str = "areal-admin-key",
        discount: float = 1.0,
        export_style: str = "individual",
        timeout: float | None = None,
    ):
        self.controller = controller
        self.agent = agent
        self.gateway_addr = gateway_addr.rstrip("/") if gateway_addr else ""
        self._admin_api_key = admin_api_key
        self.discount = discount
        self.export_style = export_style
        self.timeout = timeout

    async def _grant_capacity(self, session: aiohttp.ClientSession) -> None:
        url = f"{self.gateway_addr}/{_GRANT_CAPACITY_PATHNAME}"
        headers = {"Authorization": f"Bearer {self._admin_api_key}"}
        async with session.post(url, headers=headers) as resp:
            resp.raise_for_status()

    async def _start_session(
        self, session: aiohttp.ClientSession, task_id: str
    ) -> tuple[str, str]:
        url = f"{self.gateway_addr}/{_RL_START_SESSION_PATHNAME}"
        headers = {"Authorization": f"Bearer {self._admin_api_key}"}
        payload = {"task_id": task_id}
        async with session.post(url, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            data = await resp.json()
        return data["session_id"], data["api_key"]

    async def _set_last_reward(
        self,
        session: aiohttp.ClientSession,
        reward: float,
        session_api_key: str,
    ) -> int | None:
        url = f"{self.gateway_addr}/{_RL_SET_REWARD_PATHNAME}"
        headers = {"Authorization": f"Bearer {session_api_key}"}
        payload: dict[str, Any] = {"interaction_id": None, "reward": reward}
        async with session.post(url, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            data = await resp.json()
        trajectory_id = data.get("trajectory_id")
        return int(trajectory_id) if trajectory_id is not None else None

    async def _export_interactions(
        self,
        session: aiohttp.ClientSession,
        session_id: str,
        trajectory_id: int | None = None,
    ) -> dict[str, InteractionWithTokenLogpReward]:
        url = f"{self.gateway_addr}/{_EXPORT_TRAJECTORIES_PATHNAME}"
        headers = {"Authorization": f"Bearer {self._admin_api_key}"}
        payload = {
            "session_id": session_id,
            "trajectory_id": trajectory_id,
            "discount": self.discount,
            "style": self.export_style,
        }
        async with session.post(url, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            data = await resp.json()

        return deserialize_interactions(data["interactions"])

    async def arun_episode(
        self,
        engine: InferenceEngine,
        data: dict[str, Any],
    ) -> dict[str, InteractionWithTokenLogpReward] | None:
        del engine
        http_session = await workflow_context.get_aiohttp_session()
        await self._grant_capacity(http_session)

        if self.agent is not None:
            return await self._run_offline(http_session, data)
        return await self._run_online(http_session)

    async def _run_offline(
        self,
        http_session: aiohttp.ClientSession,
        data: dict[str, Any],
    ) -> dict[str, InteractionWithTokenLogpReward] | None:
        task_id = workflow_context.get().task_id
        session_id, session_api_key = await self._start_session(
            http_session, str(task_id)
        )

        assert self.agent is not None
        finished = False
        trajectory_id: int | None = None
        try:
            http_client = await workflow_context.get_httpx_client()
            rewards = await self.agent.run(
                data,
                base_url=self.gateway_addr,
                http_client=http_client,
                api_key=session_api_key,
            )

            if isinstance(rewards, dict):
                final_reward = float(list(rewards.values())[-1] if rewards else 0.0)
            elif isinstance(rewards, (int, float)):
                final_reward = float(rewards)
            else:
                raise ValueError(f"Invalid reward type: {type(rewards)}")

            trajectory_id = await self._set_last_reward(
                http_session, final_reward, session_api_key
            )
            finished = True
        except Exception:
            logger.warning("Agent task failed. This trajectory will be rejected.")
            if not finished:
                try:
                    await self._set_last_reward(http_session, 0.0, session_api_key)
                except Exception:
                    logger.warning(
                        "Failed to finish session %s after agent failure",
                        session_id,
                    )
            raise

        interactions = await self._export_interactions(
            http_session,
            session_id,
            trajectory_id=trajectory_id,
        )
        if not interactions:
            logger.warning(
                "Session %s has no interactions, trajectory will be rejected.",
                session_id,
            )
            return None

        last_id = list(interactions.keys())[-1]
        last_reward = interactions[last_id].reward
        stats_tracker.get(workflow_context.stat_scope()).scalar(reward=last_reward)
        return interactions

    async def _run_online(
        self,
        http_session: aiohttp.ClientSession,
    ) -> dict[str, InteractionWithTokenLogpReward] | None:
        logger.debug("Waiting for next ready online trajectory")
        export_request = await self.controller.wait_for_online_trajectory(
            timeout=self.timeout
        )
        if not export_request:
            return None

        interactions = await self._export_interactions(
            http_session,
            export_request["session_id"],
            trajectory_id=export_request["trajectory_id"],
        )
        if not interactions:
            return None

        last_id = next(reversed(interactions))
        last_reward = interactions[last_id].reward
        stats_tracker.get(workflow_context.stat_scope()).scalar(reward=last_reward)
        return interactions
