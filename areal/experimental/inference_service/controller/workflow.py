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
        assert self.agent is not None
        task_id = workflow_context.get().task_id

        http_client = await workflow_context.get_httpx_client()
        result = await self.agent.run(
            data,
            base_url=self.gateway_addr,
            http_client=http_client,
            api_key=self._admin_api_key,
            task_id=str(task_id),
        )

        if not isinstance(result, dict):
            raise TypeError(
                f"Agent.run() must return a dict with 'session_id', "
                f"'trajectory_id', and 'reward' keys, got {type(result)}"
            )
        _REQUIRED_KEYS = {"session_id", "trajectory_id", "reward"}
        missing = _REQUIRED_KEYS - result.keys()
        if missing:
            raise KeyError(f"Agent.run() result is missing required keys: {missing}")

        session_id = result["session_id"]
        trajectory_id = result["trajectory_id"]
        agent_reward = float(result["reward"])

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
        if abs(last_reward - agent_reward) > 1e-6:
            logger.warning(
                "Session %s: agent reported reward %.6f but exported "
                "interaction has %.6f; using exported value.",
                session_id,
                agent_reward,
                last_reward,
            )
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
