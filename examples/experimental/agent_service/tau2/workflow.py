"""Tau2 workflow using the experimental agent service."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from openai import AsyncOpenAI

from areal.api.workflow_api import RolloutWorkflow
from areal.infra import workflow_context
from areal.utils import logging, stats_tracker

if TYPE_CHECKING:
    from areal.api.engine_api import InferenceEngine
    from areal.experimental.agent_service.controller.controller import AgentController
    from areal.experimental.openai.types import InteractionWithTokenLogpReward

logger = logging.getLogger("Tau2AgentServiceWorkflow")


def _extract_response_text(response: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in response.get("output", []):
        if item.get("type") == "message":
            for block in item.get("content", []):
                if block.get("type") == "output_text":
                    parts.append(block.get("text", ""))
    return "\n".join(parts).strip()


def _extract_completion_text(completion: Any) -> str:
    choice = completion.choices[0]
    message = getattr(choice, "message", None)
    content = getattr(message, "content", "") if message is not None else ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and item.get("text"):
                    parts.append(str(item["text"]))
            else:
                text = getattr(item, "text", None)
                if text:
                    parts.append(str(text))
        return "\n".join(parts).strip()
    return str(content).strip()


class Tau2AgentServiceWorkflow(RolloutWorkflow):
    def __init__(
        self,
        agent_controller: AgentController,
        inference_gateway_addr: str,
        inference_admin_api_key: str,
        inference_model: str = "",
        econfig: dict[str, Any] | None = None,
        gen_args: dict[str, Any] | None = None,
        timeout: float = 600.0,
        max_turns: int = 10,
        discount: float = 1.0,
        export_style: str = "individual",
    ) -> None:
        from examples.tau2.utils import Tau2EnvConfig

        self.agent_controller = agent_controller
        self.inference_gateway_addr = inference_gateway_addr.rstrip("/")
        self.inference_admin_api_key = inference_admin_api_key
        self.inference_model = inference_model
        self.econfig = (
            Tau2EnvConfig(**econfig)
            if isinstance(econfig, dict)
            else (econfig or Tau2EnvConfig())
        )
        self.gen_args = gen_args or {}
        self.timeout = timeout
        self.max_turns = max_turns
        self.discount = discount
        self.export_style = export_style

    async def _run_dialog(
        self,
        data: dict[str, Any],
        agent_session_id: str,
    ) -> float:
        from tau2.data_model.message import AssistantMessage, UserMessage
        from tau2.data_model.simulation import SimulationRun, TerminationReason
        from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation

        from examples.tau2.agent import _get_task
        from examples.tau2.utils import Tau2EnvConfig

        econfig = self.econfig
        if "econfig" in data:
            econfig = Tau2EnvConfig(**data["econfig"])

        task = _get_task(
            domain=econfig.domain,
            task_id=data["task_id"],
            split=data.get("split", "train"),
        )
        first_user_message = str(data.get("prompt") or task.user_scenario).strip()
        if not first_user_message:
            raise ValueError("data.prompt or task.user_scenario is required")

        user_client = None
        if not econfig.solo_mode:
            if not econfig.user_llm_base_url:
                raise ValueError(
                    "econfig.user_llm_base_url is required when solo_mode is false"
                )
            user_client = AsyncOpenAI(
                base_url=econfig.user_llm_base_url,
                api_key="dummy",
                max_retries=3,
                timeout=120.0,
            )

        tau2_messages: list[UserMessage | AssistantMessage] = []
        chat_history: list[dict[str, str]] = [
            {"role": "user", "content": first_user_message}
        ]
        next_user_message = first_user_message

        for turn_idx in range(self.max_turns):
            response = await self.agent_controller.step(
                next_user_message, agent_session_id
            )
            agent_text = _extract_response_text(response) or "(no response)"

            tau2_messages.append(
                UserMessage(
                    role="user",
                    content=next_user_message,
                    turn_idx=len(tau2_messages),
                )
            )
            tau2_messages.append(
                AssistantMessage(
                    role="assistant",
                    content=agent_text,
                    turn_idx=len(tau2_messages),
                )
            )

            if turn_idx + 1 >= self.max_turns or user_client is None:
                break

            chat_history.append({"role": "assistant", "content": agent_text})
            completion = await user_client.chat.completions.create(
                model=econfig.user_llm or "dummy",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are simulating the tau2 user described below. "
                            "Respond with the user's next message only, in one turn, "
                            "based on the conversation so far.\n\n"
                            f"User scenario:\n{task.user_scenario}"
                        ),
                    },
                    *chat_history,
                ],
                **(econfig.user_llm_args or {}),
            )
            next_user_message = _extract_completion_text(completion)
            if not next_user_message:
                break
            chat_history.append({"role": "user", "content": next_user_message})

        simulation = SimulationRun(
            id=f"agent-svc-{task.id}",
            task_id=task.id,
            messages=tau2_messages,
            start_time="",
            end_time="",
            duration=0.0,
            termination_reason=TerminationReason.USER_STOP,
        )
        reward_info = evaluate_simulation(
            simulation=simulation,
            task=task,
            evaluation_type=EvaluationType.ALL,
            solo_mode=econfig.solo_mode,
            domain=econfig.domain,
        )
        return float(reward_info.reward)

    async def arun_episode(
        self,
        engine: InferenceEngine,
        data: dict[str, Any],
    ) -> dict[str, InteractionWithTokenLogpReward] | None:
        del engine
        task_id = str(data.get("task_id") or workflow_context.get().task_id)
        session = await self.agent_controller.start_session(
            task_id=task_id,
            inference_gateway_addr=self.inference_gateway_addr,
            inference_admin_api_key=self.inference_admin_api_key,
            inference_model=self.inference_model,
        )

        trajectory_id: int | None = None
        finished = False
        try:
            reward = await asyncio.wait_for(
                self._run_dialog(data, session["session_id"]),
                timeout=self.timeout,
            )
            reward_result = await self.agent_controller.set_reward(
                reward,
                session["session_id"],
            )
            raw_trajectory_id = reward_result.get("trajectory_id")
            trajectory_id = (
                int(raw_trajectory_id) if raw_trajectory_id is not None else None
            )
            finished = True
        except Exception:
            logger.warning(
                "Tau2 agent-service task failed. This trajectory will be rejected."
            )
            if not finished:
                try:
                    await self.agent_controller.set_reward(0.0, session["session_id"])
                except Exception:
                    logger.warning(
                        "Failed to finish session %s after workflow failure",
                        session["session_id"],
                    )
            raise

        interactions = await self.agent_controller.export_trajectory(
            session["session_id"],
            trajectory_id=trajectory_id,
            discount=self.discount,
            style=self.export_style,
        )
        if not interactions:
            logger.warning(
                "Session %s has no interactions, trajectory will be rejected.",
                session["session_id"],
            )
            return None

        last_id = next(reversed(interactions))
        last_reward = interactions[last_id].reward
        stats_tracker.get(workflow_context.stat_scope()).scalar(reward=last_reward)
        return interactions
