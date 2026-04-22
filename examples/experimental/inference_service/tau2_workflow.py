"""Tau2 workflows for inference-service examples.

`Tau2InferenceWorkflow` runs the Tau2 simulation locally and manages
`/rl/start_session` + `/rl/set_reward` directly against the inference-service gateway.
`Tau2AgentServiceWorkflow` keeps the agent-service-based path for the companion
example that runs the agent loop out of process.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx

from areal.utils import logging

logger = logging.getLogger("Tau2Workflow")


class Tau2InferenceWorkflow:
    """Run Tau2 locally while reporting session lifecycle to the gateway."""

    def __init__(
        self,
        gateway_addr: str,
        gateway_api_key: str,
        model: str | None = None,
        econfig: dict | None = None,
        gen_args: dict | None = None,
        timeout: float = 600.0,
    ) -> None:
        from examples.tau2.utils import Tau2EnvConfig

        if econfig is None:
            self.econfig = Tau2EnvConfig()
        elif isinstance(econfig, dict):
            self.econfig = Tau2EnvConfig(**econfig)
        else:
            self.econfig = econfig
        self.gen_args = gen_args or {}
        self.timeout = timeout
        self.gateway_addr = gateway_addr.rstrip("/")
        self.gateway_api_key = gateway_api_key
        self.model = model

    async def _request(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        api_key: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        response = await client.post(
            f"{self.gateway_addr}{endpoint}",
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        response.raise_for_status()
        return response.json()

    async def _start_session(
        self,
        client: httpx.AsyncClient,
        task_id: str,
    ) -> dict[str, str]:
        response = await self._request(
            client,
            "/rl/start_session",
            self.gateway_api_key,
            {"task_id": task_id},
        )
        return {
            "session_id": response["session_id"],
            "api_key": response["api_key"],
        }

    async def _set_reward(
        self,
        client: httpx.AsyncClient,
        session_api_key: str,
        reward: float,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"interaction_id": None, "reward": reward}
        if self.model:
            payload["model"] = self.model
        return await self._request(
            client,
            "/rl/set_reward",
            session_api_key,
            payload,
        )

    async def run(self, data: dict[str, Any], **extra_kwargs: Any) -> dict[str, Any]:
        from openai import AsyncOpenAI

        from examples.tau2.agent import Tau2Runner, _get_task
        from examples.tau2.utils import Tau2EnvConfig

        base_url: str | None = extra_kwargs.get("base_url") or os.getenv(
            "OPENAI_BASE_URL"
        )
        task_id_str = extra_kwargs.get("task_id", str(data.get("task_id", "")))
        http_client: httpx.AsyncClient | None = extra_kwargs.get("http_client")

        if base_url is None:
            raise ValueError("base_url is required for Tau2InferenceWorkflow")

        client = http_client or httpx.AsyncClient(timeout=30.0)
        owns_client = http_client is None
        try:
            session = await self._start_session(client, task_id_str)
            session_api_key = session["api_key"]

            econfig = self.econfig
            if "econfig" in data:
                econfig = Tau2EnvConfig(**data["econfig"])

            gen_args = self.gen_args.copy()
            if "gconfig" in data:
                gen_args.update(data["gconfig"])

            domain = econfig.domain
            split = data.get("split", "train")
            task = _get_task(domain=domain, task_id=data["task_id"], split=split)

            agent_client = AsyncOpenAI(
                base_url=base_url,
                api_key=session_api_key,
                http_client=client,
                max_retries=0,
            )

            user_client = None
            if not econfig.solo_mode and econfig.user_llm_base_url:
                user_client = AsyncOpenAI(
                    base_url=econfig.user_llm_base_url,
                    api_key="dummy",
                    max_retries=3,
                    timeout=120.0,
                )

            runner = Tau2Runner(
                econfig=econfig,
                gen_args=gen_args,
                agent_client=agent_client,
                user_client=user_client,
            )

            finished = False
            try:
                run_info = await asyncio.wait_for(
                    runner.run(task), timeout=self.timeout
                )
                reward = run_info.reward
                result = await self._set_reward(client, session_api_key, reward)
                finished = True
            except Exception:
                if not finished:
                    try:
                        await self._set_reward(client, session_api_key, 0.0)
                    except Exception:
                        logger.warning(
                            "Failed to set 0 reward for session %s",
                            session["session_id"],
                        )
                raise
        finally:
            if owns_client:
                await client.aclose()

        trajectory_id = result.get("trajectory_id")
        return {
            "session_id": session["session_id"],
            "trajectory_id": (
                int(trajectory_id) if trajectory_id is not None else None
            ),
            "reward": reward,
        }


def _extract_response_text(response: dict[str, Any]) -> str:
    """Extract assistant text from an OpenResponses /v1/responses result."""
    parts: list[str] = []
    for item in response.get("output", []):
        if item.get("type") == "message":
            for block in item.get("content", []):
                if block.get("type") == "output_text":
                    parts.append(block.get("text", ""))
    return "\n".join(parts)


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


class Tau2AgentServiceWorkflow:
    """Tau2 workflow using agent service + inference service.

    Orchestrates the agent loop through an ``AgentServiceController``
    using ``new_session``, ``step``, and ``set_reward``.

    The agent-service agent handles tool calls internally. This workflow
    seeds the first user turn from rollout data and generates later user
    turns through the configured user LLM.
    """

    def __init__(
        self,
        agent_controller: Any | None = None,
        econfig: dict | None = None,
        gen_args: dict | None = None,
        timeout: float = 600.0,
        max_turns: int = 10,
    ) -> None:
        from examples.tau2.utils import Tau2EnvConfig

        if econfig is None:
            self.econfig = Tau2EnvConfig()
        elif isinstance(econfig, dict):
            self.econfig = Tau2EnvConfig(**econfig)
        else:
            self.econfig = econfig
        self.gen_args = gen_args or {}
        self.timeout = timeout
        self.max_turns = max_turns
        self.agent_controller = agent_controller

    async def run(self, data: dict[str, Any], **extra_kwargs: Any) -> dict[str, Any]:
        from openai import AsyncOpenAI
        from tau2.data_model.message import AssistantMessage, UserMessage
        from tau2.data_model.simulation import SimulationRun, TerminationReason
        from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation

        from examples.tau2.agent import _get_task
        from examples.tau2.utils import Tau2EnvConfig

        ctrl = self.agent_controller
        if ctrl is None:
            raise ValueError(
                "agent_controller is required for Tau2AgentServiceWorkflow"
            )

        task_id_str = extra_kwargs.get("task_id", str(data.get("task_id", "")))

        econfig = self.econfig
        if "econfig" in data:
            econfig = Tau2EnvConfig(**data["econfig"])

        domain = econfig.domain
        split = data.get("split", "train")
        task = _get_task(domain=domain, task_id=data["task_id"], split=split)

        session = ctrl.new_session(task_id=task_id_str)
        first_user_message = str(data.get("prompt") or task.user_scenario).strip()
        if not first_user_message:
            raise ValueError("data.prompt or task.user_scenario is required")

        if not econfig.solo_mode and not econfig.user_llm_base_url:
            raise ValueError(
                "econfig.user_llm_base_url is required for Tau2AgentServiceWorkflow"
            )

        user_client = None
        if not econfig.solo_mode:
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
        finished = False
        try:
            for i in range(self.max_turns):
                response = ctrl.step(next_user_message, session["session_id"])
                agent_text = _extract_response_text(response)

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
                        content=agent_text or "(no response)",
                        turn_idx=len(tau2_messages),
                    )
                )

                if i + 1 >= self.max_turns:
                    break

                if user_client is None:
                    break

                chat_history.append(
                    {"role": "assistant", "content": agent_text or "(no response)"}
                )
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

            reward = 0.0
            if tau2_messages:
                try:
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
                        solo_mode=False,
                        domain=domain,
                    )
                    reward = reward_info.reward
                except Exception as e:
                    logger.error("Evaluation failed for task %s: %s", task.id, e)

            result = ctrl.set_reward(reward, session["session_id"])
            finished = True
        except Exception:
            if not finished:
                try:
                    ctrl.set_reward(0.0, session["session_id"])
                except Exception:
                    logger.warning(
                        "Failed to set 0 reward for session %s",
                        session["session_id"],
                    )
            raise

        trajectory_id = result.get("trajectory_id")
        return {
            "session_id": session["inference_session_id"],
            "trajectory_id": (
                int(trajectory_id) if trajectory_id is not None else None
            ),
            "reward": reward,
        }
