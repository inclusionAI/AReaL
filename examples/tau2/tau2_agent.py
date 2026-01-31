"""Tau2 agent workflow for RL training.

This module implements an AgentWorkflow for the tau2-bench benchmark,
compatible with AReaL's OpenAI proxy server mode.
"""

import asyncio
import time
from typing import Any

from litellm import acompletion, register_model
from tau2.agent.llm_agent import LLMAgent, LLMAgentState, LLMSoloAgent, LocalAgent
from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment
from tau2.environment.tool import Tool
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.registry import registry
from tau2.user.user_simulator import BaseUser, DummyUser, UserSimulator

from areal.api.cli_args import GenerationHyperparameters
from areal.api.workflow_api import AgentWorkflow
from areal.utils import logging

from .tau2_utils import Tau2EnvConfig, Tau2RunInfo

logger = logging.getLogger("Tau2Agent")

# Register a dummy model for litellm
register_model(
    {
        "dummy": {
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "litellm_provider": "openai",
            "mode": "chat",
        },
    }
)


def _get_task(domain: str, task_id: str, split: str | None = None) -> Task:
    tasks: list[Task] = registry.get_tasks_loader(domain)(split)
    for task in tasks:
        if task.id == task_id:
            return task
    raise ValueError(f"No task found with id {task_id} for domain {domain}")


def think(thoughts: str):
    """Use this tool to think. The thoughts will be visible in the history.

    Only use this tool to think when necessary.
    """
    return "Your thoughts are recorded. Please continue your work."


class Tau2Runner:
    """Runner for tau2-bench tasks."""

    def __init__(self, econfig: Tau2EnvConfig, gconfig: GenerationHyperparameters):
        self.econfig = econfig
        self.gconfig = gconfig
        self.domain = econfig.domain
        self.solo_mode = econfig.solo_mode
        self.gen_args = gconfig.to_openai_args_dict(api_format="completions")

    def _get_environment(self) -> Environment:
        environment_constructor = registry.get_env_constructor(self.domain)
        return environment_constructor(solo_mode=self.solo_mode)

    def _get_agent_and_user(self, task: Task, env: Environment, run_info: Tau2RunInfo):
        agent_policy_doc = env.get_policy()
        tools: list[Tool] = env.get_tools()
        try:
            user_tools = env.get_user_tools()
        except Exception:
            user_tools = []
        if self.econfig.add_thinking_tool:
            tools.append(Tool(think))

        async def _acompletion(*args, **kwargs):
            start_time = time.perf_counter()
            kwargs.update(
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                thinking=False,
            )
            try:
                return await acompletion(*args, **kwargs)
            finally:
                run_info.agent_time.append(time.perf_counter() - start_time)

        async def _acompletion_with_base_url(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                return await acompletion(
                    *args, base_url=self.econfig.user_llm_base_url, **kwargs
                )
            finally:
                run_info.user_time.append(time.perf_counter() - start_time)

        if self.solo_mode:
            agent = LLMSoloAgent(
                tools=tools + user_tools,
                domain_policy=agent_policy_doc,
                llm="dummy",
                llm_args=self.gen_args,
                task=task,
                completion_fn=_acompletion,
            )
            user = DummyUser()
        else:
            agent = LLMAgent(
                tools=tools,
                domain_policy=agent_policy_doc,
                llm="dummy",
                llm_args=self.gen_args,
                completion_fn=_acompletion,
            )

            user = UserSimulator(
                tools=user_tools if len(user_tools) > 0 else None,
                instructions=str(task.user_scenario),
                llm=self.econfig.user_llm,
                llm_args=self.econfig.user_llm_args,
                completion_fn=_acompletion_with_base_url,
            )
        return agent, user

    def _get_orchestrator(
        self,
        agent: LocalAgent[LLMAgentState],
        user: BaseUser,
        env: Environment,
        task: Task,
    ) -> Orchestrator:
        return Orchestrator(
            domain=self.domain,
            agent=agent,
            user=user,
            environment=env,
            task=task,
            max_steps=self.econfig.max_steps,
        )

    async def run(self, task: Task) -> Tau2RunInfo:
        domain = self.domain
        solo_mode = self.solo_mode
        logger.info(
            f"STARTING SIMULATION: Domain: {domain}, Task: {task.id}, "
            f"Solo Mode: {solo_mode}"
        )

        env = self._get_environment()
        run_info = Tau2RunInfo(
            reward=0.0,
            task=task,
            messages=[],
            agent_time=[],
            user_time=[],
            reward_info=None,
            error=None,
        )
        agent, user = self._get_agent_and_user(task=task, env=env, run_info=run_info)
        orchestrator = self._get_orchestrator(
            agent=agent, user=user, env=env, task=task
        )

        try:
            simulation = await orchestrator.arun()
            run_info.messages = simulation.messages
        except Exception as e:
            logger.error(
                f"ERROR RUNNING SIMULATION: Domain: {domain}, Task: {task.id}, "
                f"Agent: {agent.__class__.__name__}, User: {user.__class__.__name__}. "
                f"Error running simulation: {e}. Setting reward to 0.0"
            )
            run_info.messages = orchestrator.get_trajectory()
            run_info.error = str(e)
            return run_info

        try:
            reward_info = evaluate_simulation(
                domain=domain,
                task=task,
                simulation=simulation,
                evaluation_type=EvaluationType.ALL,
                solo_mode=solo_mode,
            )
            run_info.reward_info = reward_info
            run_info.reward = reward_info.reward
        except Exception as e:
            logger.error(
                f"ERROR EVALUATING SIMULATION: Domain: {domain}, Task: {task.id}, "
                f"Agent: {agent.__class__.__name__}, User: {user.__class__.__name__}. "
                f"Error evaluating simulation: {e}. Setting reward to 0.0"
            )
            run_info.reward_info = None
            run_info.error = str(e)
            return run_info

        logger.info(
            f"FINISHED SIMULATION: Domain: {domain}, Task: {task.id}, "
            f"Agent: {agent.__class__.__name__}, User: {user.__class__.__name__}. "
            f"Reward: {reward_info.reward}"
        )
        return run_info


class Tau2AgentWorkflow(AgentWorkflow):
    """AgentWorkflow implementation for tau2-bench.

    This workflow runs the tau2 agent using the OpenAI-compatible API
    provided by AReaL's proxy server.
    """

    async def run(
        self, data: dict[str, Any], **extra_kwargs: Any
    ) -> dict[str, float] | float:
        """Run the tau2 agent and return the reward.

        Parameters
        ----------
        data : dict[str, Any]
            Input data containing:
            - econfig: Tau2EnvConfig parameters
            - gconfig: GenerationHyperparameters parameters
            - task_id: Task ID to run
            - split: Dataset split
        **extra_kwargs : Any
            Extra arguments from AReaL including base_url and http_client

        Returns
        -------
        float
            The final reward from the tau2 evaluation
        """
        econfig = Tau2EnvConfig(**data.get("econfig", {}))
        gconfig = GenerationHyperparameters(**data.get("gconfig", {}))

        domain = econfig.domain
        split = data["split"]
        task_id = data["task_id"]
        task = _get_task(domain=domain, task_id=task_id, split=split)

        tau2_runner = Tau2Runner(econfig, gconfig)
        run_info = await tau2_runner.run(task)

        # Store run_info in data for potential logging by the caller
        data["_run_info"] = run_info

        return run_info.reward


# For backward compatibility with subproc mode
async def run_agent_return_reward(data: dict) -> tuple[float, Tau2RunInfo]:
    """Run the agent and return (reward, run_info) tuple."""
    econfig = Tau2EnvConfig(**data.get("econfig", {}))
    gconfig = GenerationHyperparameters(**data.get("gconfig", {}))

    domain = econfig.domain
    split = data["split"]
    task_id = data["task_id"]
    task = _get_task(domain=domain, task_id=task_id, split=split)

    tau2_runner = Tau2Runner(econfig, gconfig)
    run_info = await tau2_runner.run(task)
    return run_info.reward, run_info


async def run_and_submit(data: dict):
    """Run agent and submit rewards to proxy server.

    This function is used when running in subprocess mode where
    the agent process communicates with the proxy server via HTTP.
    """
    import os

    import aiohttp

    from areal.experimental.openai.proxy.client_session import (
        set_last_interaction_reward,
    )
    from areal.experimental.openai.proxy.server import RL_SET_REWARD_PATHNAME

    base_url = os.environ.get("OPENAI_BASE_URL", "")
    if not base_url.endswith("/"):
        base_url += "/"

    async with aiohttp.ClientSession() as session:
        reward, run_info = await run_agent_return_reward(data)
        await set_last_interaction_reward(
            session, reward=reward, url=f"{base_url}{RL_SET_REWARD_PATHNAME}"
        )
        return run_info


if __name__ == "__main__":
    import json
    import sys

    data = json.loads(sys.stdin.readline())
    asyncio.run(run_and_submit(data))
