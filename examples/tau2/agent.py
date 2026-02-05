"""Tau2 Agent Workflow for AReaL proxy mode.

This module implements a Tau2 agent that uses the AReaL proxy server
for OpenAI-compatible API calls during RL training.
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

# Import utilities (also patches tau2.utils.llm_utils)
from examples.tau2.utils import Tau2EnvConfig, Tau2RunInfo

from areal.utils import logging

logger = logging.getLogger("Tau2 Agent")

# Register dummy model for litellm
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
    """Get a task by ID from the tau2 registry."""
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
    """Runner for Tau2 environment using litellm acompletion via proxy."""

    def __init__(
        self,
        econfig: Tau2EnvConfig,
        gen_args: dict,
        base_url: str,
    ):
        self.econfig = econfig
        self.gen_args = gen_args
        self.base_url = base_url
        self.domain = econfig.domain
        self.solo_mode = econfig.solo_mode

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

        async def _acompletion_via_proxy(*args, **kwargs):
            """Completion function that uses litellm acompletion via proxy."""
            start_time = time.perf_counter()
            kwargs.update(
                extra_body={"chat_template_kwargs": {"enable_thinking": True}},
            )
            try:
                return await acompletion(*args, base_url=self.base_url, **kwargs)
            except ValueError as e:
                logger.warning(f"ValueError in _acompletion_via_proxy: {e}")
                raise
            finally:
                run_info.agent_time.append(time.perf_counter() - start_time)

        async def _acompletion_with_base_url(*args, **kwargs):
            """Completion function for user simulator using external LLM."""
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
                completion_fn=_acompletion_via_proxy,
            )
            user = DummyUser()
        else:
            agent = LLMAgent(
                tools=tools,
                domain_policy=agent_policy_doc,
                llm="dummy",
                llm_args=self.gen_args,
                completion_fn=_acompletion_via_proxy,
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
        """Run a simulation for the given task."""
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


class Tau2AgentWorkflow:
    """Tau2 agent workflow for AReaL proxy mode.

    This workflow runs a Tau2 customer service simulation using the proxy server
    for OpenAI-compatible API calls. It supports both standard multi-turn mode
    and solo mode where the agent handles both agent and user roles.

    Args:
        econfig: Tau2 environment configuration
        gen_args: Generation arguments (temperature, max_tokens, etc.)
        timeout: Maximum time allowed for a single episode (default: 7200s)
    """

    def __init__(
        self,
        econfig: Tau2EnvConfig | dict | None = None,
        gen_args: dict | None = None,
        timeout: float = 7200.0,
    ):
        if econfig is None:
            econfig = Tau2EnvConfig()
        elif isinstance(econfig, dict):
            econfig = Tau2EnvConfig(**econfig)
        self.econfig = econfig
        self.gen_args = gen_args or {}
        self.timeout = timeout

    async def run(
        self, data: dict[str, Any], **extra_kwargs: Any
    ) -> dict[str, float] | float:
        """Run a Tau2 simulation episode.

        Args:
            data: Input data containing ask_id, split, and optional econfig/gconfig
            **extra_kwargs: Additional kwargs including:
                - base_url: Proxy server URL

        Returns:
            float: The reward from the simulation
        """
        # Get proxy URL from workflow context
        base_url: str | None = extra_kwargs.get("base_url", None)

        if base_url is None:
            raise ValueError("base_url is required for Tau2AgentWorkflow")

        # Override econfig from data if provided
        econfig = self.econfig
        if "econfig" in data:
            econfig = Tau2EnvConfig(**data["econfig"])

        # Override gen_args from data if provided
        gen_args = self.gen_args.copy()
        if "gconfig" in data:
            gen_args.update(data["gconfig"])

        # Get task information
        domain = econfig.domain
        split = data.get("split", "train")
        task_id = data["task_id"]
        task = _get_task(domain=domain, task_id=task_id, split=split)

        # Create runner and execute
        runner = Tau2Runner(
            econfig=econfig,
            gen_args=gen_args,
            base_url=base_url,
        )

        try:
            run_info = await asyncio.wait_for(runner.run(task), timeout=self.timeout)
        except TimeoutError:
            logger.error(
                f"TIMEOUT: Task {task_id} exceeded {self.timeout}s limit. "
                f"Setting reward to 0.0"
            )
            return 0.0

        # Return the reward
        # The proxy server handles tracking completions and assigning rewards
        return run_info.reward
