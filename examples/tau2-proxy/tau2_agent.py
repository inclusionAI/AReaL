"""Tau2 agent workflow for RL training with proxy server.

This module implements an AgentWorkflow for the tau2-bench benchmark,
compatible with AReaL's single-controller mode using proxy server.
Use this when running with `python3 tau2_train.py scheduler.type=slurm`.
"""

import os
import sys
import time
from typing import Any

# Suppress litellm verbose logging
os.environ["LITELLM_LOG"] = "ERROR"

# Add current directory to path for local imports when running as script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import litellm
from openai import AsyncOpenAI

# Disable litellm success/failure callbacks and verbose output
litellm.suppress_debug_info = True
litellm.set_verbose = False

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

from tau2_utils import Tau2EnvConfig, Tau2RunInfo

logger = logging.getLogger("Tau2ProxyAgent")


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


class Tau2ProxyAgentWorkflow(AgentWorkflow):
    """AgentWorkflow implementation for tau2-bench using Proxy Server.

    This workflow is compatible with single-controller mode and uses
    the OpenAI-compatible proxy server for inference.
    Use this when running with `python3 tau2_train.py scheduler.type=slurm`.

    In single-controller mode, AReaL:
    1. Launches SGLang/vLLM inference servers
    2. Starts a proxy server that translates OpenAI API calls
    3. Passes `base_url` and `http_client` to the workflow's `run()` method
    4. The workflow uses these to create AsyncOpenAI clients

    This mode supports archon backend for training because:
    - The training backend (archon/fsdp/megatron) is independent of the proxy
    - Archon handles the actor/critic/ref models
    - The proxy server handles inference requests from the workflow
    """

    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        econfig: dict[str, Any],
    ):
        self.gconfig = gconfig
        self.econfig = Tau2EnvConfig(**econfig)
        self.gen_args = gconfig.to_openai_args_dict(api_format="completions")

    async def run(
        self, data: dict[str, Any], **extra_kwargs: Any
    ) -> dict[str, float] | float:
        """Run tau2 agent via proxy server.

        Parameters
        ----------
        data : dict[str, Any]
            Input data containing task_id and split
        extra_kwargs : Any
            Contains base_url and http_client from AReaL proxy

        Returns
        -------
        dict[str, float] | float
            The final reward
        """
        base_url = extra_kwargs.get("base_url")
        http_client = extra_kwargs.get("http_client")

        if base_url is None:
            raise ValueError("base_url is required for AgentWorkflow")

        # Create AsyncOpenAI client using proxy server
        agent_client = AsyncOpenAI(
            base_url=base_url,
            api_key="EMPTY",
            http_client=http_client,
            timeout=120.0,
        )

        domain = self.econfig.domain
        split = data["split"]
        task_id = data["task_id"]
        task = _get_task(domain=domain, task_id=task_id, split=split)

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

        agent, user = self._get_agent_and_user(
            task=task, env=env, run_info=run_info, agent_client=agent_client
        )
        orchestrator = self._get_orchestrator(agent=agent, user=user, env=env, task=task)

        try:
            simulation = await orchestrator.arun()
            run_info.messages = simulation.messages
        except Exception as e:
            logger.error(
                f"ERROR RUNNING SIMULATION: Domain: {domain}, Task: {task.id}, "
                f"Error: {e}. Setting reward to 0.0"
            )
            run_info.error = str(e)
            return 0.0

        try:
            reward_info = evaluate_simulation(
                domain=domain,
                task=task,
                simulation=simulation,
                evaluation_type=EvaluationType.ALL,
                solo_mode=self.econfig.solo_mode,
            )
            run_info.reward = reward_info.reward
        except Exception as e:
            logger.error(f"ERROR EVALUATING SIMULATION: {e}. Setting reward to 0.0")
            return 0.0

        logger.info(f"FINISHED SIMULATION: Task: {task.id}, Reward: {run_info.reward}")
        return run_info.reward

    def _get_environment(self) -> Environment:
        environment_constructor = registry.get_env_constructor(self.econfig.domain)
        return environment_constructor(solo_mode=self.econfig.solo_mode)

    def _get_agent_and_user(
        self,
        task: Task,
        env: Environment,
        run_info: Tau2RunInfo,
        agent_client: AsyncOpenAI,
    ):
        agent_policy_doc = env.get_policy()
        tools: list[Tool] = env.get_tools()
        try:
            user_tools = env.get_user_tools()
        except Exception:
            user_tools = []
        if self.econfig.add_thinking_tool:
            tools.append(Tool(think))

        # Use proxy server for agent completions
        async def _acompletion(*args, **kwargs):
            start_time = time.perf_counter()
            kwargs.pop("api_key", None)
            kwargs.pop("base_url", None)
            kwargs.pop("n", None)  # n > 1 not supported
            # Set model to "default" for proxy server
            kwargs["model"] = "default"
            try:
                return await agent_client.chat.completions.create(**kwargs)
            finally:
                run_info.agent_time.append(time.perf_counter() - start_time)

        # Create a dedicated client for user LLM
        user_openai_client = AsyncOpenAI(
            base_url=self.econfig.user_llm_base_url,
            api_key="EMPTY",
            timeout=120.0,
            max_retries=3,
        )

        async def _acompletion_with_base_url(*args, **kwargs):
            start_time = time.perf_counter()
            model = kwargs.get("model", self.econfig.user_llm)
            if "/" in model:
                model = model.split("/", 1)[1]
            try:
                return await user_openai_client.chat.completions.create(
                    model=model,
                    messages=kwargs.get("messages", []),
                    tools=kwargs.get("tools"),
                    tool_choice=kwargs.get("tool_choice"),
                    temperature=kwargs.get("temperature", 0.0),
                    max_completion_tokens=kwargs.get("max_completion_tokens", 2048),
                )
            except Exception as e:
                logger.error(f"User LLM call failed: {e}")
                raise
            finally:
                run_info.user_time.append(time.perf_counter() - start_time)

        if self.econfig.solo_mode:
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
            domain=self.econfig.domain,
            agent=agent,
            user=user,
            environment=env,
            task=task,
            max_steps=self.econfig.max_steps,
            solo_mode=self.econfig.solo_mode,
        )
