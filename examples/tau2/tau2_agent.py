"""Tau2 agent workflow for RL training.

This module implements a RolloutWorkflow for the tau2-bench benchmark,
compatible with AReaL's SPMD mode using ArealOpenAI client.
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
from litellm import acompletion, register_model
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
from areal.api.engine_api import InferenceEngine
from areal.api.workflow_api import AgentWorkflow, RolloutWorkflow
from areal.core import workflow_context
from areal.experimental.openai import ArealOpenAI
from areal.utils import logging, stats_tracker

from tau2_utils import Tau2EnvConfig, Tau2RunInfo

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

    def __init__(
        self,
        econfig: Tau2EnvConfig,
        gconfig: GenerationHyperparameters,
        client: ArealOpenAI,
    ):
        self.econfig = econfig
        self.gconfig = gconfig
        self.client = client
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

        # Track completion call count for debugging
        completion_call_count = [0]

        # Use ArealOpenAI client for agent completions
        async def _acompletion(*args, **kwargs):
            completion_call_count[0] += 1
            call_num = completion_call_count[0]
            logger.debug(f"[Task {task.id}] Agent completion call #{call_num}")
            start_time = time.perf_counter()
            # Remove parameters that ArealOpenAI doesn't support
            kwargs.pop("api_key", None)
            kwargs.pop("base_url", None)
            kwargs.pop("n", None)  # n > 1 not supported, GRPO handles multiple samples at workflow level
            # Add chat template kwargs to disable thinking
            kwargs.setdefault("extra_body", {})
            kwargs["extra_body"]["chat_template_kwargs"] = {"enable_thinking": False}
            try:
                # Use ArealOpenAI client directly (not via HTTP)
                result = await self.client.chat.completions.create(**kwargs)
                logger.debug(f"[Task {task.id}] Agent completion #{call_num} succeeded")
                return result
            except Exception as e:
                logger.error(f"[Task {task.id}] Agent completion #{call_num} failed: {e}")
                raise
            finally:
                run_info.agent_time.append(time.perf_counter() - start_time)

        # Store completion_call_count in run_info for later logging
        run_info._completion_call_count = completion_call_count

        # Create a dedicated AsyncOpenAI client for user LLM
        user_openai_client = AsyncOpenAI(
            base_url=self.econfig.user_llm_base_url,
            api_key="EMPTY",
            timeout=120.0,  # Increased timeout for high load scenarios
            max_retries=3,  # Retry on transient failures
        )

        async def _acompletion_with_base_url(*args, **kwargs):
            start_time = time.perf_counter()
            # Extract model name without provider prefix (e.g., "openai/qwen" -> "qwen")
            model = kwargs.get("model", self.econfig.user_llm)
            if "/" in model:
                model = model.split("/", 1)[1]

            try:
                # Use openai client directly instead of litellm
                result = await user_openai_client.chat.completions.create(
                    model=model,
                    messages=kwargs.get("messages", []),
                    tools=kwargs.get("tools"),
                    tool_choice=kwargs.get("tool_choice"),
                    temperature=kwargs.get("temperature", 0.0),
                    max_completion_tokens=kwargs.get("max_completion_tokens", 2048),
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                return result
            except Exception as e:
                logger.error(f"User LLM call failed: {e}")
                raise
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
            solo_mode=self.solo_mode,
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
            # Log simulation details for debugging
            completion_calls = getattr(run_info, '_completion_call_count', [0])[0]
            logger.debug(
                f"[Task {task.id}] Simulation completed. "
                f"Messages: {len(simulation.messages)}, "
                f"Agent completions called: {completion_calls}, "
                f"Agent time entries: {len(run_info.agent_time)}"
            )
            # Log message roles for debugging when few messages
            if len(simulation.messages) <= 5:
                roles = [m.role for m in simulation.messages]
                logger.warning(
                    f"[Task {task.id}] Short simulation with only {len(simulation.messages)} messages. "
                    f"Roles: {roles}, Agent calls: {completion_calls}"
                )
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


class Tau2RolloutWorkflow(RolloutWorkflow):
    """RolloutWorkflow implementation for tau2-bench.

    This workflow runs the tau2 agent using ArealOpenAI client,
    which is compatible with SPMD mode (archon backend).
    """

    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer,
        econfig: dict[str, Any],
        turn_discount: float = 1.0,
    ):
        from areal.utils.hf_utils import load_hf_tokenizer

        if isinstance(tokenizer, str):
            tokenizer = load_hf_tokenizer(tokenizer)
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(tokenizer)
        self.tokenizer = tokenizer
        self.econfig = Tau2EnvConfig(**econfig)
        self.turn_discount = turn_discount

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Run a single episode of the tau2 workflow.

        Parameters
        ----------
        engine : InferenceEngine
            The inference engine to use for generating responses
        data : dict[str, Any]
            Input data containing task_id and split

        Returns
        -------
        dict[str, Any] | None
            The trajectory result with interactions
        """
        # Create ArealOpenAI client from engine
        client = ArealOpenAI(
            engine=engine,
            tokenizer=self.tokenizer,
            tool_call_parser="qwen",   # Use qwen format for tool calls
            reasoning_parser="qwen3",  # Use qwen3 format for reasoning
        )

        domain = self.econfig.domain
        split = data["split"]
        task_id = data["task_id"]
        task = _get_task(domain=domain, task_id=task_id, split=split)

        # Create runner with the client
        tau2_runner = Tau2Runner(self.econfig, self.gconfig, client)
        run_info = await tau2_runner.run(task)

        # Track reward in stats
        stats_tracker.get(workflow_context.stat_scope()).scalar(reward=run_info.reward)

        # Set reward on client and export interactions
        # Filter out incomplete interactions (where completion is None)
        cache = client._cache

        # Get completion call count from run_info if available
        completion_calls = getattr(run_info, '_completion_call_count', [0])[0]

        if not cache:
            logger.warning(
                f"[Task {task_id}] No interactions in cache. "
                f"Reward: {run_info.reward}, Error: {run_info.error}, "
                f"Messages count: {len(run_info.messages)}, "
                f"Agent completion calls: {completion_calls}, "
                f"Agent time calls: {len(run_info.agent_time)}"
            )
            return None

        # Remove incomplete interactions (where completion is None)
        incomplete_ids = [
            id for id, interaction in cache.items()
            if interaction.completion is None and interaction.response is None
        ]
        if incomplete_ids:
            logger.warning(
                f"[Task {task_id}] Removing {len(incomplete_ids)} incomplete interactions "
                f"out of {len(cache)} total"
            )
        for id in incomplete_ids:
            del cache[id]

        if not cache:
            logger.warning(
                f"[Task {task_id}] All interactions were incomplete. "
                f"Reward: {run_info.reward}, Error: {run_info.error}"
            )
            return None

        client.set_last_reward(run_info.reward)
        client.apply_reward_discount(turn_discount=self.turn_discount)
        interactions = client.export_interactions(style="individual")

        logger.info(
            f"[Task {task_id}] Completed with {len(interactions)} interactions, "
            f"reward: {run_info.reward}"
        )

        # Save trajectory if enabled
        if self.econfig.save_trajectories:
            self._save_trajectory(engine, task_id, run_info)

        return interactions

    def _save_trajectory(
        self,
        engine: InferenceEngine,
        task_id: str,
        run_info: Tau2RunInfo,
    ):
        """Save trajectory to JSONL file.

        File is saved to: {trajectory_save_dir}/{step}/{task_id}.jsonl
        """
        import json
        import os

        try:
            # Get current step from engine version
            step = engine.get_version()

            # Build save path: {step}/{task_id}.jsonl
            base_dir = self.econfig.trajectory_save_dir or os.getcwd()
            save_dir = os.path.join(base_dir, str(step))
            os.makedirs(save_dir, exist_ok=True)

            # Sanitize task_id for filename (replace special chars)
            safe_task_id = task_id.replace("/", "_").replace("\\", "_").replace(":", "_")
            filepath = os.path.join(save_dir, f"{safe_task_id}.jsonl")

            # Build record
            record = run_info.to_dict()
            record["step"] = step

            # Append to file (multiple samples for same task go to same file)
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        except Exception as e:
            logger.warning(f"Failed to save trajectory for task {task_id}: {e}")


# Alias for backward compatibility
Tau2AgentWorkflow = Tau2RolloutWorkflow


class Tau2ProxyAgentWorkflow(AgentWorkflow):
    """AgentWorkflow implementation for tau2-bench using Proxy Server.

    This workflow is compatible with single-controller mode and uses
    the OpenAI-compatible proxy server for inference.
    Use this when running with `python3 tau2_train.py scheduler.type=slurm`.
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
            Contains base_url and http_client from AReaL

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
            kwargs.pop("n", None)
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
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
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
