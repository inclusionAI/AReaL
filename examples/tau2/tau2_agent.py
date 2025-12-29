import asyncio
import time

from litellm import acompletion, register_model
from tau2.agent.llm_agent import LLMAgent, LLMAgentState, LLMSoloAgent, LocalAgent
from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment
from tau2.environment.tool import Tool
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.registry import registry
from tau2.user.user_simulator import BaseUser, DummyUser, UserSimulator
from tau2_utils import Tau2EnvConfig, Tau2RunInfo

from areal.api.cli_args import GenerationHyperparameters
from areal.utils import logging
from areal.utils.proxy_utils import run_and_submit_rewards

logger = logging.getLogger("Tau2 Agent")

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
    """Use this tool to think. The thoughts will be visible in the history. Only use this tool to think when necessary."""
    return "Your thoughts are recorded. Please continue your work."


class Tau2Runner:
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

        # * Backup: use acreate to replace acompletion
        # async def _acreate(*args, **kwargs):
        #     kwargs.pop("num_retries", None)
        #         completion = await client.chat.completions.create(*args, **kwargs)
        #     return completion

        # async def _acreate_with_base_url(*args, **kwargs):
        #     kwargs.pop("num_retries", None)
        #     async with AsyncOpenAI(base_url=self.econfig.user_llm_base_url) as client:
        #         completion = await client.chat.completions.create(*args, **kwargs)
        #     return completion

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
            # max_errors=self.econfig.max_errors,
            # seed=self.econfig.seed,
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


async def run_agent_return_reward(data: dict) -> tuple[float, dict]:
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
    return await run_and_submit_rewards(func=run_agent_return_reward, data=data)


if __name__ == "__main__":
    import json
    import sys

    data = json.loads(sys.stdin.readline())
    asyncio.run(run_and_submit(data))
