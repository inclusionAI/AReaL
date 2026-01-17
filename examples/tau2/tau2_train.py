import sys
import time
from dataclasses import dataclass, field

import numpy as np
from datasets import Dataset
from loguru import logger as loguru_logger
from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment
from tau2.environment.tool import Tool
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.registry import registry
from tau2.user.user_simulator import DummyUser, UserSimulator
from tau2_utils import Tau2EnvConfig, Tau2RunInfo
import os
from litellm import acompletion, register_model
from tau2.agent.llm_agent import LLMAgent, LLMAgentState, LLMSoloAgent, LocalAgent
from tau2.user.user_simulator import BaseUser
from tau2.orchestrator.orchestrator import Orchestrator

from areal.api.cli_args import (
    GenerationHyperparameters,
    PPOConfig,
    load_expr_config,
)
from areal.api.engine_api import InferenceEngine
from areal.api.workflow_api import RolloutWorkflow
from areal.experimental.openai import ArealOpenAI
from areal.experimental.trainer.rl import PPOTrainer
from areal.utils import logging, stats_tracker
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.stats_logger import StatsLogger
from areal.core import workflow_context

logger = logging.getLogger("Tau2 Example")

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


# ================================ dataset ================================
def get_tau2_dataset(
    domain: str,
    type: str = "rl",
    split: str = "train",
) -> Dataset:
    """Create a HuggingFace Dataset from tau2 task IDs.

    Args:
        domain: The tau2 domain name, e.g., 'retail', 'airline', 'telecom'
        split: Dataset split (e.g., 'train', 'test')
        type: Dataset type (e.g., 'rl', 'sft'), only 'rl' is supported for now

    Returns:
        Dataset: HuggingFace Dataset containing task_id entries
    """
    assert type == "rl", "Only RL dataset is supported for now"
    # TODO: support SFT dataset

    splits_loader_fn = registry.get_task_splits_loader(domain)
    if splits_loader_fn is None:
        raise ValueError(f"No task splits loader found for domain {domain}")
    splits = splits_loader_fn()
    if split not in splits:
        raise ValueError(
            f"Split {split} not found in {splits}, available splits: {splits.keys()}"
        )
    task_ids = splits[split]
    # print(f"domain: {domain}, split: {split}, task_ids: {task_ids}")

    dataset_items = [{"task_id": task_id, "split": split} for task_id in task_ids]
    dataset = Dataset.from_list(dataset_items)
    return dataset


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
    """Runner for Tau2 environment using ArealOpenAI client."""

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

        async def _acompletion_via_client(*args, **kwargs):
            """Completion function that uses ArealOpenAI client."""
            start_time = time.perf_counter()
            kwargs.update(
                extra_body={"chat_template_kwargs": {"enable_thinking": True}},
                thinking=True,
            )
            # Remove litellm-specific arguments
            kwargs.pop("num_retries", None)
            try:
                # Use ArealOpenAI client for inference
                completion = await self.client.chat.completions.create(*args, **kwargs)
                return completion
            except ValueError as e:
                logger.warning(f"ValueError in _acompletion_via_client: {e}")
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
                completion_fn=_acompletion_via_client,
            )
            user = DummyUser()
        else:
            agent = LLMAgent(
                tools=tools,
                domain_policy=agent_policy_doc,
                llm="dummy",
                llm_args=self.gen_args,
                completion_fn=_acompletion_via_client,
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
            import traceback

            traceback.print_exc()
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


class Tau2Workflow(RolloutWorkflow):
    """Tau2 workflow using ArealOpenAI client for multi-turn agent interactions."""

    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        econfig: Tau2EnvConfig,
        tokenizer_path: str,
        tool_call_parser: str = "qwen25",
        export_style: str = "individual",
        max_total_tokens: int = 32768,
        dump_dir: str | None = None,
    ):
        from areal.utils.hf_utils import load_hf_tokenizer

        self.gconfig = gconfig.new(n_samples=1)
        self.econfig = econfig
        self.tokenizer = load_hf_tokenizer(tokenizer_path)
        self.tool_call_parser = tool_call_parser
        self.export_style = export_style
        self.max_total_tokens = max_total_tokens
        self.dump_dir = dump_dir
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

        # Set chat_template_type based on export_style
        self.chat_template_type = "hf" if export_style == "individual" else "concat"

    async def arun_episode(self, engine: InferenceEngine, data):
        """Run a single episode using ArealOpenAI client."""
        # Create ArealOpenAI client for this episode
        client = ArealOpenAI(
            engine=engine,
            tokenizer=self.tokenizer,
            tool_call_parser=self.tool_call_parser,
            engine_max_tokens=self.max_total_tokens,
            chat_template_type=self.chat_template_type,
        )

        # Get task information
        domain = self.econfig.domain
        split = data["split"]
        task_id = data["task_id"]
        task = _get_task(domain=domain, task_id=task_id, split=split)

        # Create runner and execute
        runner = Tau2Runner(
            econfig=self.econfig,
            gconfig=self.gconfig,
            client=client,
        )
        run_info = await runner.run(task)

        # Set reward on the last interaction
        if client._cache:
            client.set_last_reward(run_info.reward)

        # Log stats
        stats_tracker.get(workflow_context.stat_scope()).scalar(reward=run_info.reward)
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            steps_count=len(run_info.messages),
            orchestrator_error=int(run_info.error is not None),
        )

        def add_to_stats(name: str, times: list[float]):
            if len(times):
                for key in ["mean", "max", "min", "std", "sum"]:
                    stats_tracker.get(workflow_context.stat_scope()).scalar(
                        **{f"{name}_time/{key}": getattr(np.array(times), key)()}
                    )

        add_to_stats("agent", run_info.agent_time)
        add_to_stats("user", run_info.user_time)

        # Dump info to file if configured
        if self.dump_dir is not None:
            import aiofiles

            version = engine.get_version()
            version_dir = os.path.join(self.dump_dir, str(version))
            os.makedirs(version_dir, exist_ok=True)

            real_task_id = task_id[:120]
            try:
                json_path = os.path.join(version_dir, f"{real_task_id}.json")
                async with aiofiles.open(json_path, "w") as f:
                    await f.write(run_info.model_dump_json())

                file_path = os.path.join(version_dir, f"{real_task_id}.txt")
                async with aiofiles.open(file_path, "a") as f:
                    await f.write(str(run_info))
            except Exception as e:
                logger.error(f"Error dumping rollout to file: {e}")

        # Apply reward discount and export completions
        client.apply_reward_discount(turn_discount=self.econfig.turn_discount)
        completions_with_reward = client.export_interactions(style=self.export_style)
        return completions_with_reward


@dataclass
class Tau2PPOConfig(PPOConfig):
    econfig: Tau2EnvConfig = field(default_factory=Tau2EnvConfig)
    tool_call_parser: str = field(
        default="qwen25",
        metadata={"help": "Tool call parser that used by ArealOpenAI client."},
    )
    export_style: str = field(
        default="individual",
        metadata={
            "help": "Style for exporting completion results from the ArealOpenAI client."
        },
    )
    do_eval: bool = field(
        default=False,
        metadata={"help": "Whether to do evaluation."},
    )


def main(args):
    import warnings

    # TODO: figure out why pydantic UserWarning happens
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

    config, _ = load_expr_config(args, Tau2PPOConfig)
    domain = config.econfig.domain

    # remove the logging of loguru logger in tau2-bench package.
    loguru_logger.remove()
    loguru_logger.add(
        os.path.join(StatsLogger.get_log_path(config.stats_logger), "tau2.log"),
        level="INFO",
    )

    # Create dataset and dataloaders
    train_dataset = get_tau2_dataset(
        domain=domain,
        type=config.train_dataset.type,
        split=config.train_dataset.path.split("/")[-1],
    )
    valid_dataset = get_tau2_dataset(
        domain=domain,
        type=config.valid_dataset.type,
        split=config.valid_dataset.path.split("/")[-1],
    )

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        # Build workflow kwargs
        workflow_kwargs = dict(
            gconfig=config.gconfig,
            econfig=config.econfig,
            tokenizer_path=config.tokenizer_path,
            tool_call_parser=config.tool_call_parser,
            export_style=config.export_style,
            max_total_tokens=config.gconfig.max_tokens,
            dump_dir=os.path.join(
                StatsLogger.get_log_path(config.stats_logger),
                "generated",
            ),
        )

        # Eval workflow with different temperature
        eval_workflow_kwargs = workflow_kwargs.copy()
        eval_workflow_kwargs["gconfig"] = config.gconfig.new(temperature=0.6)
        eval_workflow_kwargs["dump_dir"] = os.path.join(
            StatsLogger.get_log_path(config.stats_logger),
            "generated-eval",
        )

        if not config.do_eval:
            eval_workflow_kwargs = None

        trainer.train(
            workflow="examples.tau2.tau2_train.Tau2Workflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="examples.tau2.tau2_train.Tau2Workflow",
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
