import asyncio
import os
import sys
from dataclasses import dataclass, field

from agents import Agent as OpenAIAgent
from agents import ModelSettings, OpenAIProvider, RunConfig
from agents import Runner as OpenAIRunner
from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters, GRPOConfig, load_expr_config
from areal.api.reward_api import AsyncRewardWrapper
from areal.api.workflow_api import RolloutWorkflow
from areal.dataset import get_custom_dataset
from areal.experimental.openai import ArealOpenAI
from areal.experimental.trainer.rl import GRPOTrainer
from areal.utils import stats_tracker
from areal.utils.dynamic_import import import_from_string
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.stats_logger import StatsLogger
from areal.workflow.rlvr import RLVRWorkflow


class OpenAIAgentWrapper:
    def __init__(
        self,
        agent_builder_path: str,
        agent_builder_kwargs: dict,
        reward_fn_path: str,
        temperature: float = 1.0,
        max_completion_tokens: int = 512,
    ):
        self.agent_builder = import_from_string(agent_builder_path)
        self.agent_builder_kwargs = agent_builder_kwargs
        self.async_reward_fn = AsyncRewardWrapper(import_from_string(reward_fn_path))
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens

    async def run_agent(self, data, client: ArealOpenAI):
        agent: OpenAIAgent = self.agent_builder(**self.agent_builder_kwargs)
        run_config = RunConfig(
            model_provider=OpenAIProvider(openai_client=client),
            tracing_disabled=True,
            model_settings=ModelSettings(
                temperature=self.temperature,
                extra_args={"max_completion_tokens": self.max_completion_tokens},
            ),
        )
        result = await OpenAIRunner.run(
            agent, input=data["messages"][-1]["content"], run_config=run_config
        )
        reward = await self.async_reward_fn(
            result=result.final_output, answer=data["answer"]
        )
        client.set_final_reward(reward)
        return reward


class OpenAIAgentWorkflow(RolloutWorkflow):
    def __init__(
        self,
        agent_builder_path: str,
        agent_builder_kwargs: dict,
        reward_fn_path: str,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        dump_dir: str | None = None,
        rollout_stat_scope: str = "rollout",
        n_trajs: int = 1,
    ):
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(tokenizer)
        self.gconfig.n_samples = 1
        self.tokenizer = tokenizer
        self.dump_dir = dump_dir
        self.rollout_stat_scope = rollout_stat_scope
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

        # Search hyper-parameters
        self.n_trajs = n_trajs
        self.agent = OpenAIAgentWrapper(
            agent_builder_kwargs=agent_builder_kwargs,
            temperature=gconfig.temperature,
            max_completion_tokens=gconfig.max_new_tokens,
            agent_builder_path=agent_builder_path,
            reward_fn_path=reward_fn_path,
        )

    async def arun_episode(self, engine, data):
        clients = [
            ArealOpenAI(engine=engine, tokenizer=self.tokenizer)
            for _ in range(self.n_trajs)
        ]

        # Collect trajectories
        rewards = await asyncio.gather(
            *[
                self.agent.run_agent(
                    data=data,
                    client=clients[i],
                )
                for i in range(self.n_trajs)
            ]
        )
        for reward in rewards:
            stats_tracker.get(self.rollout_stat_scope).scalar(reward=reward)

        interactions_with_reward = {}
        for client in clients:
            client.apply_reward_discount(turn_discount=0.9)
            interactions = client.export_interactions(style="individual")
            interactions_with_reward.update(interactions)
        return interactions_with_reward


@dataclass
class AgentRLConfig(GRPOConfig):
    reward_fn_path: str = "areal.reward.gsm8k.gsm8k_reward_fn"
    agent_builder_path: str = "areal.workflow.openai_agent.math_agent.build_math_agent"
    agent_builder_kwargs: dict = field(default_factory=dict)
    n_trajs: int = field(
        default=1,
        metadata={
            "help": "We could collect multiple trajectories for a single query. By default n_trajs=1."
        },
    )


def main(args):
    config, _ = load_expr_config(args, AgentRLConfig)

    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )

    valid_dataset = get_custom_dataset(
        split="test",
        dataset_config=config.valid_dataset,
        tokenizer=tokenizer,
    )

    with GRPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        tokenizer=tokenizer,
    ) as trainer:
        workflow = OpenAIAgentWorkflow(
            agent_builder_path=config.agent_builder_path,
            agent_builder_kwargs=config.agent_builder_kwargs,
            reward_fn_path=config.reward_fn_path,
            gconfig=config.gconfig,
            n_trajs=config.n_trajs,
            tokenizer=tokenizer,
            dump_dir=os.path.join(
                StatsLogger.get_log_path(config.stats_logger), "generated"
            ),
        )
        eval_workflow = RLVRWorkflow(
            agent_builder_path=config.agent_builder_path,
            agent_builder_kwargs=config.agent_builder_kwargs,
            reward_fn_path=config.reward_fn_path,
            gconfig=config.gconfig,
            n_trajs=config.n_trajs,
            tokenizer=tokenizer,
            dump_dir=os.path.join(
                StatsLogger.get_log_path(config.stats_logger), "generated-eval"
            ),
        )
        trainer.train(workflow, eval_workflow)


if __name__ == "__main__":
    main(sys.argv[1:])
