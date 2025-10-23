import asyncio
import os

from agents import Agent as OpenAIAgent
from agents import ModelSettings, OpenAIProvider, RunConfig, SQLiteSession
from agents import Runner as OpenAIRunner
from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters
from areal.api.reward_api import AsyncRewardWrapper
from areal.api.workflow_api import RolloutWorkflow
from areal.experimental.openai import ArealOpenAI
from areal.utils import stats_tracker


def gsm8k_reward_fn(result, answer):
    from areal.reward.math_parser import process_results

    return int(process_results(result, answer)[0])


class MultiturnMathAgent:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        max_tokens_per_turn: int = 1024,
        max_turns: int = 8,
        max_total_tokens: int = 32768,
    ):
        self.tokenizer = tokenizer
        self.max_tokens_per_turn = max_tokens_per_turn
        self.max_turns = max_turns
        self.max_total_tokens = max_total_tokens
        self.async_reward_fn = AsyncRewardWrapper(gsm8k_reward_fn)

    async def run_agent(self, data, client: ArealOpenAI):
        num_turns_left = self.max_turns
        run_config = RunConfig(
            model_provider=OpenAIProvider(openai_client=client),
            tracing_disabled=True,
            model_settings=ModelSettings(
                temperature=1.0,
                extra_args={"max_completion_tokens": self.max_tokens_per_turn},
            ),
        )
        agent = OpenAIAgent(
            name="RLVR",
        )
        session = SQLiteSession("math")
        content = data["messages"][-1]["content"]
        reward = 0
        while num_turns_left > 0:
            result = await OpenAIRunner.run(
                agent, input=content, session=session, run_config=run_config
            )
            reward = await self.async_reward_fn(
                result=result.final_output, answer=data["answer"]
            )
            client.set_final_reward(reward)
            if reward == 1:
                break
            else:
                content = "Your answer is either wrong or not parsable to the reward function. You may misunderstand the original question. "
                "Please carefully read the original question, check the preivous errors, and try to answer it again."
            num_turns_left -= 1
        return reward


class MultiturnRLVRAgentWorkflow(RolloutWorkflow):
    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        dump_dir: str | None = None,
        rollout_stat_scope: str = "rollout",
        n_trajs: int = 1,
        max_tokens: int = 32768,
        max_turns: int = 8,
    ):
        self.gconfig = gconfig
        self.gconfig.n_samples = 1
        self.tokenizer = tokenizer
        self.dump_dir = dump_dir
        self.max_tokens = max_tokens
        self.rollout_stat_scope = rollout_stat_scope
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

        # Search hyper-parameters
        self.n_trajs = n_trajs
        self.agent = MultiturnMathAgent(
            tokenizer=self.tokenizer,
            max_tokens_per_turn=self.gconfig.max_new_tokens,
            max_turns=max_turns,
            max_total_tokens=max_tokens,
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
