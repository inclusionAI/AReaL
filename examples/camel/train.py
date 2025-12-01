import asyncio
import os
from dataclasses import dataclass, field

from camel.agents import ChatAgent
from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters, GRPOConfig, load_expr_config
from areal.api.reward_api import AsyncRewardWrapper
from areal.api.workflow_api import RolloutWorkflow
from areal.dataset import get_custom_dataset
from areal.experimental.camel.openai_model import AReaLOpenAICompatibleModel
from areal.experimental.openai import ArealOpenAI
from areal.experimental.trainer import PPOTrainer
from areal.utils import stats_tracker
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.stats_logger import StatsLogger


@dataclass
class AgentRLConfig(GRPOConfig):
    n_trajs: int = field(
        default=1,
        metadata={
            "help": "We could collect multiple trajectories for a single query. By default n_trajs=1."
        },
    )
    max_tokens_per_trajectory: int = field(
        default=32768,
        metadata={
            "help": "Maximum number of tokens per trajectory. By default max_tokens_per_trajectory=32768."
        },
    )


def gsm8k_reward_fn(result, answer):
    from areal.reward.math_parser import process_results

    return int(process_results(result, answer)[0])


class CamelMathAgent:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        max_tokens_per_turn: int = 1024,
        max_total_tokens: int = 32768,
    ):
        self.tokenizer = tokenizer
        self.max_tokens_per_turn = max_tokens_per_turn
        self.max_total_tokens = max_total_tokens
        self.async_reward_fn = AsyncRewardWrapper(gsm8k_reward_fn)

    async def run_agent(self, data, client: ArealOpenAI):
        messages = data["messages"].copy()
        agent = ChatAgent(
            model=AReaLOpenAICompatibleModel(
                openai_client=client, tokenizer=self.tokenizer, model_type="areal"
            ),
            token_limit=self.max_total_tokens,
        )
        response = await agent.astep(messages[-1]["content"])
        content = response.msg.content
        reward = await self.async_reward_fn(result=content, answer=data["answer"])
        client.set_final_reward(reward)
        return reward


class CamelRLVRWorkflow(RolloutWorkflow):
    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        dump_dir: str | None = None,
        rollout_stat_scope: str = "rollout",
        n_trajs: int = 1,
        max_tokens: int = 32768,
    ):
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(tokenizer)
        self.gconfig.n_samples = 1
        self.tokenizer = tokenizer
        self.dump_dir = dump_dir
        self.max_tokens = max_tokens
        self.rollout_stat_scope = rollout_stat_scope
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

        # Search hyper-parameters
        self.n_trajs = n_trajs
        self.agent = CamelMathAgent(
            tokenizer=self.tokenizer,
            max_tokens_per_turn=self.gconfig.max_new_tokens,
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


def main(args):
    config, _ = load_expr_config(args, AgentRLConfig)

    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    # Load dataset
    train_dataset = get_custom_dataset(
        split="train", dataset_config=config.train_dataset, tokenizer=tokenizer
    )

    # Create trainer (no valid_dataset for this example)
    with PPOTrainer(config, train_dataset, valid_dataset=None) as trainer:
        # Create rollout workflow
        workflow = CamelRLVRWorkflow(
            gconfig=config.gconfig,
            tokenizer=trainer.tokenizer,
            n_trajs=config.n_trajs,
            max_tokens=config.max_tokens_per_trajectory,
            dump_dir=os.path.join(
                StatsLogger.get_log_path(config.stats_logger), "generated"
            ),
        )

        # Run training
        trainer.train(workflow, eval_workflow=None)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
