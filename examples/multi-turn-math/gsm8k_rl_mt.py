import asyncio
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field

from openai.types.chat import ChatCompletion
from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters, GRPOConfig, load_expr_config
from areal.api.reward_api import AsyncRewardWrapper
from areal.api.workflow_api import RolloutWorkflow
from areal.dataset import get_custom_dataset
from areal.experimental.openai import ArealOpenAI
from areal.experimental.trainer import PPOTrainer
from areal.reward import get_math_verify_worker
from areal.utils import stats_tracker
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.stats_logger import StatsLogger


def gsm8k_reward_fn(result, answer):
    try:
        worker = get_math_verify_worker()
        return worker.verify(str(result), str(answer))
    except Exception:
        return 0.0


class MultiTurnMathAgent:
    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        reward_fn: Callable[[str, str], float | int],
        max_turns: int = 2,
    ):
        self.gconfig = gconfig
        self.max_turns = max_turns
        self.async_reward_fn = AsyncRewardWrapper(reward_fn)

    async def run_agent(self, data, client: ArealOpenAI):
        messages = data["messages"].copy()
        for _ in range(self.max_turns):
            response: ChatCompletion = await client.chat.completions.create(
                messages=messages,
                **self.gconfig.to_openai_args_dict(),
            )
            message = response.choices[0].message
            messages.append(message)
            reward = await self.async_reward_fn(
                result=message.content, answer=data["answer"]
            )
            client.set_reward(response.id, reward)
            if reward == 1:
                break
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": "Your answer is either wrong or not parsable to the reward function. You may misunderstand the original question. "
                        "Please carefully read the original question, check the previous errors, and try to answer it again.",
                    }
                )
        return reward


class MultiturnRLVRWorkflow(RolloutWorkflow):
    def __init__(
        self,
        reward_fn: Callable[[str, str], float | int],
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        dump_dir: str | None = None,
        rollout_stat_scope: str = "rollout",
        export_style: str = "concat",
        max_turns: int = 2,
    ):
        self.n_trajs = gconfig.n_samples
        self.tokenizer = tokenizer
        self.dump_dir = dump_dir
        self.rollout_stat_scope = rollout_stat_scope
        self.export_style = export_style
        if export_style not in ["individual", "concat"]:
            raise ValueError(f"Invalid export style: {export_style}")
        self.chat_template_type = "concat" if export_style == "concat" else "hf"

        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

        # Search hyper-parameters
        self.agent = MultiTurnMathAgent(
            gconfig=gconfig.new(n_samples=1),
            reward_fn=reward_fn,
            max_turns=max_turns,
        )

    async def arun_episode(self, engine, data):
        clients = [
            ArealOpenAI(
                engine=engine,
                tokenizer=self.tokenizer,
                chat_template_type=self.chat_template_type,
            )
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

        completions_with_reward = {}
        for client in clients:
            client.apply_reward_discount(turn_discount=0.9)
            completions = client.export_interactions(style=self.export_style)
            completions_with_reward.update(completions)
        return completions_with_reward


@dataclass
class MultiTurnGRPOConfig(GRPOConfig):
    agent_run_args: dict = field(
        default_factory=dict,
        metadata={"help": "Arguments for running the agent."},
    )
    export_style: str = field(
        default="concat",
        metadata={
            "help": "Export style for the completions. By default export_style=concat."
        },
    )


def main(args):
    config, _ = load_expr_config(args, MultiTurnGRPOConfig)
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

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        max_turns = config.agent_run_args.get("max_turns", 2)
        log_path = StatsLogger.get_log_path(config.stats_logger)

        workflow = MultiturnRLVRWorkflow(
            reward_fn=gsm8k_reward_fn,
            gconfig=config.gconfig,
            tokenizer=trainer.tokenizer,
            dump_dir=os.path.join(log_path, "generated"),
            export_style=config.export_style,
            max_turns=max_turns,
        )
        eval_workflow = MultiturnRLVRWorkflow(
            reward_fn=gsm8k_reward_fn,
            gconfig=config.gconfig.new(temperature=0.6, n_samples=1),
            tokenizer=trainer.tokenizer,
            rollout_stat_scope="eval-rollout",
            dump_dir=os.path.join(log_path, "generated-eval"),
            export_style=config.export_style,
            max_turns=max_turns,
        )
        trainer.train(workflow, eval_workflow)


if __name__ == "__main__":
    main(sys.argv[1:])
