import json
import os
import sys

import torch.distributed as dist

from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.experimental.trainer import PPOTrainer
from areal.reward.math_parser import process_results
from areal.utils import stats_tracker
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.stats_logger import StatsLogger
from areal.workflow.rlvr import RLVRWorkflow


def gsm8k_reward_fn(prompt, completions, prompt_ids, completion_ids, answer, **kwargs):
    return int(process_results(completions, answer)[0])


class MinimalPPOTrainer(PPOTrainer):
    """Trainer that collects stats to JSON for testing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rewards_history = []

    def _export_and_commit_stats(self, epoch, epoch_step, global_step):
        # Collect stats before committing
        stats = stats_tracker.export_all(reduce_group=self.actor.data_parallel_group)
        self.rewards_history.append(stats["ppo_actor/task_reward/avg"])


def main() -> None:
    config, _ = load_expr_config(sys.argv[1:], GRPOConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )

    with MinimalPPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=None,
    ) as trainer:
        workflow = RLVRWorkflow(
            reward_fn=gsm8k_reward_fn,
            gconfig=config.gconfig,
            tokenizer=trainer.tokenizer,
            enable_thinking=False,
            dump_dir=os.path.join(
                StatsLogger.get_log_path(config.stats_logger), "generated"
            ),
        )

        trainer.train(workflow)

        # Save rewards to JSON for test assertions
        if dist.get_rank() == 0:
            with open(os.path.join(config.cluster.fileroot, "rewards.json"), "w") as f:
                json.dump(trainer.rewards_history, f)


if __name__ == "__main__":
    main()
