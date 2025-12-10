import os
import sys

from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.experimental.trainer import PPOTrainer
from areal.scheduler.local import LocalScheduler
from areal.utils.environ import is_single_controller
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.stats_logger import StatsLogger


def main(args):
    config, _ = load_expr_config(args, GRPOConfig)
    if not is_single_controller():
        raise RuntimeError(
            "This script should be directly run wihout using areal.launcher"
        )

    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    # Create dataset and dataloaders
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

    # Initialize scheduler
    scheduler = LocalScheduler(exp_config=config)

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        scheduler=scheduler,
    ) as trainer:
        workflow_kwargs = dict(
            reward_fn="areal.reward.gsm8k.gsm8k_reward_fn",
            gconfig=config.gconfig,
            tokenizer=config.tokenizer_path,
            enable_thinking=False,
            dump_dir=os.path.join(
                StatsLogger.get_log_path(config.stats_logger),
                "generated",
            ),
        )
        eval_workflow_kwargs = dict(
            reward_fn="areal.reward.gsm8k.gsm8k_reward_fn",
            gconfig=config.gconfig.new(temperature=0.6),
            tokenizer=config.tokenizer_path,
            enable_thinking=False,
            rollout_stat_scope="eval-rollout",
            dump_dir=os.path.join(
                StatsLogger.get_log_path(config.stats_logger),
                "generated-eval",
            ),
        )
        trainer.train(
            workflow="areal.workflow.rlvr.RLVRWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="areal.workflow.rlvr.RLVRWorkflow",
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
