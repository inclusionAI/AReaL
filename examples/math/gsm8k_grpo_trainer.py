import os
import sys

from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.trainer.rl import GRPOTrainer
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.stats_logger import StatsLogger
from areal.workflow.rlvr import RLVRWorkflow


def gsm8k_reward_fn(prompt, completions, prompt_ids, completion_ids, answer, **kwargs):
    from areal.reward.math_parser import process_results

    return int(process_results(completions, answer)[0])


def main(args):
    config, _ = load_expr_config(args, GRPOConfig)
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
        workflow = RLVRWorkflow(
            reward_fn=gsm8k_reward_fn,
            gconfig=config.gconfig,
            tokenizer=tokenizer,
            enable_thinking=False,
            dump_dir=os.path.join(
                StatsLogger.get_log_path(config.stats_logger), "generated"
            ),
        )
        eval_workflow = RLVRWorkflow(
            reward_fn=gsm8k_reward_fn,
            gconfig=config.gconfig.new(temperature=0.6),
            tokenizer=tokenizer,
            enable_thinking=False,
            rollout_stat_scope="eval-rollout",
            dump_dir=os.path.join(
                StatsLogger.get_log_path(config.stats_logger), "generated-eval"
            ),
        )
        trainer.train(workflow, eval_workflow)


if __name__ == "__main__":
    main(sys.argv[1:])
