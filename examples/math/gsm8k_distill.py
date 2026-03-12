"""GSM8K self-distillation training.

Uses the SelfDistillRLVRWorkflow to generate responses, compute rewards,
and generate feedback for successful responses. The SelfDistillActor then
computes the self-distillation KL loss between the student (conditioned
on prompt only) and teacher (conditioned on prompt + feedback).
"""

import sys

from areal import SelfDistillationTrainer
from areal.api.cli_args import SelfDistillConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_tokenizer


def main(args):
    config, _ = load_expr_config(args, SelfDistillConfig)
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

    workflow_kwargs = dict(
        reward_fn="areal.reward.gsm8k.gsm8k_reward_fn",
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        feedback_template=config.actor.self_distillation.feedback_template,
        success_reward_threshold=0.5,
        enable_thinking=False,
    )

    with SelfDistillationTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow="areal.workflow.rlvr_distill.SelfDistillRLVRWorkflow",
            workflow_kwargs=workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
