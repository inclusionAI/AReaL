"""
Multi-turn agentic multi-modal RL training example.
"""

import pathlib
import re
import sys

sys.path.append(str(pathlib.Path(__file__).parent))

from configs import VisionMultiTurnGRPOConfig
from mathruler.grader import extract_boxed_content, grade_answer

from areal import PPOTrainer
from areal.api.cli_args import load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_processor_and_tokenizer


def format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else 0.0


def acc_reward(predict_str: str, ground_truth: str) -> float:
    answer = extract_boxed_content(predict_str)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def geometry3k_reward_fn(
    prompt, completions, prompt_ids, completion_ids, answer, **kwargs
):
    format_reward_val = format_reward(completions)
    acc_reward_val = acc_reward(completions, answer)
    format_score = 0.1
    score = (1.0 - format_score) * (acc_reward_val) + format_score * format_reward_val
    return score


def main(args):
    """Main training function."""
    config, _ = load_expr_config(args, VisionMultiTurnGRPOConfig)

    # Load tokenizer and processor
    processor, tokenizer = load_hf_processor_and_tokenizer(config.tokenizer_path)

    # Load datasets
    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
        processor=processor,
    )

    valid_dataset = get_custom_dataset(
        split="test",
        dataset_config=config.valid_dataset,
        tokenizer=tokenizer,
        processor=processor,
    )

    # Setup workflow kwargs
    workflow_kwargs = dict(
        reward_fn="examples.vlm_multiturn.vlm_multiturn_grpo.geometry3k_reward_fn",
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        processor=config.tokenizer_path,
        max_turns=config.max_turns,
        turn_discount=config.turn_discount,
    )

    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["gconfig"] = config.gconfig.new(temperature=0.6, n_samples=1)

    # Train
    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow=("areal.workflow.vision_multiturn.VisionMultiTurnWorkflow"),
            workflow_kwargs=workflow_kwargs,
            eval_workflow=("areal.workflow.vision_multiturn.VisionMultiTurnWorkflow"),
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
