import os
import re
import sys

from mathruler.grader import extract_boxed_content, grade_answer

from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.experimental.trainer import PPOTrainer
from areal.utils.hf_utils import load_hf_processor_and_tokenizer
from areal.utils.stats_logger import StatsLogger
from areal.workflow.vision_rlvr import VisionRLVRWorkflow


def format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else 0.0


def acc_reward(predict_str: str, ground_truth: str) -> float:
    answer = extract_boxed_content(predict_str)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0

def geometry3k_reward_fn(
    prompt, completions, prompt_ids, completion_ids, answer, **kwargs):
    Format_reward = format_reward(completions)
    Acc_reward = acc_reward(completions, answer)
    format_score = 0.1
    score = (1.0 - format_score) * (Acc_reward) + format_score * Format_reward
    return score



def main(args):
    config, _ = load_expr_config(args, GRPOConfig)
    processor, tokenizer = load_hf_processor_and_tokenizer(config.tokenizer_path)

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

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        workflow = VisionRLVRWorkflow(
            reward_fn=geometry3k_reward_fn,
            gconfig=config.gconfig,
            tokenizer=trainer.tokenizer,
            processor=trainer.processor,
            enable_thinking=False,
            dump_dir=os.path.join(
                StatsLogger.get_log_path(config.stats_logger), "generated"
            ),
        )
        eval_workflow = VisionRLVRWorkflow(
            reward_fn=geometry3k_reward_fn,
            gconfig=config.gconfig.new(temperature=0.6),
            tokenizer=trainer.tokenizer,
            processor=trainer.processor,
            enable_thinking=False,
            rollout_stat_scope="eval-rollout",
            dump_dir=os.path.join(
                StatsLogger.get_log_path(config.stats_logger), "generated-eval"
            ),
        )
        trainer.train(workflow, eval_workflow)


if __name__ == "__main__":
    main(sys.argv[1:])
