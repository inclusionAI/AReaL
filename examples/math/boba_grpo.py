import os
import sys

from datasets import load_dataset

from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.experimental.trainer import PPOTrainer
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.stats_logger import StatsLogger
from areal.workflow.rlvr import RLVRWorkflow


def get_input_ids_fn(data, tokenizer, enable_thinking):
    user_token = "<｜User｜>"
    assistant_token = "<｜Assistant｜>"
    think_token = "<think>"
    has_think_token = think_token in data
    data = (
        data.replace(user_token, "")
        .replace(assistant_token, "")
        .replace(think_token, "")
    )
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": data}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking or has_think_token,
    )
    return input_ids


def data_extract_prompt_fn(data):
    return data["prompt"]


def get_boba_math_dataset(path, tokenizer):
    dataset = load_dataset(
        path="json",
        split="train",
        data_files=path,
    )
    dataset = dataset.filter(lambda x: len(tokenizer.encode(x["prompt"])) <= 1024)
    return dataset


def boba_reward_fn(
    prompts, completions, prompt_ids, completion_ids, solutions, **kwargs
):
    from areal.reward.math_parser import process_results

    label = 0
    for sol in solutions:
        x = process_results(completions, sol)
        label = label or x[0]
    return label


def main(args):
    config, _ = load_expr_config(args, GRPOConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    train_dataset = get_boba_math_dataset(config.train_dataset.path, tokenizer)

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=None,
    ) as trainer:
        workflow = RLVRWorkflow(
            reward_fn=boba_reward_fn,
            gconfig=config.gconfig,
            tokenizer=trainer.tokenizer,
            enable_thinking=True,
            dump_dir=os.path.join(
                StatsLogger.get_log_path(config.stats_logger), "generated"
            ),
            get_input_ids_fn=get_input_ids_fn,
            data_extract_prompt_fn=data_extract_prompt_fn,
        )
        trainer.train(workflow, eval_workflow=None)


if __name__ == "__main__":
    main(sys.argv[1:])
