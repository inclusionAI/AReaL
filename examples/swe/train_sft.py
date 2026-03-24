import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent))
from config import SweSFTConfig

from areal import SFTTrainer
from areal.api.cli_args import load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_tokenizer


def main(args):
    config, _ = load_expr_config(args, SweSFTConfig)

    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    swe_kwargs = {
        "num_proc": config.swe.num_proc,
        "pre_split": config.swe.pre_split,
        "filter_errors": config.swe.filter_errors,
        "strip_all_thinking": config.swe.strip_all_thinking,
        "no_tools": config.swe.no_tools,
    }

    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
        **swe_kwargs,
    )
    valid_dataset = None
    if config.valid_dataset is not None:
        valid_dataset = get_custom_dataset(
            split="test",
            dataset_config=config.valid_dataset,
            tokenizer=tokenizer,
            **swe_kwargs,
        )

    with SFTTrainer(
        config, train_dataset=train_dataset, valid_dataset=valid_dataset
    ) as trainer:
        trainer.train()


if __name__ == "__main__":
    main(sys.argv[1:])
