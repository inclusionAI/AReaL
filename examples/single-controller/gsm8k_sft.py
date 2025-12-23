import sys

from areal.api.cli_args import SFTConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.experimental.trainer import SFTTrainer
from areal.utils.hf_utils import load_hf_tokenizer


def main(args):
    config, _ = load_expr_config(args, SFTConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    train_dataset = get_custom_dataset("train", config.train_dataset, tokenizer)
    valid_dataset = get_custom_dataset("test", config.valid_dataset, tokenizer)

    with SFTTrainer(config, train_dataset, valid_dataset) as trainer:
        trainer.train()


if __name__ == "__main__":
    main(sys.argv[1:])
