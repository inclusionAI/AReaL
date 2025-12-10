import sys

from areal.api.cli_args import SFTConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.experimental.trainer import SFTTrainer
from areal.scheduler.local import LocalScheduler
from areal.utils.environ import is_single_controller
from areal.utils.hf_utils import load_hf_tokenizer


def main(args):
    config, _ = load_expr_config(args, SFTConfig)
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

    with SFTTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        scheduler=scheduler,
    ) as trainer:
        trainer.train()


if __name__ == "__main__":
    main(sys.argv[1:])
