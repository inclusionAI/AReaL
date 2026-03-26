import getpass
import os
import pathlib
import shutil
import sys

sys.path.append(str(pathlib.Path(__file__).parent))
from config import SweSFTConfig

from areal import SFTTrainer
from areal.api.cli_args import load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.logging import getLogger

logger = getLogger("SweSFTTrain")


def _get_cache_dir(config: SweSFTConfig) -> str:
    """Build the processed-dataset cache path next to checkpoints.

    Layout: ``{fileroot}/checkpoints/{user}/{experiment}/{trial}/processed_dataset``
    """
    return os.path.join(
        config.cluster.fileroot,
        "checkpoints",
        getpass.getuser(),
        config.experiment_name,
        config.trial_name,
        "processed_dataset",
    )


def main(args):
    config, _ = load_expr_config(args, SweSFTConfig)

    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    rank = int(os.getenv("RANK", "0"))
    cache_dir = _get_cache_dir(config)

    dump_dir = None
    if config.swe.dump_samples != 0:
        dump_dir = os.path.join(
            config.cluster.fileroot,
            "logs",
            getpass.getuser(),
            config.experiment_name,
            config.trial_name,
            "dumped_samples",
        )

    swe_kwargs = {
        "num_proc": config.swe.num_proc,
        "pre_split": config.swe.pre_split,
        "filter_errors": config.swe.filter_errors,
        "strip_all_thinking": config.swe.strip_all_thinking,
        "filter_empty_tool_calls": config.swe.filter_empty_tool_calls,
        "filter_bare_text_tool_calls": config.swe.filter_bare_text_tool_calls,
        "truncate_task_notifications": config.swe.truncate_task_notifications,
        "no_tools": config.swe.no_tools,
        "cache_dir": cache_dir,
        "dump_dir": dump_dir,
        "dump_samples": config.swe.dump_samples,
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

    # Cleanup processed dataset cache after training.
    if config.swe.cleanup_processed_dataset and rank == 0:
        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir)
            logger.info(f"Cleaned up processed dataset cache: {cache_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
