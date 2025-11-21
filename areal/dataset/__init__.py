from typing import TYPE_CHECKING, Optional

from datasets.distributed import split_dataset_by_node

from areal.api.cli_args import DatasetConfig
from areal.utils import logging

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers.processing_utils import ProcessorMixin
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

VALID_DATASETS = ["gsm8k", "clevr_count_70k", "geometry3k", "hh-rlhf", "torl_data"]

logger = logging.getLogger("Dataset")


def _get_custom_dataset(
    path: str,
    type: str = "sft",
    split: str | None = None,
    max_length: int | None = None,
    tokenizer: Optional["PreTrainedTokenizerFast"] = None,
    processor: Optional["ProcessorMixin"] = None,
    **kwargs,
) -> "Dataset":

    if "gsm8k" in path and type == "sft":
        from .gsm8k import get_gsm8k_sft_dataset

        return get_gsm8k_sft_dataset(
            path=path,
            split=split,
            tokenizer=tokenizer,
            max_length=max_length,
            **kwargs,
        )
    elif "gsm8k" in path and type == "rl":
        # Check if this is the reasoning variant
        if "reasoning" in path.lower() or kwargs.get("reasoning", False):
            from .gsm8k import get_gsm8k_reasoning_rl_dataset

            return get_gsm8k_reasoning_rl_dataset(
                path=path,
                split=split,
                tokenizer=tokenizer,
                max_length=max_length,
                **kwargs,
            )
        else:
            from .gsm8k import get_gsm8k_rl_dataset

            return get_gsm8k_rl_dataset(
                path=path,
                split=split,
            tokenizer=tokenizer,
            max_length=max_length,
            **kwargs,
        )
    elif "clevr_count_70k" in path and type == "sft":
        from .clevr_count_70k import get_clevr_count_70k_sft_dataset

        return get_clevr_count_70k_sft_dataset(
            path=path,
            split=split,
            processor=processor,
            max_length=max_length,
            **kwargs,
        )
    elif "clevr_count_70k" in path and type == "rl":
        from .clevr_count_70k import get_clevr_count_70k_rl_dataset

        return get_clevr_count_70k_rl_dataset(
            path=path,
            split=split,
            processor=processor,
            max_length=max_length,
            **kwargs,
        )
    elif "geometry3k" in path and type == "sft":
        from .geometry3k import get_geometry3k_sft_dataset

        return get_geometry3k_sft_dataset(
            path=path,
            split=split,
            processor=processor,
            max_length=max_length,
            **kwargs,
        )
    elif "geometry3k" in path and type == "rl":
        from .geometry3k import get_geometry3k_rl_dataset

        return get_geometry3k_rl_dataset(
            path=path,
            split=split,
            processor=processor,
            max_length=max_length,
            **kwargs,
        )
    elif "hh-rlhf" in path and type == "rw":
        from .hhrlhf import get_hhrlhf_rw_dataset

        return get_hhrlhf_rw_dataset(
            path=path,
            split=split,
            tokenizer=tokenizer,
            max_length=max_length,
            **kwargs,
        )
    elif "torl_data" in path and type == "rl":
        from .torl_data import get_torl_data_rl_dataset

        return get_torl_data_rl_dataset(
            path=path,
            split=split,
            tokenizer=tokenizer,
            max_length=max_length,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Dataset {path} with split {split} and training type {type} is not supported. "
            f"Supported datasets are: {VALID_DATASETS}. "
        )


def get_custom_dataset_legacy(
    path: str,
    rank: int,
    world_size: int,
    type: str = "sft",
    split: str | None = None,
    max_length: int | None = None,
    tokenizer: Optional["PreTrainedTokenizerFast"] = None,
    processor: Optional["ProcessorMixin"] = None,
    **kwargs,
) -> "Dataset":
    logger.warning(
        "get_custom_dataset using rank and world_size is deprecated. "
        "Please use DistributedSampler in dataloader instead for distributed training."
    )
    dataset = _get_custom_dataset(
        path=path,
        type=type,
        split=split,
        max_length=max_length,
        tokenizer=tokenizer,
        processor=processor,
        **kwargs,
    )
    return split_dataset_by_node(dataset, rank=rank, world_size=world_size)


def get_custom_dataset(
    split: str | None = None,
    dataset_config: DatasetConfig | None = None,
    tokenizer: Optional["PreTrainedTokenizerFast"] = None,
    processor: Optional["ProcessorMixin"] = None,
    **kwargs,
) -> "Dataset":
    if "rank" in kwargs:
        # compatibility for legacy get_custom_dataset
        return get_custom_dataset_legacy(
            split=split,
            tokenizer=tokenizer,
            processor=processor,
            **kwargs,
        )

    if dataset_config is not None:
        return _get_custom_dataset(
            path=dataset_config.path,
            type=dataset_config.type,
            split=split,
            max_length=dataset_config.max_length,
            tokenizer=tokenizer,
            processor=processor,
            **kwargs,
        )

    # try to pass arguments directly to legacy get_custom_dataset
    logger.warning("dataset_config is not provided")
    return _get_custom_dataset(
        split=split,
        tokenizer=tokenizer,
        processor=processor,
        **kwargs,
    )
