from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers.processing_utils import ProcessorMixin
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

VALID_DATASETS = ["gsm8k", "clevr_count_70k", "geometry3k", "hh-rlhf", "torl_data"]


def get_custom_dataset(
    path: str,
    rank: int,
    world_size: int,
    type: str = "sft",
    split: Optional[str] = None,
    max_length: Optional[int] = None,
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
            rank=rank,
            world_size=world_size,
            max_length=max_length,
            **kwargs,
        )
    elif "gsm8k" in path and type == "rl":
        from .gsm8k import get_gsm8k_rl_dataset

        return get_gsm8k_rl_dataset(
            path=path,
            split=split,
            tokenizer=tokenizer,
            rank=rank,
            world_size=world_size,
            max_length=max_length,
            **kwargs,
        )
    elif "clevr_count_70k" in path and type == "sft":
        from .clevr_count_70k import get_clevr_count_70k_sft_dataset

        return get_clevr_count_70k_sft_dataset(
            path=path,
            split=split,
            processor=processor,
            rank=rank,
            world_size=world_size,
            max_length=max_length,
            **kwargs,
        )
    elif "clevr_count_70k" in path and type == "rl":
        from .clevr_count_70k import get_clevr_count_70k_rl_dataset

        return get_clevr_count_70k_rl_dataset(
            path=path,
            split=split,
            processor=processor,
            rank=rank,
            world_size=world_size,
            max_length=max_length,
            **kwargs,
        )
    elif "geometry3k" in path and type == "sft":
        from .geometry3k import get_geometry3k_sft_dataset

        return get_geometry3k_sft_dataset(
            path=path,
            split=split,
            processor=processor,
            rank=rank,
            world_size=world_size,
            max_length=max_length,
            **kwargs,
        )
    elif "geometry3k" in path and type == "rl":
        from .geometry3k import get_geometry3k_rl_dataset

        return get_geometry3k_rl_dataset(
            path=path,
            split=split,
            processor=processor,
            rank=rank,
            world_size=world_size,
            max_length=max_length,
            **kwargs,
        )
    elif "hh-rlhf" in path and type == "rw":
        from .hhrlhf import get_hhrlhf_rw_dataset

        return get_hhrlhf_rw_dataset(
            path=path,
            split=split,
            tokenizer=tokenizer,
            rank=rank,
            world_size=world_size,
            max_length=max_length,
            **kwargs,
        )
    elif "torl_data" in path and type == "rl":
        from .torl_data import get_torl_data_rl_dataset

        return get_torl_data_rl_dataset(
            path=path,
            split=split,
            tokenizer=tokenizer,
            rank=rank,
            world_size=world_size,
            max_length=max_length,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Dataset {path} with split {split} and training type {type} is not supported. "
            f"Supported datasets are: {VALID_DATASETS}. "
        )
