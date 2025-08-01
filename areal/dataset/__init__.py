from typing import Optional

import transformers

VALID_DATASETS = ["gsm8k", "clevr_count_70k"]


def get_custom_dataset(
    path: str,
    rank: int,
    world_size: int,
    type: str = "sft",
    split: Optional[str] = None,
    tokenizer: Optional[transformers.PreTrainedTokenizerFast] = None,
    processor: Optional[transformers.AutoProcessor] = None,
    **kwargs,
):

    if "gsm8k" in path and type == "sft":
        from areal.dataset.gsm8k import get_gsm8k_sft_dataset

        return get_gsm8k_sft_dataset(path, split, tokenizer, rank, world_size, **kwargs)
    elif "gsm8k" in path and type == "rl":
        from areal.dataset.gsm8k import get_gsm8k_rl_dataset

        return get_gsm8k_rl_dataset(path, split, rank, world_size, **kwargs)
    elif "clevr_count_70k" in path and type == "sft":
        from areal.dataset.clevr_count_70k import get_clevr_count_70k_sft_dataset

        return get_clevr_count_70k_sft_dataset(
            path, split, processor, rank, world_size, **kwargs
        )
    elif "clevr_count_70k" in path and type == "rl":
        from areal.dataset.clevr_count_70k import get_clevr_count_70k_rl_dataset

        return get_clevr_count_70k_rl_dataset(
            path, split, processor, rank, world_size, **kwargs
        )
    else:
        raise ValueError(
            f"Dataset {path} with split {split} and training type {type} is not supported. "
            f"Supported datasets are: {VALID_DATASETS}. "
        )
