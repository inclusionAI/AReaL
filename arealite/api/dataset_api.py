from dataclasses import dataclass

from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node

from arealite.api.cli_args import DatasetConfig
from arealite.utils import TrainingArgs


def create_distributed_dataset(cfg: DatasetConfig, rank, world_size):
    dataset = load_dataset(
        cfg.path,
        name=cfg.name,
        split=cfg.split,
        data_files=cfg.data_files,
    )
    dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
    return dataset


@dataclass
class DatasetFactory:
    args: TrainingArgs

    def make_dataset(
        self, config: DatasetConfig, rank: int, world_size: int
    ) -> Dataset:
        dataset = create_distributed_dataset(config, rank, world_size)
        if config.preprocessor.type == "gsm8k_rl":
            from arealite.impl.dataset.gsm8k import process_gsm8k_rl_dataset

            tokenizer_path = self.args.rollout.model_path
            assert self.args.rollout.model_path is not None
            from realhf.api.core.data_api import load_hf_tokenizer

            tokenizer = load_hf_tokenizer(tokenizer_path)
            return process_gsm8k_rl_dataset(
                dataset,
                tokenizer=tokenizer,
                reward_mode=config.preprocessor.gsm8k.reward_mode,
            )
        if config.preprocessor.type == "gsm8k_sft":
            from arealite.impl.dataset.gsm8k import process_gsm8k_sft_dataset

            tokenizer_path = self.args.trainer.sft.model.path
            from realhf.api.core.data_api import load_hf_tokenizer

            tokenizer = load_hf_tokenizer(tokenizer_path)
            return process_gsm8k_sft_dataset(dataset, tokenizer=tokenizer)
        if config.preprocessor.type == "areal":
            tokenizer_path = self.args.rollout.model_path
            assert self.args.rollout.model_path is not None
            from realhf.api.core.data_api import load_hf_tokenizer

            tokenizer = load_hf_tokenizer(tokenizer_path)
            from arealite.impl.dataset.areal import process_areal_dataset

            return process_areal_dataset(dataset, tokenizer=tokenizer)
        if config.preprocessor.type == "llava_cot":
            tokenizer_path = self.args.rollout.llm_client.tokenizer_path
            assert self.args.rollout.llm_client.tokenizer_path is not None
            from realhf.api.core.data_api import load_hf_processor_and_tokenizer

            processor, _ = load_hf_processor_and_tokenizer(tokenizer_path)
            from arealite.impl.dataset.llava_cot100k import process_llava_cot_dataset

            return process_llava_cot_dataset(dataset, processor=processor)
        if config.preprocessor.type == "math_instruct":
            tokenizer_path = self.args.rollout.llm_client.tokenizer_path
            assert self.args.rollout.llm_client.tokenizer_path is not None
            from realhf.api.core.data_api import load_hf_processor_and_tokenizer

            processor, _ = load_hf_processor_and_tokenizer(tokenizer_path)
            from arealite.impl.dataset.MM_MathInstruct import (
                process_MathInstruct_dataset,
            )

            return process_MathInstruct_dataset(dataset, processor=processor)
        raise NotImplementedError(
            f"Unknown dataset preprocessor type: {config.preprocessor.type}"
        )
