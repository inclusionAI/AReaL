import os
import sys

from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from torchdata.stateful_dataloader import StatefulDataLoader

from arealite.api.cli_args import SFTConfig, load_expr_config
from arealite.api.io_struct import FinetuneSpec
from arealite.engine.sft.lm_engine import FSDPLMEngine
from arealite.utils.data import pad_sequences_to_tensors
from arealite.utils.evaluator import Evaluator
from arealite.utils.saver import Saver
from arealite.utils.stats_logger import StatsLogger
from realhf.api.core.data_api import load_hf_tokenizer
from realhf.base import stats_tracker


def process_gsm8k_sft_dataset(dataset: Dataset, tokenizer):
    def process(sample):
        seq_token = tokenizer.encode(
            sample["question"] + sample["answer"] + tokenizer.eos_token
        )
        prompt_token = tokenizer.encode(sample["question"])
        loss_mask = [0] * len(prompt_token) + [1] * (len(seq_token) - len(prompt_token))
        return {"input_ids": seq_token, "loss_mask": loss_mask}

    dataset = dataset.map(process).remove_columns(["question", "answer"])
    return dataset


def get_gsm8k_dataset(split, tokenizer, rank, world_size):
    dataset = load_dataset(path="openai/gsm8k", name="main", split=split)
    dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
    return process_gsm8k_sft_dataset(dataset, tokenizer)


def main_sft():
    config, _ = load_expr_config(sys.argv[1:], SFTConfig)
    config: SFTConfig

    rank = int(os.getenv("RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    # Create dataset and dataloaders
    train_dataloader = StatefulDataLoader(
        get_gsm8k_dataset("train", tokenizer, rank, world_size),
        batch_size=config.train_dataset.batch_size // world_size,
        shuffle=config.train_dataset.shuffle,
        num_workers=config.train_dataset.num_workers,
        collate_fn=pad_sequences_to_tensors,
        drop_last=config.train_dataset.drop_last,
    )
    valid_dataloader = StatefulDataLoader(
        get_gsm8k_dataset("test", tokenizer, rank, world_size),
        batch_size=config.valid_dataset.batch_size // world_size,
        shuffle=config.valid_dataset.shuffle,
        num_workers=config.valid_dataset.num_workers,
        collate_fn=pad_sequences_to_tensors,
        drop_last=config.valid_dataset.drop_last,
    )

    # Initialize engine
    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(train_dataloader) * config.train_dataset.batch_size,
        train_batch_size=config.train_dataset.batch_size,
    )
    engine = FSDPLMEngine(config=config.model)
    engine.initialize(None, ft_spec)

    # Run training.
    saver = Saver(config.saver, ft_spec, for_recover=False)
    logger = StatsLogger(config.stats_logger, ft_spec)
    evaluator = Evaluator(config.evaluator, ft_spec)

    total_epochs = config.total_train_epochs
    steps_per_epoch = len(train_dataloader)

    logger.info(f"total_epochs={total_epochs} step_per_epoch={steps_per_epoch}")
    global_step = 0
    for epoch in range(total_epochs):
        for step, data in enumerate(train_dataloader):
            with (
                stats_tracker.record_timing("train_step"),
                stats_tracker.scope("sft"),
            ):
                stats = engine.train_lm(data)
                engine.step_lr_scheduler()
                stats_tracker.scalar(**stats)

            with stats_tracker.record_timing("save"):
                saver.save(engine, epoch, step, global_step)

            with stats_tracker.record_timing("eval"):
                # No need to log anything. Logging will be handled outside
                # via stats_tracker.export().
                def evaluate_fn():
                    with stats_tracker.scope("sft-eval"):
                        for data in valid_dataloader:
                            engine.evaluate_lm(data)

                evaluator.evaluate(
                    evaluate_fn,
                    epoch,
                    step,
                    global_step,
                )

            logger.commit(
                epoch,
                step,
                global_step,
                stats_tracker.export(reduce_group=engine.parallelism_group),
            )
            global_step += 1

    logger.close()
    engine.destroy()


if __name__ == "__main__":
    main_sft()
