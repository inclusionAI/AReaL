import os
import sys

from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.cli_args import SFTConfig, load_expr_config
from areal.api.io_struct import FinetuneSpec
from areal.dataset.__init__ import get_custom_dataset
from areal.engine.sft.lm_engine import FSDPLMEngine
from areal.utils.data import pad_sequences_to_tensors
from areal.utils.evaluator import Evaluator
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger
from realhf.api.core.data_api import load_hf_processor_and_tokenizer
from realhf.base import seeding, stats_tracker


def main_sft():
    config, _ = load_expr_config(sys.argv[1:], SFTConfig)
    config: SFTConfig

    rank = int(os.getenv("RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))

    seeding.set_random_seed(config.seed, f"trainer{rank}")

    processor, tokenizer = load_hf_processor_and_tokenizer(config.tokenizer_path)
    train_dataset = get_custom_dataset(
        path=config.train_dataset.path,
        rank=rank,
        world_size=world_size,
        split="train",
        type=config.train_dataset.type,
        tokenizer=tokenizer,
        processor=processor,
    )
    valid_dataset = get_custom_dataset(
        path=config.valid_dataset.path,
        rank=rank,
        world_size=world_size,
        split="test",
        type=config.valid_dataset.type,
        tokenizer=tokenizer,
        processor=processor,
    )

    # Create dataset and dataloaders
    train_dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=config.train_dataset.batch_size // world_size,
        shuffle=config.train_dataset.shuffle,
        num_workers=config.train_dataset.num_workers,
        collate_fn=pad_sequences_to_tensors,
        drop_last=config.train_dataset.drop_last,
    )

    valid_dataloader = StatefulDataLoader(
        valid_dataset,
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
    stats_logger = StatsLogger(config.stats_logger, ft_spec)
    evaluator = Evaluator(config.evaluator, ft_spec)

    total_epochs = config.total_train_epochs
    len(train_dataloader)

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

            stats_logger.commit(
                epoch,
                step,
                global_step,
                stats_tracker.export(reduce_group=engine.parallelism_group),
            )
            global_step += 1

    stats_logger.close()
    engine.destroy()


if __name__ == "__main__":
    main_sft()
