import os
import sys
os.environ["RANK"]='0'
os.environ["WORLD_SIZE"]='1'
from torchdata.stateful_dataloader import StatefulDataLoader

from arealite.api.cli_args import SFTConfig, load_expr_config
from arealite.api.io_struct import FinetuneSpec
from arealite.engine.sft.lm_engine import VL_FSDPLMEngine
from arealite.utils.data import pad_sequences_to_tensors
from arealite.utils.evaluator import Evaluator
from arealite.utils.saver import Saver
from arealite.utils.stats_logger import StatsLogger
from realhf.api.core.data_api import load_hf_processor_and_tokenizer
from realhf.base import stats_tracker
from arealite.dataset.__init__ import get_custom_dataset


def main_sft():
    config, _ = load_expr_config(sys.argv[1:], SFTConfig)
    config: SFTConfig

    rank = int(os.getenv("RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))
    processor, tokenizer = load_hf_processor_and_tokenizer(config.tokenizer_path)
    train_dataset=get_custom_dataset(
                    path=config.train_dataset.path,
                    rank=rank,
                    world_size=world_size,
                    split="train",
                    training_type="sft",
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

    # Initialize engine
    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(train_dataloader) * config.train_dataset.batch_size,
        train_batch_size=config.train_dataset.batch_size,
    )
    engine = VL_FSDPLMEngine(config=config.model)
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

            # with stats_tracker.record_timing("eval"), stats_tracker.scope("sft-eval"):
            #     # No need to log anything. Logging will be handled outside
            #     # via stats_tracker.export().
            #     evaluator.evaluate(
            #         valid_dataloader,
            #         engine.evaluate_lm,
            #         epoch,
            #         step,
            #         global_step,
            #     )

            logger.commit(epoch, step, global_step, stats_tracker.export())
            global_step += 1

    engine.destroy()
    logger.close()


if __name__ == "__main__":
    main_sft()
