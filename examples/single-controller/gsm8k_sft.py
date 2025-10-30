import sys

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import SFTConfig, load_expr_config
from areal.api.io_struct import FinetuneSpec, StepInfo
from areal.api.scheduler_api import ScheduleStrategy
from areal.controller.train_controller import TrainController
from areal.dataset import get_custom_dataset
from areal.engine.sft.lm_engine import FSDPLMEngine
from areal.scheduler.local import LocalScheduler
from areal.utils import logging, stats_tracker
from areal.utils.data import (
    pad_sequences_to_tensors,
)
from areal.utils.dataloader import create_dataloader
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger

logger = logging.getLogger("Trainer")


def main(args):
    config, _ = load_expr_config(args, SFTConfig)
    config: SFTConfig

    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    # Create dataset and dataloaders
    train_dataset = get_custom_dataset(
        split="train", dataset_config=config.train_dataset, tokenizer=tokenizer
    )
    valid_dataset = get_custom_dataset(
        split="test", dataset_config=config.valid_dataset, tokenizer=tokenizer
    )
    train_dataloader = create_dataloader(
        train_dataset,
        rank=0,
        world_size=1,
        dataset_config=config.train_dataset,
        collate_fn=pad_sequences_to_tensors,
    )
    valid_dataloader = create_dataloader(
        valid_dataset,
        rank=0,
        world_size=1,
        dataset_config=config.valid_dataset,
        collate_fn=pad_sequences_to_tensors,
    )

    # Initialize engine
    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(train_dataloader) * config.train_dataset.batch_size,
        train_batch_size=config.train_dataset.batch_size,
    )

    # Initialize scheduler
    scheduler = LocalScheduler(
        fileroot=config.cluster.fileroot,
        experiment_name=config.experiment_name,
        trial_name=config.trial_name,
    )
    # Initialize train controller
    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    engine = TrainController(FSDPLMEngine, config=config.model, scheduler=scheduler)
    engine.initialize(
        role="default",
        alloc_mode=allocation_mode,
        ft_spec=ft_spec,
        schedule_strategy=ScheduleStrategy(),
        addr=None,
    )

    # Run training.
    saver = Saver(config.saver, ft_spec)
    stats_logger = StatsLogger(config, ft_spec)
    evaluator = Evaluator(config.evaluator, ft_spec)

    recover_handler = RecoverHandler(config.recover, ft_spec)
    recover_info = recover_handler.load(
        engine,
        saver,
        evaluator,
        stats_logger,
        train_dataloader,
    )
    start_step = (
        recover_info.last_step_info.next().global_step
        if recover_info is not None
        else 0
    )

    total_epochs = config.total_train_epochs

    global_step = 0
    for epoch in range(total_epochs):
        for step, data in enumerate(train_dataloader):
            if global_step < start_step:
                global_step += 1
                continue
            step_info = StepInfo(
                global_step=global_step,
                epoch=epoch,
                epoch_step=step,
                steps_per_epoch=len(train_dataloader),
            )

            with (
                stats_tracker.record_timing("train_step"),
            ):
                engine.train_lm(data)
                engine.step_lr_scheduler()

            with stats_tracker.record_timing("save"):
                saver.save(engine, epoch, step, global_step, tokenizer=tokenizer)

            with stats_tracker.record_timing("checkpoint_for_recover"):
                recover_handler.dump(
                    engine,
                    step_info,
                    saver,
                    evaluator,
                    stats_logger,
                    train_dataloader,
                    tokenizer=tokenizer,
                )

            with stats_tracker.record_timing("eval"):

                def evaluate_fn():
                    for data in valid_dataloader:
                        engine.evaluate_lm(data)

                evaluator.evaluate(evaluate_fn, epoch, step, global_step)

            stats_logger.commit(epoch, step, global_step, engine.export_stats())
            global_step += 1

    stats_logger.close()
    engine.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])
