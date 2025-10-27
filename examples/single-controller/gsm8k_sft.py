import sys

from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import SFTConfig, load_expr_config
from areal.api.io_struct import FinetuneSpec, StepInfo
from areal.api.scheduler_api import ScheduleStrategy
from areal.controller.batch import DistributedBatchMemory
from areal.controller.train_controller import DistributedTrainController
from areal.dataset import get_custom_dataset
from areal.engine.sft.lm_engine import FSDPLMEngine
from areal.scheduler.local import LocalScheduler
from areal.utils import logging, stats_tracker
from areal.utils.data import (
    pad_sequences_to_tensors,
    tensor_container_to,
)
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger

logger = logging.getLogger("Trainer")


def main(args):
    config, _ = load_expr_config(args, SFTConfig)
    config: SFTConfig

    # rank = int(os.getenv("RANK"))
    #
    # seeding.set_random_seed(config.seed, f"trainer{rank}")
    AllocationMode.from_str(config.allocation_mode)

    engine = FSDPLMEngine(config=config.model)
    # engine.create_process_group(parallel_strategy=parallel_strategy)

    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    train_dataset = get_custom_dataset(
        path=config.train_dataset.path,
        rank=0,
        world_size=1,
        split="train",
        max_length=config.train_dataset.max_length,
        type=config.train_dataset.type,
        tokenizer=tokenizer,
    )
    valid_dataset = get_custom_dataset(
        path=config.valid_dataset.path,
        rank=0,
        world_size=1,
        split="test",
        max_length=config.valid_dataset.max_length,
        type=config.valid_dataset.type,
        tokenizer=tokenizer,
    )

    # Create dataset and dataloaders
    train_dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=config.train_dataset.batch_size,
        shuffle=config.train_dataset.shuffle,
        num_workers=config.train_dataset.num_workers,
        collate_fn=pad_sequences_to_tensors,
        drop_last=config.train_dataset.drop_last,
    )
    valid_dataloader = StatefulDataLoader(
        valid_dataset,
        batch_size=config.valid_dataset.batch_size,
        shuffle=config.valid_dataset.shuffle,
        num_workers=config.valid_dataset.num_workers,
        collate_fn=pad_sequences_to_tensors,
        drop_last=config.valid_dataset.drop_last,
    )

    # Initialize engine
    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(train_dataloader),
        train_batch_size=config.train_dataset.batch_size,
    )
    logger.info(f"FinetuneSpec: {ft_spec}")
    # Initialize scheduler
    scheduler = LocalScheduler(config)
    # Initialize train controller
    train_controller = DistributedTrainController(engine, config.model, scheduler)
    train_controller.initialize(
        config.allocation_mode,
        ft_spec,
        ScheduleStrategy(),
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

            with stats_tracker.record_timing("to_device"):
                data = tensor_container_to(data, "cpu")
                data = DistributedBatchMemory.from_dict(data)

            with (
                stats_tracker.record_timing("train_step"),
                stats_tracker.scope("sft"),
            ):
                stat = train_controller.train_lm(data)
                train_controller.step_lr_scheduler()
                logger.info(f"train stat: {stat}")

            with stats_tracker.record_timing("save"):
                saver.save(
                    train_controller, epoch, step, global_step, tokenizer=tokenizer
                )

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
                    with stats_tracker.scope("sft-eval"):
                        for data in valid_dataloader:
                            data = tensor_container_to(data, "cpu")
                            data = DistributedBatchMemory.from_dict(data)
                            train_controller.evaluate_lm(data)

                evaluator.evaluate(
                    evaluate_fn,
                    epoch,
                    step,
                    global_step,
                )

            stats = list()
            # todo: gather stats from all ranks
            stats.append(stat)
            stats.append(stats_tracker.export_all())
            stats_logger.commit(
                epoch,
                step,
                global_step,
                stats,
            )
            global_step += 1

    stats_logger.close()
    train_controller.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])
