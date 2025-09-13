import sys

from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import load_expr_config
from areal.api.io_struct import FinetuneSpec, StepInfo
from areal.dataset import get_custom_dataset
from areal.engine.rw.rw_engine import FSDPRWEngine
from areal.experimental.api.cli_args import ExperimentalSFTConfig as SFTConfig
from areal.utils import stats_tracker
from areal.utils.data import pairwise_pad_sequences_to_tensors
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger


def main(args):
    config, _ = load_expr_config(args, SFTConfig)
    config: SFTConfig

    # rank = int(os.getenv("RANK"))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    # seeding.set_random_seed(config.seed, f"trainer{rank}")

    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    parallel_strategy = allocation_mode.train

    # Initialize Megatron parallel states
    # NOTE: we assume all MegatronEngine has the same parallel strategy.
    engine = FSDPRWEngine(config=config.model)
    engine.create_process_group(parallel_strategy=parallel_strategy)

    train_dataset = get_custom_dataset(
        path=config.train_dataset.path,
        rank=0,
        world_size=1,
        split="train",
        type=config.train_dataset.type,
        tokenizer=tokenizer,
    )
    valid_dataset = get_custom_dataset(
        path=config.valid_dataset.path,
        rank=0,
        world_size=1,
        split="test",
        type=config.valid_dataset.type,
        tokenizer=tokenizer,
    )

    # Create dataset and dataloaders
    train_dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=config.train_dataset.batch_size,
        shuffle=config.train_dataset.shuffle,
        num_workers=config.train_dataset.num_workers,
        collate_fn=pairwise_pad_sequences_to_tensors,
        drop_last=config.train_dataset.drop_last,
    )
    valid_dataloader = StatefulDataLoader(
        valid_dataset,
        batch_size=config.valid_dataset.batch_size,
        shuffle=config.valid_dataset.shuffle,
        num_workers=config.valid_dataset.num_workers,
        collate_fn=pairwise_pad_sequences_to_tensors,
        drop_last=config.valid_dataset.drop_last,
    )

    # Initialize engine
    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(train_dataloader) * config.train_dataset.batch_size,
        train_batch_size=config.train_dataset.batch_size,
    )
    engine.initialize(None, ft_spec)

    # Run training.
    saver = Saver(config.saver, ft_spec)
    stats_logger = StatsLogger(config.stats_logger, ft_spec)
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
                stats_tracker.scope("rw"),
            ):
                stats = engine.train_rw(data)
                engine.step_lr_scheduler()

            with stats_tracker.record_timing("eval"):

                def evaluate_fn():
                    with stats_tracker.scope("rw-eval"):
                        for data in valid_dataloader:
                            engine.evaluate_rw(data)

                evaluator.evaluate(
                    evaluate_fn,
                    epoch,
                    step,
                    global_step,
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

            stats.update(stats_tracker.export_all())
            stats_logger.commit(epoch, step, global_step, stats)
            global_step += 1

    stats_logger.close()
    engine.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])
