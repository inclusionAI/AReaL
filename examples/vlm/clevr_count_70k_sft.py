import os
import sys

import torch.distributed as dist
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import SFTConfig, load_expr_config
from areal.api.io_struct import FinetuneSpec, StepInfo
from areal.dataset import get_custom_dataset
from areal.engine.sft.lm_engine import FSDPLMEngine
from areal.platforms import current_platform
from areal.utils import seeding, stats_tracker
from areal.utils.data import (
    broadcast_tensor_container,
    pad_sequences_to_tensors,
    tensor_container_to,
)
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_processor_and_tokenizer
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger


def main_sft():
    config, _ = load_expr_config(sys.argv[1:], SFTConfig)
    config: SFTConfig

    rank = int(os.getenv("RANK"))

    seeding.set_random_seed(config.seed, f"trainer{rank}")
    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    parallel_strategy = allocation_mode.train

    engine = FSDPLMEngine(config=config.model)
    engine.create_process_group(parallel_strategy=parallel_strategy)

    processor, tokenizer = load_hf_processor_and_tokenizer(config.tokenizer_path)
    train_dataset = get_custom_dataset(
        path=config.train_dataset.path,
        rank=engine.data_parallel_rank,
        world_size=engine.data_parallel_world_size,
        split="train",
        max_length=config.train_dataset.max_length,
        type=config.train_dataset.type,
        tokenizer=tokenizer,
        processor=processor,
    )
    valid_dataset = get_custom_dataset(
        path=config.valid_dataset.path,
        rank=engine.data_parallel_rank,
        world_size=engine.data_parallel_world_size,
        split="test",
        max_length=config.valid_dataset.max_length,
        type=config.valid_dataset.type,
        tokenizer=tokenizer,
        processor=processor,
    )

    # Create dataset and dataloaders
    train_dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=config.train_dataset.batch_size // engine.data_parallel_world_size,
        shuffle=config.train_dataset.shuffle,
        num_workers=config.train_dataset.num_workers,
        collate_fn=pad_sequences_to_tensors,
        drop_last=config.train_dataset.drop_last,
    )

    valid_dataloader = StatefulDataLoader(
        valid_dataset,
        batch_size=config.valid_dataset.batch_size // engine.data_parallel_world_size,
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
    engine.initialize(None, ft_spec)

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
                # NOTE: data are identical across model+context parallel group
                data = tensor_container_to(data, current_platform.current_device())

            with stats_tracker.record_timing("bcast"):
                data = broadcast_tensor_container(
                    data,
                    src_rank=engine.current_data_parallel_head(),
                    group=engine.context_and_model_parallel_group,
                )

            with (
                stats_tracker.record_timing("train_step"),
                stats_tracker.scope("sft"),
            ):
                stats = engine.train_lm(data)
                engine.step_lr_scheduler()
                stats_tracker.scalar(**stats)

            with stats_tracker.record_timing("save"):
                saver.save(
                    engine,
                    epoch,
                    step,
                    global_step,
                    tokenizer=tokenizer,
                    processor=processor,
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
                    processor=processor,
                )

            dist.barrier(device_ids=[engine.device.index])
            current_platform.synchronize()

            with stats_tracker.record_timing("eval"):
                # No need to log anything. Logging will be handled outside
                # via stats_tracker.export().
                def evaluate_fn():
                    with stats_tracker.scope("sft-eval"):
                        for data in valid_dataloader:
                            data = tensor_container_to(
                                data, current_platform.current_device()
                            )
                            data = broadcast_tensor_container(
                                data,
                                src_rank=engine.current_data_parallel_head(),
                                group=engine.context_and_model_parallel_group,
                            )
                            engine.evaluate_lm(data)

                evaluator.evaluate(
                    evaluate_fn,
                    epoch,
                    step,
                    global_step,
                )

            dist.barrier(device_ids=[engine.device.index])
            current_platform.synchronize()

            stats_logger.commit(
                epoch,
                step,
                global_step,
                stats_tracker.export(reduce_group=engine.data_parallel_group),
            )
            global_step += 1

    stats_logger.close()
    engine.destroy()


if __name__ == "__main__":
    main_sft()
