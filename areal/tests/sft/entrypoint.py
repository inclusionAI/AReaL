import json
import os
import sys
from typing import List, cast

import torch
import torch.distributed as dist
import torch.utils.data
from torchdata.stateful_dataloader import StatefulDataLoader

import areal.api.cli_args as cli_args
import areal.dataset
import areal.utils.data
import areal.utils.seeding as seeding
import areal.utils.stats_tracker as stats_tracker
from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import SFTConfig
from areal.api.io_struct import FinetuneSpec
from areal.engine.sft.lm_engine import FSDPLMEngine
from areal.platforms import current_platform
from areal.utils.data import broadcast_tensor_container, tensor_container_to
from areal.utils.hf_utils import load_hf_processor_and_tokenizer


def main() -> None:
    config, _ = cli_args.load_expr_config(sys.argv[1:], SFTConfig)
    assert isinstance(config, SFTConfig)

    rank = int(os.environ.get("RANK", "0"))

    seeding.set_random_seed(config.seed, str(rank))
    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    parallel_strategy = allocation_mode.train

    engine = FSDPLMEngine(config=config.model)
    engine.create_process_group(parallel_strategy=parallel_strategy)

    processor, tokenizer = load_hf_processor_and_tokenizer(config.tokenizer_path)

    train_dataset = areal.dataset.get_custom_dataset(
        path=config.train_dataset.path,
        rank=engine.data_parallel_rank,
        world_size=engine.data_parallel_world_size,
        type="sft",
        split="train",
        tokenizer=tokenizer,
        processor=processor,
    )

    train_dataloader = StatefulDataLoader(
        cast(torch.utils.data.Dataset, train_dataset),
        batch_size=config.train_dataset.batch_size // engine.data_parallel_world_size,
        collate_fn=areal.utils.data.pad_sequences_to_tensors,
    )
    assert train_dataloader.batch_size is not None

    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(train_dataloader) * train_dataloader.batch_size,
        train_batch_size=train_dataloader.batch_size,
    )
    engine.initialize(
        addr=None,
        ft_spec=ft_spec,
    )

    losses: List[float] = []

    global_step = 0
    for epoch in range(config.total_train_epochs):
        for step, data in enumerate(train_dataloader):
            if (
                config.total_train_steps is not None
                and global_step >= config.total_train_steps
            ):
                break

            data = tensor_container_to(data, current_platform.current_device())
            data = broadcast_tensor_container(
                data,
                src_rank=engine.current_data_parallel_head(),
                group=engine.context_and_model_parallel_group,
            )

            engine.train_lm(data)
            engine.step_lr_scheduler()

            stat = stats_tracker.export(reduce_group=engine.data_parallel_group)
            losses.append(stat["loss/avg"])

            global_step += 1

    if dist.get_rank() == 0:
        with open(os.path.join(config.cluster.fileroot, "losses.json"), "w") as f:
            json.dump(losses, f)

    engine.destroy()


if __name__ == "__main__":
    main()
