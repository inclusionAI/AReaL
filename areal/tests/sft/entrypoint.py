import json
import os
import sys
from typing import List, cast

import torch.utils.data
from torchdata.stateful_dataloader import StatefulDataLoader

import areal.api.cli_args as cli_args
import areal.dataset
import areal.utils.data
import realhf.api.core.data_api as data_api
import realhf.base.seeding as seeding
import realhf.base.stats_tracker as stats_tracker
from areal.api.cli_args import SFTConfig
from areal.api.io_struct import FinetuneSpec
from areal.engine.sft.lm_engine import FSDPLMEngine
from areal.utils.stats_logger import StatsLogger


def main() -> None:
    config, _ = cli_args.load_expr_config(sys.argv[1:], SFTConfig)
    assert isinstance(config, SFTConfig)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))

    seeding.set_random_seed(config.seed, str(rank))

    processor, tokenizer = data_api.load_hf_processor_and_tokenizer(
        config.tokenizer_path
    )

    train_dataset = areal.dataset.get_custom_dataset(
        path=config.train_dataset.path,
        rank=rank,
        world_size=world_size,
        type="sft",
        split="train",
        tokenizer=tokenizer,
        processor=processor,
    )
    train_dataset = train_dataset.select(range(256))

    train_dataloader = StatefulDataLoader(
        cast(torch.utils.data.Dataset, train_dataset),
        batch_size=config.train_dataset.batch_size // world_size,
        collate_fn=areal.utils.data.pad_sequences_to_tensors,
    )
    assert train_dataloader.batch_size is not None

    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(train_dataloader) * train_dataloader.batch_size,
        train_batch_size=train_dataloader.batch_size,
    )
    engine = FSDPLMEngine(config=config.model)
    engine.initialize(
        addr=None,
        ft_spec=ft_spec,
    )

    losses: List[float] = []

    global_step = 0
    for epoch in range(config.total_train_epochs):
        for step, data in enumerate(train_dataloader):
            engine.train_lm(data)
            engine.step_lr_scheduler()

            stat = stats_tracker.export(reduce_group=engine.parallelism_group)
            losses.append(stat["loss/avg"])

            global_step += 1

    with open(os.path.join(config.cluster.fileroot, "losses.json"), "w") as f:
        json.dump(losses, f)

    engine.destroy()


if __name__ == "__main__":
    main()
