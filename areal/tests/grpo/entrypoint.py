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
import areal.utils.seeding as seeding
from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import GRPOConfig
from areal.api.io_struct import FinetuneSpec, WeightUpdateMeta
from areal.engine.ppo.actor import FSDPPPOActor
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.reward.math_parser import process_results
from areal.utils import seeding
from areal.utils.hf_utils import load_hf_processor_and_tokenizer
from areal.utils.stats_logger import StatsLogger
from areal.workflow.rlvr import RLVRWorkflow


def gsm8k_reward_fn(prompt, completions, prompt_ids, completion_ids, answer, **kwargs):
    return int(process_results(completions, answer)[0])


def main() -> None:
    config, _ = cli_args.load_expr_config(sys.argv[1:], GRPOConfig)
    assert isinstance(config, GRPOConfig)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))

    seeding.set_random_seed(config.seed, str(rank))

    processor, tokenizer = load_hf_processor_and_tokenizer(config.tokenizer_path)

    train_dataset = areal.dataset.get_custom_dataset(
        path=config.train_dataset.path,
        rank=rank,
        world_size=world_size,
        type="rl",
        split="train",
        tokenizer=tokenizer,
        processor=processor,
    )

    train_dataloader = StatefulDataLoader(
        cast(torch.utils.data.Dataset, train_dataset),
        batch_size=config.train_dataset.batch_size // world_size,
        collate_fn=lambda x: x,
    )
    assert train_dataloader.batch_size is not None

    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(train_dataloader) * train_dataloader.batch_size,
        train_batch_size=train_dataloader.batch_size,
    )

    rollout = RemoteSGLangEngine(config.rollout)
    rollout.initialize(None, ft_spec)

    actor = FSDPPPOActor(config=config.actor)
    actor.initialize(None, ft_spec)

    ref = FSDPPPOActor(config=config.ref)
    ref.initialize(None, ft_spec)

    weight_update_meta = [
        WeightUpdateMeta.from_fsdp_nccl(
            AllocationMode.from_str(config.allocation_mode), actor
        )
    ]
    dist.broadcast_object_list(weight_update_meta, src=0)
    weight_update_meta = weight_update_meta[0]

    if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(cast(int, tokenizer.pad_token_id))
    if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(cast(int, tokenizer.eos_token_id))
    workflow = RLVRWorkflow(
        reward_fn=gsm8k_reward_fn,
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        enable_thinking=False,
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated"
        ),
    )

    rewards: List[float] = []

    global_step = 0
    for epoch in range(config.total_train_epochs):
        for step in range(len(train_dataloader)):
            if (
                config.total_train_steps is not None
                and global_step >= config.total_train_steps
            ):
                break

            batch = rollout.prepare_batch(train_dataloader, workflow=workflow)
            batch = batch.to(actor.device)

            dist.barrier(device_ids=[actor.device.index])
            torch.cuda.synchronize()

            batch["ref_logp"] = ref.compute_logp(batch)

            actor.compute_advantages(batch)

            stats = actor.ppo_update(batch)
            actor.step_lr_scheduler()

            rollout.pause()
            if dist.get_rank() == 0:
                future = rollout.update_weights(weight_update_meta)
            else:
                future = None
            actor.upload_weights(weight_update_meta)
            if future is not None:
                future.result()
            dist.barrier(device_ids=[actor.device.index])
            torch.cuda.synchronize()
            rollout.resume()
            actor.set_version(global_step + 1)
            rollout.set_version(global_step + 1)

            rewards.append(stats[0]["task_reward/avg"])

            global_step += 1

    if dist.get_rank() == 0:
        with open(os.path.join(config.cluster.fileroot, "rewards.json"), "w") as f:
            json.dump(rewards, f)

    rollout.destroy()
    ref.destroy()
    actor.destroy()


if __name__ == "__main__":
    main()
