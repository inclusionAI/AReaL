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
from areal.platforms import current_platform
from areal.reward.math_parser import process_results
from areal.utils import seeding
from areal.utils.data import broadcast_tensor_container, tensor_container_to
from areal.utils.hf_utils import load_hf_processor_and_tokenizer
from areal.utils.stats_logger import StatsLogger
from areal.workflow.rlvr import RLVRWorkflow


def gsm8k_reward_fn(prompt, completions, prompt_ids, completion_ids, answer, **kwargs):
    return int(process_results(completions, answer)[0])


def main() -> None:
    config, _ = cli_args.load_expr_config(sys.argv[1:], GRPOConfig)
    assert isinstance(config, GRPOConfig)

    rank = int(os.environ.get("RANK", "0"))

    seeding.set_random_seed(config.seed, str(rank))
    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    parallel_strategy = allocation_mode.train
    assert parallel_strategy is not None

    actor = FSDPPPOActor(config=config.actor)
    actor.create_process_group(parallel_strategy=parallel_strategy)

    processor, tokenizer = load_hf_processor_and_tokenizer(config.tokenizer_path)

    train_dataset = areal.dataset.get_custom_dataset(
        path=config.train_dataset.path,
        rank=actor.data_parallel_rank,
        world_size=actor.data_parallel_world_size,
        type="rl",
        split="train",
        tokenizer=tokenizer,
        processor=processor,
    )

    train_dataloader = StatefulDataLoader(
        cast(torch.utils.data.Dataset, train_dataset),
        batch_size=config.train_dataset.batch_size // actor.data_parallel_world_size,
        collate_fn=lambda x: x,
    )
    assert train_dataloader.batch_size is not None

    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(train_dataloader) * train_dataloader.batch_size,
        train_batch_size=train_dataloader.batch_size,
    )

    rollout = RemoteSGLangEngine(config.rollout)
    rollout.initialize(train_data_parallel_size=parallel_strategy.dp_size)

    weight_update_meta = WeightUpdateMeta.from_fsdp_xccl(allocation_mode)

    actor.initialize(None, ft_spec)
    actor.connect_engine(rollout, weight_update_meta)

    ref = FSDPPPOActor(config=config.ref)
    ref.create_process_group(parallel_strategy=parallel_strategy)
    ref.initialize(None, ft_spec)

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

            batch = None
            if actor.is_data_parallel_head():
                batch = rollout.prepare_batch(train_dataloader, workflow=workflow)
                batch = tensor_container_to(batch, actor.device)
            batch = broadcast_tensor_container(
                batch,
                src_rank=actor.current_data_parallel_head(),
                group=actor.context_and_model_parallel_group,
            )

            dist.barrier(device_ids=[actor.device.index])
            current_platform.synchronize()

            batch["ref_logp"] = ref.compute_logp(batch)

            actor.compute_advantages(batch)

            stats = actor.ppo_update(batch)
            actor.step_lr_scheduler()

            rollout.pause()
            actor.update_weights(weight_update_meta)
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
