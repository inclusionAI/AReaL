import os
import sys
import uuid

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
print(sys.path)

import copy
import sys
from dataclasses import dataclass, field

import torch.distributed as dist
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from loguru import logger
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.utils.logging import logging
from areal.utils.redistributor import redistribute

logger.remove()

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.api.io_struct import FinetuneSpec, StepInfo, WeightUpdateMeta
from areal.engine.ppo.actor import FSDPPPOActor
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.platforms import current_platform
from areal.utils import seeding, stats_tracker
from areal.utils.data import (
    broadcast_tensor_container,
    cycle_dataloader,
)
from areal.utils.device import log_gpu_stats
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger

from .workflow_tau2 import Tau2Workflow

worker_id = uuid.uuid4().hex[:4]
logger = logging.getLogger(f"Tau @ {worker_id}")


@dataclass
class TauRLConfig(GRPOConfig):
    max_turns: int = field(
        default=32, metadata={"help": "maximum number of turns for search agent"}
    )
    n_trajs: int = field(
        default=1,
        metadata={
            "help": "We could collect multiple trajectories for a single query. By default n_trajs=1."
        },
    )
    user_model: str = field(
        default="/storage/openpsi/models/Qwen__Qwen2.5-72B-Instruct/",
        metadata={"help": "Model name for user simulation."},
    )

    user_api_key: str = field(
        default="empty", metadata={"help": "api_key for user simulation model"}
    )

    user_base_url: str = field(
        default="http://33.180.160.63:30000/v1/",
        metadata={"help": "base_url for user simulation model"},
    )

    max_context_length: int = field(
        default=32768, metadata={"help": "Maximum context length of the trained model"}
    )

    reward_type: str = field(
        default="all",
        metadata={"help": "Reward type for training or evaluation. Options: db, all"},
    )
    dynamic_filtering: bool = field(
        default=False, metadata={"help": "Whether to filter out the tasks dynamicly"}
    )


def get_tau2_dataset(dataset_path, tokenizer, rank, world_size):
    dataset = load_dataset(
        path="json",
        split="train",
        data_files=dataset_path,
    )
    # dataset = dataset.filter(lambda x: len(tokenizer.encode(x["question"])) <= 1024)
    return split_dataset_by_node(dataset, rank=rank, world_size=world_size)


def main(args):
    config, _ = load_expr_config(args, TauRLConfig)
    config: TauRLConfig

    rank = int(os.getenv("RANK"))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    seeding.set_random_seed(config.seed, key=f"trainer{rank}")
    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    parallel_strategy = allocation_mode.train
    assert parallel_strategy is not None

    # Initialize train engine
    actor = FSDPPPOActor(config=config.actor)
    actor.create_process_group(parallel_strategy=parallel_strategy)

    train_dataset = get_tau2_dataset(
        config.train_dataset.path,
        tokenizer,
        actor.data_parallel_rank,
        actor.data_parallel_world_size,
    )
    valid_dataset = (
        get_tau2_dataset(
            config.valid_dataset.path,
            tokenizer,
            actor.data_parallel_rank,
            actor.data_parallel_world_size,
        ),
    )

    # Create dataset and dataloaders
    train_dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=config.train_dataset.batch_size // actor.data_parallel_world_size,
        shuffle=config.train_dataset.shuffle,
        num_workers=config.train_dataset.num_workers,
        collate_fn=lambda x: x,
        drop_last=config.train_dataset.drop_last,
    )

    valid_dataloader = StatefulDataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=config.valid_dataset.shuffle,
        num_workers=config.valid_dataset.num_workers,
        collate_fn=lambda x: x,
        drop_last=config.valid_dataset.drop_last,
    )
    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(train_dataloader) * config.train_dataset.batch_size,
        train_batch_size=config.train_dataset.batch_size,
    )

    # Initialize inference engine
    rollout = RemoteSGLangEngine(config.rollout)
    rollout.initialize(train_data_parallel_size=parallel_strategy.dp_size)
    eval_rollout = RemoteSGLangEngine(copy.deepcopy(config.rollout))
    # NOTE: eval does not have any offpolicyness control
    eval_rollout.config.max_head_offpolicyness = int(1e12)
    eval_rollout.initialize()

    actor.initialize(None, ft_spec)
    ref = None

    # NOTE: Weight update meta only requires address and free port of rank 0,
    # but `WeightUpdateMeta.from_fsdp_xccl` has to be executed on all ranks
    # due to `engine.get_param_specs()`.
    # Therefore, we create weight update meta on all ranks, then broadcast the one on rank 0.

    # weight_update_meta = [
    #     WeightUpdateMeta.from_fsdp_xccl(
    #         AllocationMode.from_str(config.allocation_mode), actor
    #     )
    # ]
    # dist.broadcast_object_list(weight_update_meta, src=0)
    # weight_update_meta = weight_update_meta[0]

    weight_update_meta = WeightUpdateMeta.from_disk(
        config.experiment_name, config.trial_name, config.cluster.fileroot
    )

    # Create rollout workflow
    if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)
    workflow = Tau2Workflow(
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated"
        ),
        max_num_turns=config.max_turns,
        n_trajs=config.n_trajs,
        user_model=config.user_model,
        user_api_key=config.user_api_key,
        user_base_url=config.user_base_url,
        max_context_length=config.max_context_length,
        reward_type=config.reward_type,
        dynamic_filtering=config.dynamic_filtering,
    )

    eval_workflow = Tau2Workflow(
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated-eval"
        ),
        max_num_turns=config.max_turns,
        n_trajs=4,
        user_model=config.user_model,
        user_api_key=config.user_api_key,
        user_base_url=config.user_base_url,
        max_context_length=config.max_context_length,
        reward_type="db",
        dynamic_filtering=False,
    )

    # Run training.
    saver = Saver(config.saver, ft_spec)
    stats_logger = StatsLogger(config, ft_spec)
    evaluator = Evaluator(config.evaluator, ft_spec)

    recover_handler = RecoverHandler(config.recover, ft_spec)
    recover_info = recover_handler.load(
        actor,
        saver,
        evaluator,
        stats_logger,
        train_dataloader,
        inference_engine=rollout,
        weight_update_meta=weight_update_meta,
    )
    start_step = (
        recover_info.last_step_info.next().global_step
        if recover_info is not None
        else 0
    )

    total_epochs = config.total_train_epochs
    steps_per_epoch = len(train_dataloader)
    max_steps = total_epochs * steps_per_epoch

    data_generator = cycle_dataloader(train_dataloader)
    eval_rollout.set_version(start_step)
    for global_step in range(start_step, max_steps):
        epoch = global_step // steps_per_epoch
        step = global_step % steps_per_epoch
        step_info = StepInfo(
            global_step=global_step,
            epoch=epoch,
            epoch_step=step,
            steps_per_epoch=steps_per_epoch,
        )

        rollout.pause()
        dist.barrier(device_ids=[actor.device.index])
        with stats_tracker.record_timing("eval"):

            def evaluate_fn():
                # Stats are logged in the workflow
                # and will be exported later
                cnt = 0
                for data in valid_dataloader:
                    for item in data:
                        eval_rollout.submit(item, eval_workflow)
                        cnt += 1
                eval_rollout.wait(cnt, timeout=None)

            if actor.is_data_parallel_head():
                evaluator.evaluate(
                    evaluate_fn,
                    epoch,
                    step,
                    global_step,
                )
        dist.barrier(device_ids=[actor.device.index])
        rollout.resume()

        with stats_tracker.record_timing("rollout"):
            batch = None
            if actor.is_data_parallel_head():
                if config.async_training:
                    batch = rollout.prepare_batch(
                        train_dataloader,
                        workflow=workflow,
                        should_accept=lambda sample: True,
                    )
                else:
                    batch = rollout.rollout_batch(
                        next(data_generator),
                        workflow=workflow,
                        should_accept=lambda sample: True,
                    )
                batch = batch.to(actor.device)
                batch = redistribute(batch, group=actor.data_parallel_group).data
            batch = broadcast_tensor_container(
                batch,
                src_rank=actor.current_data_parallel_head(),
                group=actor.context_and_model_parallel_group,
            )
        # Create barrier to synchronize all rollout processes.
        dist.barrier(device_ids=[actor.device.index])
        current_platform.synchronize()

        if config.actor.recompute_logprob or config.actor.use_decoupled_loss:
            with stats_tracker.record_timing("recompute_logp"):
                logp = actor.compute_logp(batch)
                batch["prox_logp"] = logp
                log_gpu_stats("recompute logp")

        if ref is not None:
            with stats_tracker.record_timing("ref_logp"):
                batch["ref_logp"] = ref.compute_logp(batch)
                log_gpu_stats("ref logp")

        with stats_tracker.record_timing("compute_advantage"):
            actor.compute_advantages(batch)
            log_gpu_stats("compute advantages")

        with (
            stats_tracker.record_timing("train_step"),
            stats_tracker.scope("grpo_actor"),
        ):
            stats = actor.ppo_update(batch)
            actor.step_lr_scheduler()
            log_gpu_stats("ppo update")

        # pause inference for updating weights, save, and evaluation
        rollout.pause()

        with stats_tracker.record_timing("update_weights"):
            if dist.get_rank() == 0:
                future = rollout.update_weights(weight_update_meta)
            actor.upload_weights(weight_update_meta)
            if dist.get_rank() == 0:
                future.result()
            dist.barrier(device_ids=[actor.device.index])
            current_platform.synchronize()

            actor.set_version(global_step + 1)
            rollout.set_version(global_step + 1)
            eval_rollout.set_version(global_step + 1)

        with stats_tracker.record_timing("save"):
            saver.save(actor, epoch, step, global_step, tokenizer=tokenizer)

        with stats_tracker.record_timing("checkpoint_for_recover"):
            recover_handler.dump(
                actor,
                step_info,
                saver,
                evaluator,
                stats_logger,
                train_dataloader,
                tokenizer=tokenizer,
            )

        dist.barrier(device_ids=[actor.device.index])
        current_platform.synchronize()

        # Upload statistics to the logger (e.g., wandb)
        stats[0].update(
            stats_tracker.export_all(reduce_group=actor.data_parallel_group)
        )
        stats_logger.commit(epoch, step, global_step, stats)

        dist.barrier(device_ids=[actor.device.index])
        current_platform.synchronize()

        # Resume rollout
        rollout.resume()

    stats_logger.close()
    eval_rollout.destroy()
    rollout.destroy()
    if ref is not None:
        ref.destroy()
    actor.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])
