import os
import sys
from copy import deepcopy

import torch.distributed as dist

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.api.io_struct import FinetuneSpec, StepInfo, WeightUpdateMeta
from areal.dataset import get_custom_dataset
from areal.engine.ppo.actor import FSDPPPOActor
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.platforms import current_platform
from areal.utils import perf_tracer, seeding, stats_tracker
from areal.utils.data import (
    cycle_dataloader,
)
from areal.utils.dataloader import create_dataloader
from areal.utils.device import log_gpu_stats
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.perf_tracer import Category
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger
from areal.workflow.rlvr import RLVRWorkflow


def gsm8k_reward_fn(prompt, completions, prompt_ids, completion_ids, answer, **kwargs):
    from areal.reward.math_parser import process_results

    return int(process_results(completions, answer)[0])


def main(args):
    config, _ = load_expr_config(args, GRPOConfig)
    config: GRPOConfig

    rank = int(os.getenv("RANK"))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    seeding.set_random_seed(config.seed, key=f"trainer{rank}")
    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    parallel_strategy = allocation_mode.train
    assert parallel_strategy is not None

    # Initialize train engine
    actor = FSDPPPOActor(config=config.actor)
    actor.create_process_group(parallel_strategy=parallel_strategy)

    # Create dataset and dataloaders
    train_dataset = get_custom_dataset(
        split="train", dataset_config=config.train_dataset, tokenizer=tokenizer
    )
    valid_dataset = get_custom_dataset(
        split="test", dataset_config=config.valid_dataset, tokenizer=tokenizer
    )

    train_dataloader = create_dataloader(
        train_dataset,
        rank=actor.data_parallel_rank,
        world_size=actor.data_parallel_world_size,
        dataset_config=config.train_dataset,
    )
    valid_dataloader = create_dataloader(
        valid_dataset,
        rank=actor.data_parallel_rank,
        world_size=actor.data_parallel_world_size,
        dataset_config=config.valid_dataset,
    )
    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(train_dataloader) * config.train_dataset.batch_size,
        train_batch_size=config.train_dataset.batch_size,
    )

    # Initialize inference engine
    rollout = RemoteSGLangEngine(config.rollout)
    rollout.initialize(train_data_parallel_size=parallel_strategy.dp_size)
    eval_rollout = RemoteSGLangEngine(deepcopy(config.rollout))
    # NOTE: eval does not have any offpolicyness control
    eval_rollout.config.max_head_offpolicyness = int(1e12)
    eval_rollout.initialize()

    weight_update_meta = WeightUpdateMeta.from_fsdp_xccl(allocation_mode)

    actor.initialize(None, ft_spec)
    actor.connect_engine(rollout, weight_update_meta)

    ref = None
    if config.actor.kl_ctl > 0 and config.ref is not None:
        ref = FSDPPPOActor(config=config.ref)
        ref.create_process_group(parallel_strategy=parallel_strategy)
        ref.initialize(None, ft_spec)

    # Create rollout workflow
    if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)
    workflow = RLVRWorkflow(
        reward_fn=gsm8k_reward_fn,
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        enable_thinking=False,
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated"
        ),
    )
    eval_workflow = RLVRWorkflow(
        reward_fn=gsm8k_reward_fn,
        gconfig=config.gconfig.new(temperature=0.6),
        tokenizer=tokenizer,
        enable_thinking=False,
        rollout_stat_scope="eval-rollout",
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated-eval"
        ),
    )

    # Configure performance tracer
    if config.perf_tracer is None:
        raise ValueError("Performance tracer config must be provided.")
    perf_tracer.configure(config.perf_tracer, rank=rank)

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
    for global_step in range(start_step, max_steps):
        epoch = global_step // steps_per_epoch
        step = global_step % steps_per_epoch
        step_info = StepInfo(
            global_step=global_step,
            epoch=epoch,
            epoch_step=step,
            steps_per_epoch=steps_per_epoch,
        )

        with (
            stats_tracker.record_timing("rollout"),
            perf_tracer.trace_scope(
                "train.rollout",
                category=Category.COMPUTE,
                args={
                    "global_step": global_step,
                    "epoch_step": step,
                },
            ),
        ):
            if config.async_training:
                batch = actor.prepare_batch(
                    train_dataloader,
                    granularity=actor.config.group_size,
                    workflow=workflow,
                    should_accept=lambda sample: True,
                )
            else:
                batch = actor.rollout_batch(
                    next(data_generator),
                    granularity=actor.config.group_size,
                    workflow=workflow,
                    should_accept=lambda sample: True,
                )

        if config.actor.recompute_logprob or config.actor.use_decoupled_loss:
            with (
                stats_tracker.record_timing("recompute_logp"),
                perf_tracer.trace_scope(
                    "train.recompute_logp",
                    category=Category.COMPUTE,
                    args={"global_step": global_step},
                ),
            ):
                logp = actor.compute_logp(batch)
                batch["prox_logp"] = logp
                log_gpu_stats("recompute logp")

        if ref is not None:
            with (
                stats_tracker.record_timing("ref_logp"),
                perf_tracer.trace_scope(
                    "train.ref_logp",
                    category=Category.COMPUTE,
                    args={"global_step": global_step},
                ),
            ):
                batch["ref_logp"] = ref.compute_logp(batch)
                log_gpu_stats("ref logp")

        with (
            stats_tracker.record_timing("compute_advantage"),
            perf_tracer.trace_scope(
                "train.compute_advantage",
                category=Category.COMPUTE,
                args={"global_step": global_step},
            ),
        ):
            actor.compute_advantages(batch)
            log_gpu_stats("compute advantages")

        with (
            stats_tracker.record_timing("train_step"),
            stats_tracker.scope("grpo_actor"),
            perf_tracer.trace_scope(
                "train.ppo_update",
                category=Category.COMPUTE,
                args={"global_step": global_step},
            ),
        ):
            stats = actor.ppo_update(batch)
            actor.step_lr_scheduler()
            log_gpu_stats("ppo update")

        # pause inference for updating weights, save, and evaluation
        rollout.pause()

        with (
            stats_tracker.record_timing("update_weights"),
            perf_tracer.trace_scope(
                "train.update_weights",
                category=Category.COMM,
                args={"global_step": global_step},
            ),
        ):
            actor.update_weights(weight_update_meta)

            actor.set_version(global_step + 1)
            rollout.set_version(global_step + 1)
            eval_rollout.set_version(global_step + 1)

        with (
            stats_tracker.record_timing("save"),
            perf_tracer.trace_scope(
                "train.save",
                category=Category.IO,
                args={"global_step": global_step},
            ),
        ):
            saver.save(actor, epoch, step, global_step, tokenizer=tokenizer)

        with (
            stats_tracker.record_timing("checkpoint_for_recover"),
            perf_tracer.trace_scope(
                "train.checkpoint",
                category=Category.IO,
                args={"global_step": global_step},
            ),
        ):
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

        with (
            stats_tracker.record_timing("eval"),
            perf_tracer.trace_scope(
                "train.eval",
                category=Category.COMPUTE,
                args={"global_step": global_step},
            ),
        ):

            def evaluate_fn():
                if actor.is_data_parallel_head():
                    cnt = 0
                    for data in valid_dataloader:
                        for item in data:
                            eval_rollout.submit(item, eval_workflow)
                            cnt += 1
                    eval_rollout.wait(cnt, timeout=None)
                dist.barrier(device_ids=[actor.device.index])
                current_platform.synchronize()

            evaluator.evaluate(
                evaluate_fn,
                epoch,
                step,
                global_step,
            )

        dist.barrier(device_ids=[actor.device.index])
        current_platform.synchronize()

        # Upload statistics to the logger (e.g., wandb)
        with perf_tracer.trace_scope(
            "train.log_stats",
            category=Category.INSTR,
            args={"global_step": global_step},
        ):
            stats[0].update(
                stats_tracker.export_all(reduce_group=actor.data_parallel_group)
            )
            stats_logger.commit(epoch, step, global_step, stats)

        dist.barrier(device_ids=[actor.device.index])
        current_platform.synchronize()

        # Resume rollout
        rollout.resume()

        perf_tracer.save(step=global_step)

    stats_logger.close()
    eval_rollout.destroy()
    rollout.destroy()
    if ref is not None:
        ref.destroy()
    actor.destroy()
    perf_tracer.save(force=True)


if __name__ == "__main__":
    main(sys.argv[1:])
