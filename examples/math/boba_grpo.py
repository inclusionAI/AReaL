import os
import sys

import torch.distributed as dist
from datasets import load_dataset

from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.api.io_struct import AllocationMode, FinetuneSpec, StepInfo, WeightUpdateMeta
from areal.engine.ppo.actor import FSDPPPOActor
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.engine.vllm_remote import RemotevLLMEngine
from areal.platforms import current_platform
from areal.utils import logging, perf_tracer, seeding, stats_tracker
from areal.utils.dataloader import create_dataloader
from areal.utils.device import log_gpu_stats
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.perf_tracer import Category
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger
from areal.workflow.rlvr import RLVRWorkflow

logger = logging.getLogger("boba_grpo")


def get_input_ids_fn(data, tokenizer, enable_thinking):
    user_token = "<｜User｜>"
    assistant_token = "<｜Assistant｜>"
    think_token = "<think>"
    has_think_token = think_token in data
    data = (
        data.replace(user_token, "")
        .replace(assistant_token, "")
        .replace(think_token, "")
    )
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": data}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking or has_think_token,
    )
    return input_ids


def data_extract_prompt_fn(data):
    return data["prompt"]


def get_boba_math_dataset(path, tokenizer):
    dataset = load_dataset(
        path="json",
        split="train",
        data_files=path,
    )
    dataset = dataset.filter(lambda x: len(tokenizer.encode(x["prompt"])) <= 1024)
    return dataset


def boba_reward_fn(
    prompts, completions, prompt_ids, completion_ids, solutions, **kwargs
):
    from areal.reward.math_parser import process_results

    label = 0
    for sol in solutions:
        x = process_results(completions, sol)
        label = label or x[0]
    return label


def main(args):
    config, _ = load_expr_config(args, GRPOConfig)
    rank = int(os.getenv("RANK"))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    seeding.set_random_seed(config.seed, key=f"trainer{rank}")
    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    parallel_strategy = allocation_mode.train
    assert parallel_strategy is not None

    # Initialize train engine
    actor = FSDPPPOActor(config=config.actor)
    actor.create_process_group(parallel_strategy=parallel_strategy)

    # Configure performance tracer
    if config.perf_tracer is not None:
        perf_tracer.configure(config.perf_tracer, rank=rank)

    world_size = actor.data_parallel_world_size
    if config.train_dataset.batch_size < world_size:
        raise ValueError(
            f"batch size({config.train_dataset.batch_size}) "
            f"must larger or equal than world_size({world_size})!"
        )

    # Create dataset and dataloaders
    train_dataset = get_boba_math_dataset(config.train_dataset.path, tokenizer)
    train_dataloader = create_dataloader(
        train_dataset,
        rank=actor.data_parallel_rank,
        world_size=world_size,
        dataset_config=config.train_dataset,
    )

    train_dataset_len = len(train_dataloader)
    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=train_dataset_len * config.train_dataset.batch_size,
        train_batch_size=config.train_dataset.batch_size,
    )

    # Initialize inference engine
    if allocation_mode.gen_backend == "vllm":
        rollout = RemotevLLMEngine(config.rollout)
    elif allocation_mode.gen_backend == "sglang":
        rollout = RemoteSGLangEngine(config.rollout)
    rollout.initialize(train_data_parallel_size=parallel_strategy.dp_size)

    weight_update_meta = WeightUpdateMeta.from_fsdp_xccl(allocation_mode)

    # Initialize train engine
    actor.initialize(None, ft_spec)
    actor.connect_engine(rollout, weight_update_meta)

    ref = None
    if config.actor.kl_ctl > 0 and config.ref is not None:
        ref = FSDPPPOActor(config=config.ref)
        ref.initialize(None, ft_spec)

    # Create rollout workflow
    workflow = RLVRWorkflow(
        reward_fn=boba_reward_fn,
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        enable_thinking=True,
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated"
        ),
        get_input_ids_fn=get_input_ids_fn,
        data_extract_prompt_fn=data_extract_prompt_fn,
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

    stop_step = config.total_train_steps
    total_epochs = config.total_train_epochs
    steps_per_epoch = train_dataset_len
    max_steps = total_epochs * steps_per_epoch

    for global_step in range(start_step, max_steps):
        if stop_step and global_step >= stop_step:
            logger.info("Training stopped at step %d", global_step)
            exit()

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
                args={"global_step": global_step},
            ),
        ):
            batch = actor.prepare_batch(
                train_dataloader,
                granularity=actor.config.group_size,
                workflow=workflow,
                should_accept_fn=lambda sample: True,
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
            perf_tracer.trace_scope(
                "train.ppo_update",
                category=Category.COMPUTE,
                args={"global_step": global_step},
            ),
        ):
            actor.ppo_update(batch)
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

        current_platform.synchronize()
        dist.barrier(group=actor.cpu_group)

        # Upload statistics to the logger (e.g., wandb)
        with perf_tracer.trace_scope(
            "train.log_stats",
            category=Category.INSTR,
            args={"global_step": global_step},
        ):
            stats = stats_tracker.export_all(reduce_group=actor.data_parallel_group)
            stats_logger.commit(epoch, step, global_step, stats)

        current_platform.synchronize()
        dist.barrier(group=actor.cpu_group)

        # Resume rollout
        rollout.resume()

        perf_tracer.save(step=global_step)

    stats_logger.close()
    rollout.destroy()
    if ref is not None:
        ref.destroy()
    actor.destroy()
    perf_tracer.save(force=True)


if __name__ == "__main__":
    main(sys.argv[1:])
