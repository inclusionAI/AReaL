import os
import sys
from concurrent.futures import ThreadPoolExecutor

from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.api.io_struct import FinetuneSpec, StepInfo, WeightUpdateMeta
from areal.api.scheduler_api import ScheduleStrategy
from areal.controller.batch import DistributedBatchMemory
from areal.controller.rollout_controller import DistributedRolloutController
from areal.controller.train_controller import DistributedTrainController
from areal.dataset import get_custom_dataset
from areal.engine.ppo.actor import FSDPPPOActor
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.reward.gsm8k_reward import gsm8k_reward_fn
from areal.scheduler.local import LocalScheduler
from areal.utils import logging, stats_tracker
from areal.utils.data import (
    cycle_dataloader,
)
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.http import wait_future_ordered
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger
from areal.workflow.rlvr import RLVRWorkflow

logger = logging.getLogger("Trainer")


def main(args):
    config, _ = load_expr_config(args, GRPOConfig)
    config: GRPOConfig

    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    parallel_strategy = allocation_mode.train
    assert parallel_strategy is not None

    train_dataset = get_custom_dataset(
        path=config.train_dataset.path,
        rank=0,
        world_size=1,
        split="train",
        max_length=config.train_dataset.max_length,
        type=config.train_dataset.type,
        tokenizer=tokenizer,
    )
    # valid_dataset = get_custom_dataset(
    #     path=config.valid_dataset.path,
    #     rank=actor.data_parallel_rank,
    #     world_size=actor.data_parallel_world_size,
    #     split="test",
    #     max_length=config.valid_dataset.max_length,
    #     type=config.valid_dataset.type,
    #     tokenizer=tokenizer,
    # )

    # Create dataset and dataloaders
    train_dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=config.train_dataset.batch_size,
        shuffle=config.train_dataset.shuffle,
        num_workers=config.train_dataset.num_workers,
        collate_fn=lambda x: x,
        drop_last=config.train_dataset.drop_last,
    )

    # valid_dataloader = StatefulDataLoader(
    #     valid_dataset,
    #     batch_size=config.valid_dataset.batch_size // actor.data_parallel_world_size,
    #     shuffle=config.valid_dataset.shuffle,
    #     num_workers=config.valid_dataset.num_workers,
    #     collate_fn=lambda x: x,
    #     drop_last=config.valid_dataset.drop_last,
    # )
    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(train_dataloader),
        train_batch_size=config.train_dataset.batch_size,
    )

    # Initialize scheduler
    scheduler = LocalScheduler(config)
    # Initialize train engine
    train_engine = FSDPPPOActor(config=config.actor)
    actor = DistributedTrainController(train_engine, config.actor, scheduler)
    actor.initialize(
        config.allocation_mode,
        ft_spec,
        ScheduleStrategy(),
        group_size=config.gconfig.n_samples,
    )

    # Initialize inference engine
    inf_engine = RemoteSGLangEngine(config.rollout)
    rollout = DistributedRolloutController(inf_engine, config.rollout, scheduler)

    rollout.initialize(config, "")

    # eval_rollout = RemoteSGLangEngine(deepcopy(config.rollout))
    # # NOTE: eval does not have any offpolicyness control
    # eval_rollout.config.max_head_offpolicyness = int(1e12)
    # eval_rollout.initialize()
    #
    # actor.initialize(None, ft_spec)
    ref = None
    if config.actor.kl_ctl > 0 and config.ref is not None:
        ref_engine = FSDPPPOActor(config=config.ref)
        ref = DistributedTrainController(ref_engine, config.ref, scheduler)
        ref.initialize(
            config.allocation_mode,
            ft_spec,
            ScheduleStrategy(),
            group_size=config.gconfig.n_samples,
        )

    # NOTE: Weight update meta only requires address and free port of rank 0,
    # but `WeightUpdateMeta.from_fsdp_xccl` has to be executed on all ranks
    # due to `engine.get_param_specs()`.
    # Therefore, we create weight update meta on all ranks, then broadcast the one on rank 0.

    param_specs = actor.get_param_specs()
    weight_update_meta = WeightUpdateMeta(
        type="nccl",
        alloc_mode=allocation_mode,
        nccl_master_address="127.0.0.1",
        nccl_master_port=41653,
        nccl_param_specs=param_specs,
        nccl_group_name="update_weight_group",
    )
    # weight_update_meta = [
    #     WeightUpdateMeta.from_fsdp_xccl(
    #         AllocationMode.from_str(config.allocation_mode), actor
    #     )
    # ]
    # dist.broadcast_object_list(weight_update_meta, src=0)
    # weight_update_meta = weight_update_meta[0]

    # Create rollout workflow
    if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)

    workflow = RLVRWorkflow(
        reward_fn=gsm8k_reward_fn,
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        enable_thinking=False,
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated"
        ),
    )
    # eval_workflow = RLVRWorkflow(
    #     reward_fn=gsm8k_reward_fn,
    #     gconfig=config.gconfig.new(temperature=0.6),
    #     tokenizer=tokenizer,
    #     enable_thinking=False,
    #     rollout_stat_scope="eval-rollout",
    #     dump_dir=os.path.join(
    #         StatsLogger.get_log_path(config.stats_logger), "generated-eval"
    #     ),
    # )

    # Run training.
    Saver(config.saver, ft_spec)
    StatsLogger(config, ft_spec)
    # evaluator = Evaluator(config.evaluator, ft_spec)

    RecoverHandler(config.recover, ft_spec)
    # recover_info = recover_handler.load(
    #     actor,
    #     saver,
    #     evaluator,
    #     stats_logger,
    #     train_dataloader,
    #     inference_engine=rollout,
    #     weight_update_meta=weight_update_meta,
    # )
    # start_step = (
    #     recover_info.last_step_info.next().global_step
    #     if recover_info is not None
    #     else 0
    # )
    start_step = 0
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

        with stats_tracker.record_timing("rollout"):
            batch = None
            if config.async_training:
                batch = rollout.prepare_batch(
                    train_dataloader,
                    workflow=workflow,
                    should_accept=lambda sample: True,
                )
            else:
                data = next(data_generator)
                batch = DistributedBatchMemory.from_list(data)
                batch = rollout.rollout_batch(
                    batch,
                    workflow=workflow,
                )
            # batch = tensor_container_to(batch, actor.device)

            # batch = broadcast_tensor_container(
            #     batch,
            #     src_rank=actor.current_data_parallel_head(),
            #     group=actor.context_and_model_parallel_group,
            # )

        #     # Create barrier to synchronize all rollout processes.
        #     dist.barrier(device_ids=[actor.device.index])
        #     current_platform.synchronize()
        #

        if config.actor.recompute_logprob or config.actor.use_decoupled_loss:
            with stats_tracker.record_timing("recompute_logp"):
                logp = actor.compute_logp(batch)
                batch["prox_logp"] = logp
                # log_gpu_stats("recompute logp")

        if ref is not None:
            with stats_tracker.record_timing("ref_logp"):
                batch["ref_logp"] = ref.compute_logp(batch)
                # log_gpu_stats("ref logp")

        with stats_tracker.record_timing("compute_advantage"):
            batch = actor.compute_advantages(batch)
            # actor.compute_advantages(batch)
            # log_gpu_stats("compute advantages")

        with (
            stats_tracker.record_timing("train_step"),
            stats_tracker.scope("grpo_actor"),
        ):
            actor.ppo_update(batch)
            # stats = actor.ppo_update(batch)
            actor.step_lr_scheduler()
            # log_gpu_stats("ppo update")

        # pause inference for updating weights, save, and evaluation
        rollout.pause()

        with stats_tracker.record_timing("update_weights"):
            with ThreadPoolExecutor(max_workers=2) as executor:
                upload_future = executor.submit(
                    actor.upload_weights, weight_update_meta
                )
                update_future = executor.submit(
                    rollout.update_weights, weight_update_meta, rank=0
                )
                wait_future_ordered([upload_future, update_future])
            logger.info(
                f"{update_future} update weight succeeded (parallel), step: {step}, epoch: {epoch}"
            )

            # todo: 同步语句需要在upload_weights和upload_weights接口中包掉
            # dist.barrier(device_ids=[actor.device.index])
            # current_platform.synchronize()

            actor.set_version(global_step + 1)
            rollout.set_version(global_step + 1)
            # eval_rollout.set_version(global_step + 1)
    #
    #     with stats_tracker.record_timing("save"):
    #         saver.save(actor, epoch, step, global_step, tokenizer=tokenizer)
    #
    #     with stats_tracker.record_timing("checkpoint_for_recover"):
    #         recover_handler.dump(
    #             actor,
    #             step_info,
    #             saver,
    #             evaluator,
    #             stats_logger,
    #             train_dataloader,
    #             tokenizer=tokenizer,
    #         )
    #
    #     dist.barrier(device_ids=[actor.device.index])
    #     current_platform.synchronize()
    #
    #     with stats_tracker.record_timing("eval"):
    #
    #         def evaluate_fn():
    #             if actor.is_data_parallel_head():
    #                 cnt = 0
    #                 for data in valid_dataloader:
    #                     for item in data:
    #                         eval_rollout.submit(item, eval_workflow)
    #                         cnt += 1
    #                 eval_rollout.wait(cnt, timeout=None)
    #             dist.barrier(device_ids=[actor.device.index])
    #             current_platform.synchronize()
    #
    #         evaluator.evaluate(
    #             evaluate_fn,
    #             epoch,
    #             step,
    #             global_step,
    #         )
    #
    #     dist.barrier(device_ids=[actor.device.index])
    #     current_platform.synchronize()
    #
    #     # Upload statistics to the logger (e.g., wandb)
    #     stats[0].update(
    #         stats_tracker.export_all(reduce_group=actor.data_parallel_group)
    #     )
    #     stats_logger.commit(epoch, step, global_step, stats)
    #
    #     dist.barrier(device_ids=[actor.device.index])
    #     current_platform.synchronize()
    #
    #     # Resume rollout
    #     rollout.resume()
    #
    # stats_logger.close()
    # eval_rollout.destroy()
    # rollout.destroy()
    # if ref is not None:
    #     ref.destroy()
    # actor.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])
