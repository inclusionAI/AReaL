import os
import sys

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import GRPOConfig, SGLangConfig, load_expr_config, vLLMConfig
from areal.api.io_struct import FinetuneSpec, StepInfo, WeightUpdateMeta
from areal.controller.rollout_controller import RolloutController
from areal.controller.train_controller import TrainController
from areal.dataset import get_custom_dataset
from areal.engine.ppo.actor import FSDPPPOActor
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.engine.vllm_remote import RemotevLLMEngine
from areal.scheduler.local import LocalScheduler
from areal.utils import stats_tracker
from areal.utils.data import (
    cycle_dataloader,
)
from areal.utils.dataloader import create_dataloader
from areal.utils.device import log_gpu_stats
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger


def main(args):
    config, _ = load_expr_config(args, GRPOConfig)
    config: GRPOConfig

    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    # Create dataset and dataloaders
    train_dataset = get_custom_dataset(
        split="train", dataset_config=config.train_dataset, tokenizer=tokenizer
    )

    train_dataloader = create_dataloader(
        train_dataset,
        rank=0,
        world_size=1,
        dataset_config=config.train_dataset,
    )

    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(train_dataloader) * config.train_dataset.batch_size,
        train_batch_size=config.train_dataset.batch_size,
    )

    # Initialize scheduler
    scheduler = LocalScheduler(exp_config=config)

    # Initialize train controller
    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    actor = TrainController(FSDPPPOActor, config=config.actor, scheduler=scheduler)
    actor.initialize(
        role="actor", alloc_mode=allocation_mode, ft_spec=ft_spec, addr=None
    )

    # Initialize inference engine

    if allocation_mode.gen_backend == "sglang":
        engine_class = RemoteSGLangEngine
        server_args = SGLangConfig.build_args(
            sglang_config=config.sglang,
            tp_size=allocation_mode.gen.tp_size,
            base_gpu_id=0,
        )
    elif allocation_mode.gen_backend == "vllm":
        engine_class = RemotevLLMEngine
        server_args = vLLMConfig.build_args(
            vllm_config=config.vllm,
            tp_size=allocation_mode.gen.tp_size,
            pp_size=allocation_mode.gen.pp_size,
        )
    else:
        raise ValueError(f"Unsupported gen_backend: '{allocation_mode.gen_backend}'")

    rollout = RolloutController(
        engine_class, config=config.rollout, scheduler=scheduler
    )
    rollout.initialize(
        role="rollout",
        alloc_mode=allocation_mode,
        server_args=server_args,
    )

    weight_update_meta = WeightUpdateMeta.from_disk(
        experiment_name=config.experiment_name,
        trial_name=config.trial_name,
        file_root=config.cluster.fileroot,
    )
    actor.connect_engine(rollout, weight_update_meta)

    ref = None
    if config.actor.kl_ctl > 0 and config.ref is not None:
        ref = TrainController(FSDPPPOActor, config=config.ref, scheduler=scheduler)
        ref.initialize(
            role="ref", alloc_mode=allocation_mode, ft_spec=ft_spec, addr=None
        )

    # Run training.
    saver = Saver(config.saver, ft_spec)
    stats_logger = StatsLogger(config, ft_spec)
    evaluator = Evaluator(config.evaluator, ft_spec)

    recover_handler = RecoverHandler(config.recover, ft_spec)

    try:
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

            with stats_tracker.record_timing("rollout"):
                workflow_kwargs = dict(
                    reward_fn="areal.reward.gsm8k.gsm8k_reward_fn",
                    gconfig=config.gconfig,
                    tokenizer=config.tokenizer_path,
                    enable_thinking=False,
                    dump_dir=os.path.join(
                        StatsLogger.get_log_path(config.stats_logger),
                        "generated",
                    ),
                )
                if config.rollout.max_head_offpolicyness > 0:
                    batch = actor.prepare_batch(
                        train_dataloader,
                        workflow="areal.workflow.rlvr.RLVRWorkflow",
                        workflow_kwargs=workflow_kwargs,
                    )
                else:
                    batch = actor.rollout_batch(
                        next(data_generator),
                        workflow="areal.workflow.rlvr.RLVRWorkflow",
                        workflow_kwargs=workflow_kwargs,
                    )

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
                batch = actor.compute_advantages(batch)
                log_gpu_stats("compute advantages")

            with stats_tracker.record_timing("train_step"):
                actor.ppo_update(batch)
                actor.step_lr_scheduler()
                log_gpu_stats("ppo update")

            # pause inference for updating weights, save, and evaluation
            rollout.pause()

            with stats_tracker.record_timing("update_weights"):
                actor.update_weights(weight_update_meta)

                actor.set_version(global_step + 1)
                rollout.set_version(global_step + 1)

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

            # Upload statistics to the logger (e.g., wandb)
            stats_logger.commit(epoch, step, global_step, actor.export_stats())

            # Resume rollout
            rollout.resume()

    finally:
        stats_logger.close()
        rollout.destroy()
        if ref is not None:
            ref.destroy()
        actor.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])
