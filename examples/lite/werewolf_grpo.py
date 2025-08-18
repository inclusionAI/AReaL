import itertools
import os
import sys

import torch
import torch.distributed as dist
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.api.io_struct import AllocationMode, FinetuneSpec, StepInfo, WeightUpdateMeta
from areal.dataset import get_custom_dataset
from areal.engine.ppo.actor import FSDPPPOActor
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.utils.device import log_gpu_stats
from areal.utils.evaluator import Evaluator
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger
from areal.workflow.werewolf import WerewolfWorkflow
from realhf.api.core.data_api import load_hf_tokenizer
from realhf.base import seeding, stats_tracker

from realhf.base import logging
logger = logging.getLogger("Werewolf GRPO Exp")

LOGGING_KEYS = [
    "traj_steps",
    "traj_len",
    "traj_input_len",
    "traj_output_len",
    "vill_rewards",
    "were_rewards",
    "vill_wins",
    "were_wins",
    "werewolf_kills",
    "werewolf_correct_kills",
    "villager_correct_votes",
    "villager_wrong_votes",
    "witch_heals",
    "witch_correct_heals",
    "witch_poisons",
    "witch_correct_poisons",
    "hunter_shots",
    "hunter_correct_shots",
]
def main(args):
    config, _ = load_expr_config(args, GRPOConfig)
    config: GRPOConfig

    rank = int(os.getenv("RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    opp_tokenizer = None
    if config.opp_tokenizer_path:
        opp_tokenizer = load_hf_tokenizer(config.opp_tokenizer_path)
    teacher_tokenizer = None
    if config.teacher_tokenizer_path:
        teacher_tokenizer = load_hf_tokenizer(config.teacher_tokenizer_path)

    seeding.set_random_seed(config.seed, key=f"trainer{rank}")

    train_dataset = get_custom_dataset(
        path=config.train_dataset.path,
        rank=rank,
        world_size=world_size,
        split="train",
        type=config.train_dataset.type,
        tokenizer=tokenizer,
    )

    train_dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=config.train_dataset.batch_size // world_size,
        shuffle=config.train_dataset.shuffle,
        num_workers=config.train_dataset.num_workers,
        collate_fn=lambda x: x,
        drop_last=config.train_dataset.drop_last,
    )
    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(train_dataloader) * config.train_dataset.batch_size,
        train_batch_size=config.train_dataset.batch_size,
    )

    rollout = RemoteSGLangEngine(config.rollout)
    rollout.initialize(None, ft_spec)

    opp_server_addrs = config.opp_server_addrs
    opp_rollout = None
    if opp_server_addrs != "":
        opp_rollout = RemoteSGLangEngine(config.rollout, opp_server_addrs)
        opp_rollout.initialize(None, ft_spec)
    teacher_server_addrs = config.teacher_server_addrs
    teacher_rollout = None
    if teacher_server_addrs != "":
        teacher_rollout = RemoteSGLangEngine(config.rollout, teacher_server_addrs)
        teacher_rollout.initialize(None, ft_spec)

    actor = FSDPPPOActor(config=config.actor)
    actor.initialize(None, ft_spec)
    ref = None
    if config.actor.kl_ctl > 0 and config.ref is not None:
        ref = FSDPPPOActor(config=config.ref)
        ref.initialize(None, ft_spec)

    weight_update_meta = [
        WeightUpdateMeta.from_disk(
            config.saver.experiment_name,
            config.saver.trial_name,
            config.saver.fileroot
        )
        # WeightUpdateMeta.from_fsdp_nccl(
        #     AllocationMode.from_str(config.allocation_mode), actor
        # )
    ]
    dist.broadcast_object_list(weight_update_meta, src=0)
    weight_update_meta = weight_update_meta[0]

    if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)
    if opp_tokenizer and (opp_tokenizer.pad_token_id not in config.gconfig.stop_token_ids):
        config.gconfig.stop_token_ids.append(opp_tokenizer.pad_token_id)
    if opp_tokenizer and (opp_tokenizer.eos_token_id not in config.gconfig.stop_token_ids):
        config.gconfig.stop_token_ids.append(opp_tokenizer.eos_token_id)
    if teacher_tokenizer and (teacher_tokenizer.pad_token_id not in config.gconfig.stop_token_ids):
        config.gconfig.stop_token_ids.append(teacher_tokenizer.pad_token_id)
    if teacher_tokenizer and (teacher_tokenizer.eos_token_id not in config.gconfig.stop_token_ids):
        config.gconfig.stop_token_ids.append(teacher_tokenizer.eos_token_id)
    workflow = WerewolfWorkflow(
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated"
        ),
        role=config.role,
        opp_rollout=opp_rollout,
        opp_tokenizer=opp_tokenizer,
        teacher_rollout=teacher_rollout,
        teacher_tokenizer=teacher_tokenizer,
    )

    saver = Saver(config.saver, ft_spec)
    stats_logger = StatsLogger(config.stats_logger, ft_spec)
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

    data_generator = itertools.cycle(train_dataloader)
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
            if config.async_training:
                batch = rollout.prepare_batch(train_dataloader, workflow=workflow)
            else:
                batch = rollout.rollout_batch(next(data_generator), workflow=workflow)

        batch = batch.to(actor.device)
        dist.barrier(device_ids=[actor.device.index])
        torch.cuda.synchronize()

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
        
        if "logging" in batch:
            vals = batch["logging"].float().cpu()
            mask = vals[:, 0] > 0
            if mask.any():
                filtered = vals[mask]
                
                # Average values (existing behavior)
                avg = filtered.mean(0).tolist()
                extra = dict(zip(LOGGING_KEYS, avg))

                # Min and Max values
                mins = filtered.min(0).values.tolist()
                maxs = filtered.max(0).values.tolist()

                for key, min_val, max_val in zip(LOGGING_KEYS, mins, maxs):
                    extra[f"{key}/min"] = min_val
                    extra[f"{key}/max"] = max_val

                stats[0].update(extra)


        with stats_tracker.record_timing("update_weights"):
            rollout.pause()
            if dist.get_rank() == 0:
                future = rollout.update_weights(weight_update_meta)
            actor.upload_weights(weight_update_meta)
            if dist.get_rank() == 0:
                future.result()
            dist.barrier(device_ids=[actor.device.index])
            torch.cuda.synchronize()
            rollout.resume()
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

        stats_logger.commit(epoch, step, global_step, stats)

    stats_logger.close()
    rollout.destroy()
    if ref is not None:
        ref.destroy()
    actor.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])


"""
vim ../AReaL-Lite/examples/lite/configs/gsm8k_grpo.yaml

sudo srun --mpi=pmi2 --ntasks=1 --gres=gpu:8 \
    --cpus-per-task=10 --mem=1500G --pty \
    singularity shell --nv --no-home --writable-tmpfs \
    --bind /storage:/storage /storage/openpsi/images/sglang-v0.4.9.post2-cu126-v2.sif

cd /storage/openpsi/users/xmy/inclusionAI/AReaL
pip install -e . --no-deps
export HF_ENDPOINT="https://hf-mirror.com"
export WANDB_API_KEY=local-667d8d7f101dad4eb9597d718d0c68f40e3792f9
export WANDB_BASE_URL=http://8.150.1.98:8080

python -m areal.launcher.slurm examples/lite/werewolf_grpo.py \
    --config examples/lite/configs/werewolf_grpo.yaml \
    stats_logger.wandb.mode=online \
    experiment_name=xmy-werewolf-double3 \
    trial_name=villager1

python -m areal.launcher.slurm examples/lite/gsm8k_grpo.py \
    --config examples/lite/configs/gsm8k_grpo.yaml \
    stats_logger.wandb.mode=online \
    experiment_name=xmy-gsm8k-slurm \
    trial_name=trial1
"""
