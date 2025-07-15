import os
import re
import sys

import torch
import torch.distributed as dist
from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from torchdata.stateful_dataloader import StatefulDataLoader

from arealite.api.cli_args import GRPOConfig, load_expr_config
from arealite.api.io_struct import FinetuneSpec, WeightUpdateMeta
from arealite.engine.ppo.actor import FSDPPPOActor
from arealite.engine.sglang_remote import RemoteSGLangEngine
from arealite.utils.device import log_gpu_stats
from arealite.utils.evaluator import Evaluator
from arealite.utils.saver import Saver
from arealite.utils.stats_logger import StatsLogger
from arealite.workflow.rlvr import RLVRWorkflow
from realhf.api.core.data_api import load_hf_tokenizer
from realhf.base import stats_tracker





# Adapted from verl.
def extract_solution(solution_str, method="strict") -> str | None:
    assert method in ["strict", "flexible"]

    final_answer = None
    if method == "strict":
        # this also tests the formatting of the model
        solutions = re.findall("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if len(solutions) == 0:
            final_answer = None
        else:
            # take the last solution
            final_answer = solutions[-1].replace(",", "").replace("$", "")
    elif method == "flexible":
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ["", "."]
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer


def gsm8k_reward_fn(prompt, completions, prompt_ids, completion_ids, answer, **kwargs):
    from realhf.impl.dataset.math_parser import extract_answer

    sol = extract_answer(completions, data_name="math")
    ans = extract_solution(solution_str=answer, method="strict")
    if sol is None:
        return 0
    if ans is None:
        return 0
    return int(sol.strip() == ans.strip())


def main_grpo():
    config, _ = load_expr_config(sys.argv[1:], GRPOConfig)
    config: GRPOConfig

    rank = int(os.getenv("RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    # Create dataset and dataloaders
    train_dataloader = StatefulDataLoader(
        get_gsm8k_dataset("train", rank, world_size),
        batch_size=config.train_dataset.batch_size // world_size,
        shuffle=config.train_dataset.shuffle,
        num_workers=config.train_dataset.num_workers,
        collate_fn=lambda x: x,
        drop_last=config.train_dataset.drop_last,
    )
    valid_dataloader = StatefulDataLoader(
        get_gsm8k_dataset("test", rank, world_size),
        batch_size=config.valid_dataset.batch_size // world_size,
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
    rollout.initialize(None, ft_spec)
    eval_rollout = RemoteSGLangEngine(config.rollout)
    eval_rollout.initialize(None, ft_spec)
    # NOTE: set a large version such that eval does not have any offpolicyness control
    eval_rollout.set_version(int(1e12))

    # Initialize train engine
    actor = FSDPPPOActor(config=config.actor)
    actor.initialize(None, ft_spec)
    ref = None
    if config.actor.kl_ctl > 0 and config.ref is not None:
        ref = FSDPPPOActor(config=config.ref)
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
    )

    # Run training.
    saver = Saver(config.saver, ft_spec, for_recover=False)
    logger = StatsLogger(config.stats_logger, ft_spec)
    evaluator = Evaluator(config.evaluator, ft_spec)

    total_epochs = config.total_train_epochs
    steps_per_epoch = len(train_dataloader)
    max_steps = total_epochs * steps_per_epoch

    logger.info(f"total_epochs={total_epochs} step_per_epoch={steps_per_epoch}")
    data_generator = iter(train_dataloader)
    for global_step in range(max_steps):
        epoch = global_step // steps_per_epoch
        step = global_step % steps_per_epoch

        with stats_tracker.record_timing("rollout"):
            if config.async_training:
                batch = rollout.prepare_batch(
                    data_generator,
                    train_dataloader,
                    workflow=workflow,
                )
            else:
                try:
                    data = next(data_generator)
                except StopIteration:
                    data_generator = iter(train_dataloader)
                    data = next(data_generator)
                batch = rollout.rollout_batch(data, workflow=workflow)

        batch = batch.to(actor.device)
        # Create barrier to synchronize all rollout processes.
        dist.barrier()
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

        with stats_tracker.record_timing("update_weights"):
            meta = WeightUpdateMeta(
                type="disk",
                path=os.path.join(
                    Saver.get_save_checkpoint_root(config.saver),
                    "update_weights",
                    str(global_step),
                ),
                alloc_mode=None,
                comm_backend=None,
                model_version=global_step + 1,
            )
            if dist.get_rank() == 0:
                future = rollout.update_weights(meta)
            actor.upload_weights(meta)
            if dist.get_rank() == 0:
                future.result()
            rollout.set_version(global_step + 1)
            dist.barrier()

        with stats_tracker.record_timing("save"):
            saver.save(actor, epoch, step, global_step)

        with stats_tracker.record_timing("eval"):

            def evaluate_fn():
                rollout.pause()
                cnt = 0
                for data in valid_dataloader:
                    for item in data:
                        eval_rollout.submit(item, workflow)
                        cnt += 1
                batch = eval_rollout.wait(cnt, timeout=None)
                rewards = batch["rewards"].float().to(actor.device)
                with stats_tracker.scope("grpo-eval"):
                    stats_tracker.denominator(
                        n_seqs=torch.ones(
                            rewards.shape[0],
                            device=rewards.device,
                            dtype=torch.bool,
                        )
                    )
                    stats_tracker.stat(task_reward=rewards, denominator="n_seqs")
                rollout.resume()

            evaluator.evaluate(
                evaluate_fn,
                epoch,
                step,
                global_step,
            )

        logger.commit(epoch, step, global_step, stats)

    logger.close()
    eval_rollout.destroy()
    rollout.destroy()
    if ref is not None:
        ref.destroy()
    actor.destroy()


if __name__ == "__main__":
    main_grpo()
