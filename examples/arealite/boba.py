import asyncio
import os
import shutil
import sys
import uuid

import colorama
import torch
import torch.distributed as dist
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from tensordict import TensorDict
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizerFast

from arealite.api.cli_args import (
    GenerationHyperparameters,
    GRPOConfig,
    load_expr_config,
)
from arealite.api.io_struct import FinetuneSpec, LLMRequest, WeightUpdateMeta
from arealite.api.workflow_api import RolloutWorkflow
from arealite.engine.ppo.actor import FSDPPPOActor
from arealite.engine.sglang_remote import RemoteSGLangEngine
from arealite.utils.data import concat_padded_tensors
from arealite.utils.device import log_gpu_stats
from arealite.utils.saver import Saver
from arealite.utils.stats_logger import StatsLogger
from realhf.api.core.data_api import load_hf_tokenizer
from realhf.base import logging, seeding, stats_tracker

logger = logging.getLogger("boba math")


class RLVRWorkflow(RolloutWorkflow):
    def __init__(
        self,
        reward_fn,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        dump_dir: str | None = None,
    ):
        self.reward_fn = reward_fn
        self.gconfig = gconfig
        self.tokenizer = tokenizer
        self.dump_dir = dump_dir
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

    async def arun_episode(self, engine, data):
        input_ids = self.tokenizer.encode(data["prompt"])
        n_samples = self.gconfig.n_samples
        req = LLMRequest(
            rid=uuid.uuid4().hex,
            input_ids=input_ids,
            gconfig=self.gconfig.new(n_samples=1),
        )
        resps = await asyncio.gather(*[engine.agenerate(req) for _ in range(n_samples)])

        version = engine.get_version()
        prompt_strs = []
        completions_strs = []
        rewards = []
        seqlens = []

        results = []
        for resp in resps:
            seq = resp.input_tokens + resp.output_tokens
            logprobs = [0.0] * resp.input_len + resp.output_logprobs
            loss_mask = [0] * resp.input_len + [1] * resp.output_len
            versions = [-1] * resp.input_len + resp.output_versions

            prompt_str = data["prompt"]
            completions_str = self.tokenizer.decode(resp.output_tokens)
            prompt_strs.append(prompt_str)
            completions_strs.append(completions_str)
            seqlens.append(len(seq))
            reward = self.reward_fn(
                completions=completions_str,
                prompt_ids=resp.input_tokens,
                completion_ids=resp.output_tokens,
                **data,
            )
            rewards.append(reward)
            res = dict(
                # unsqueeze to add an additional batch dimension
                input_ids=torch.tensor(seq).unsqueeze(0),
                loss_mask=torch.tensor(loss_mask).unsqueeze(0),
                logprobs=torch.tensor(logprobs).unsqueeze(0),
                versions=torch.tensor(versions).unsqueeze(0),
                attention_mask=torch.ones(len(seq), dtype=torch.bool).unsqueeze(0),
                # reward
                rewards=torch.tensor([float(reward)]),
            )
            results.append(TensorDict(res, batch_size=[1]))

        if self.dump_dir is not None:
            os.makedirs(os.path.join(self.dump_dir, str(version)), exist_ok=True)
            # Get the unique identifier for this prompt
            qid = None
            for key in ["query_id", "id", "qid"]:
                qid = data.get(key, None)
                if qid is not None:
                    break
            qid = qid or uuid.uuid4().hex

            # Dump rollout to file
            with open(
                os.path.join(self.dump_dir, str(version), f"{qid}.txt"), "a"
            ) as f:
                n_samples = self.gconfig.n_samples
                for i, (p, c, r, sl) in enumerate(
                    zip(prompt_strs, completions_strs, rewards, seqlens)
                ):
                    info = "\n".join(
                        [
                            f"idx: {i + 1} / {n_samples}, seqlen: {sl}, reward is {r}.",
                            f"prompt is \n{colorama.Fore.YELLOW + colorama.Style.DIM}{p}{colorama.Style.RESET_ALL}",
                            f"sequence is: \n{colorama.Fore.YELLOW + colorama.Style.DIM}{c}{colorama.Style.RESET_ALL}",
                        ]
                    )
                    f.write(info + "\n")

        return concat_padded_tensors(results)


def get_boba_math_dataset(tokenizer, rank, world_size):
    dataset = load_dataset(
        path="json",
        split="train",
        data_files="/storage/openpsi/users/xushusheng.xss/training_data/boba_106k_0319.jsonl",
    )
    dataset = dataset.filter(lambda x: len(tokenizer.encode(x["prompt"])) <= 1024)
    return split_dataset_by_node(dataset, rank=rank, world_size=world_size)


def boba_reward_fn(
    prompt, completions, prompt_ids, completion_ids, query_id, solutions, **kwargs
):
    from pebble import ProcessExpired, ProcessPool

    from realhf.impl.dataset.math_parser import process_results

    jobs = []
    with ProcessPool(max_workers=1) as executor:
        for sol in solutions:
            job = executor.schedule(
                process_results, args=[completions, sol], timeout=15
            )
            jobs.append(job)

    label = 0
    for job in jobs:
        try:
            x = job.result()
        except TimeoutError:
            # print("[debug: timeout]")
            logger.warning(f"Timeout occurred while justifying the math answer.")
            x = (0, "timeout", "timeout")
        except ProcessExpired as e:
            logger.warning(f"Process terminated abnormally: {e}")
            x = (0, "error", "error")
        except Exception as e:
            logger.warning(f"Other error occurred: {e.__class__.__name__}, {e}")
            x = (0, "error", "error")
        label = label or x[0]
    return label


def main(args):
    config, _ = load_expr_config(args, GRPOConfig)
    config: GRPOConfig

    rank = int(os.getenv("RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    seeding.set_random_seed(config.seed, key=f"trainer{rank}")

    # Create dataset and dataloaders
    train_dataloader = StatefulDataLoader(
        get_boba_math_dataset(tokenizer, rank, world_size),
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

    # Initialize inference engine
    rollout = RemoteSGLangEngine(config.rollout)
    rollout.initialize(None, ft_spec)

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
        reward_fn=boba_reward_fn,
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated"
        ),
    )

    # Run training.
    saver = Saver(config.saver, ft_spec, for_recover=False)
    logger = StatsLogger(config.stats_logger, ft_spec)

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
                batch = rollout.prepare_batch(train_dataloader, workflow=workflow)
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
            path = os.path.join(
                Saver.get_save_checkpoint_root(config.saver),
                "update_weights",
                str(global_step + 1),
            )
            meta = WeightUpdateMeta(
                type="disk",
                path=path,
                alloc_mode=None,
                comm_backend=None,
                model_version=global_step + 1,
            )
            if dist.get_rank() == 0:
                future = rollout.update_weights(meta)
            actor.upload_weights(meta)
            if dist.get_rank() == 0:
                future.result()
                shutil.rmtree(path, ignore_errors=True)
            dist.barrier()
            torch.cuda.synchronize()
            rollout.set_version(global_step + 1)

        with stats_tracker.record_timing("save"):
            saver.save(actor, epoch, step, global_step)

        logger.commit(epoch, step, global_step, stats)

    logger.close()
    rollout.destroy()
    if ref is not None:
        ref.destroy()
    actor.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])
