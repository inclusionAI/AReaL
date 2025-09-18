import asyncio
import functools
import os
import time
import sys
import uuid
from concurrent.futures import ProcessPoolExecutor
from pebble import ProcessPool

import colorama
import torch
import torch.distributed as dist
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from tensordict import TensorDict
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizerFast
from areal.api.cli_args import GenerationHyperparameters, GRPOConfig, load_expr_config
from areal.api.io_struct import (
    AllocationMode,
    FinetuneSpec,
    ModelRequest,
    WeightUpdateMeta,
    StepInfo,
)
from areal.api.workflow_api import RolloutWorkflow
from areal.engine.ppo.actor import FSDPPPOActor
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.engine.vllm_remote import RemotevLLMEngine
from areal.utils.data import concat_padded_tensors
from areal.utils.device import log_gpu_stats
from areal.utils.saver import Saver
from areal.utils.recover import RecoverHandler
from areal.utils.evaluator import Evaluator
from areal.utils.stats_logger import StatsLogger
from areal.utils.model import get_model_update_meta
from realhf.api.core.data_api import load_hf_tokenizer
from realhf.base import logging, seeding, stats_tracker
from areal.platforms import is_npu_available
if is_npu_available:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu

import realhf.base.logging as logging
logger = logging.getLogger("boba_grpo")

logger = logging.getLogger("boba math")


REWARD_TIMEOUT_SECONDS = 30


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
        self.rw_executor = ProcessPool(max_workers=self.gconfig.max_workers) if self.gconfig.interruptable_processpool else ProcessPoolExecutor(max_workers=self.gconfig.max_workers)
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

    def preprocess_data(self, data):
        user_token = "<｜User｜>"
        assistant_token = "<｜Assistant｜>"
        think_token = "<think>"
        enable_thinking = False
        if user_token in data:
            data = data.replace("<｜User｜>", "")
        if assistant_token in data:
            data = data.replace("<｜Assistant｜>", "")
        if think_token in data:
            enable_thinking = True
            data = data.replace("<think>", "")
        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": data}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        return input_ids

    async def arun_episode(self, engine, data):
        input_ids = self.preprocess_data(data["prompt"])
        n_samples = self.gconfig.n_samples
        req = ModelRequest(
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
        loop = asyncio.get_event_loop()
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
            try:
                if self.gconfig.interruptable_processpool:
                    future = asyncio.wrap_future(self.rw_executor.schedule(
                        functools.partial(
                            self.reward_fn,
                            completions=completions_str,
                            prompt_ids=resp.input_tokens,
                            completion_ids=resp.output_tokens,
                            **data,
                        ),
                        timeout=REWARD_TIMEOUT_SECONDS,
                    ))
                else:
                    future = loop.run_in_executor(
                        self.rw_executor,
                        functools.partial(
                            self.reward_fn,
                            completions=completions_str,
                            prompt_ids=resp.input_tokens,
                            completion_ids=resp.output_tokens,
                            **data,
                        ),
                    )
                reward = await asyncio.wait_for(
                    future,
                    timeout=REWARD_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"Computing reward timeout after {REWARD_TIMEOUT_SECONDS}s. Set reward to 0."
                )
                reward = 0
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


def get_boba_math_dataset(path, tokenizer, rank, world_size):
    dataset = load_dataset(
        path="json",
        split="train",
        data_files=path,
    )
    dataset = dataset.filter(lambda x: len(tokenizer.encode(x["prompt"])) <= 1024)
    return split_dataset_by_node(dataset, rank=rank, world_size=world_size)


def boba_reward_fn(
    prompt, completions, prompt_ids, completion_ids, query_id, solutions, **kwargs
):
    from realhf.impl.dataset.math_parser import process_results

    label = 0
    for sol in solutions:
        x = process_results(completions, sol)
        label = label or x[0]
    return label


def main(args):
    config, _ = load_expr_config(args, GRPOConfig)
    config: GRPOConfig
    rank = int(os.getenv("RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))
    assert config.train_dataset.batch_size >= world_size, f"batch size({config.train_dataset.batch_size}) must larger or equal than world_size({world_size})!" 
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    seeding.set_random_seed(config.seed, key=f"trainer{rank}")
    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    parallel_strategy = allocation_mode.train
    train_dataset = get_boba_math_dataset(config.train_dataset.path, tokenizer, rank=rank, world_size=world_size)
        # Create dataset and dataloaders
    train_dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=config.train_dataset.batch_size,
        shuffle=config.train_dataset.shuffle,
        num_workers=config.train_dataset.num_workers,
        collate_fn=lambda x: x,
        drop_last=config.train_dataset.drop_last,
    )
    config.rollout.consumer_batch_size *= world_size
    config.rollout.max_concurrent_rollouts *= world_size

    actor = FSDPPPOActor(config=config.actor)
    actor.create_process_group(parallel_strategy=parallel_strategy)
    device=torch.device(int(os.environ["LOCAL_RANK"]))
    train_dataset_len = len(train_dataloader)
    dateset_len_tensor = torch.tensor([train_dataset_len], dtype=torch.long, device=device)
    train_dataset_len = dateset_len_tensor.item()
    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=train_dataset_len * config.train_dataset.batch_size,
        train_batch_size=config.train_dataset.batch_size,
    )

    # Initialize inference engine
    allocation_mode = config.allocation_mode
    allocation_mode = AllocationMode.from_str(allocation_mode)
    if allocation_mode.gen_backend == "vllm":
        rollout = RemotevLLMEngine(config.rollout)
    elif allocation_mode.gen_backend == "sglang":
        rollout = RemoteSGLangEngine(config.rollout)
    rollout.initialize(None, ft_spec)

    # Initialize train engine
    actor.initialize(None, ft_spec)
    ref = None
    if config.actor.kl_ctl > 0 and config.ref is not None:
        ref = FSDPPPOActor(config=config.ref)
        ref.initialize(None, ft_spec)

    # NOTE: Weight update meta only requires address and free port of rank 0,
    # but `WeightUpdateMeta.from_fsdp_nccl` has to be executed on all ranks
    # due to `engine.get_param_specs()`.
    # Therefore, we create weight update meta on all ranks, then broadcast the one on rank 0.
    # NOTE: Change to NCCL if running on local or ray
    # weight_update_meta = [
    #     WeightUpdateMeta.from_fsdp_nccl(
    #         AllocationMode.from_str(config.allocation_mode), actor
    #     )
    # ]
    weight_update_meta = get_model_update_meta(config, actor)
    weight_update_meta = weight_update_meta[0]

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
        data_generator = iter(train_dataloader)
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
        # Create barrier to synchronize all rollout processes.
        dist.barrier(device_ids=[actor.device.index])
        batch = batch.to(actor.device)
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
            saver.save(actor, epoch, step, global_step)
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
