import itertools
import os
import sys
from copy import deepcopy
import uuid
import asyncio
import functools
import colorama
from concurrent.futures import ProcessPoolExecutor

import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import load_expr_config, GenerationHyperparameters
from areal.api.io_struct import FinetuneSpec, StepInfo, WeightUpdateMeta, ModelRequest
from areal.dataset import get_custom_dataset
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.experimental.api.cli_args import ExperimentalGRPOConfig as GRPOConfig
from areal.experimental.megatron_actor import MegatronPPOActor
from areal.platforms import current_platform
from areal.utils import seeding, logging, stats_tracker
from areal.utils.data import (
    concat_padded_tensors,
    broadcast_tensor_container,
    cycle_dataloader,
    tensor_container_to,
)
from areal.utils.device import log_gpu_stats
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger
# from areal.workflow.rlvr import RLVRWorkflow

from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from areal.api.workflow_api import RolloutWorkflow

from tensordict import TensorDict
from transformers import PreTrainedTokenizerFast

logger = logging.getLogger("boba grpo math")

REWARD_TIMEOUT_SECONDS = 150


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
        # self.rw_executor = ProcessPoolExecutor(max_workers=32)
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

    async def arun_episode(self, engine, data):
        input_ids = self.tokenizer.encode(data["prompt"])
        n_samples = self.gconfig.n_samples
        req = ModelRequest(
            rid=uuid.uuid4().hex,
            input_ids=input_ids,
            gconfig=self.gconfig.new(n_samples=1),
            tokenizer=self.tokenizer,
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
            with ProcessPoolExecutor(max_workers=1) as executor:
                try:
                    reward = await asyncio.wait_for(
                        loop.run_in_executor(
                            executor,
                            functools.partial(
                                self.reward_fn,
                                completions=completions_str,
                                prompt_ids=resp.input_tokens,
                                completion_ids=resp.output_tokens,
                                **data,
                            ),
                        ),
                        timeout=REWARD_TIMEOUT_SECONDS,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        f"Computing reward timeout after {REWARD_TIMEOUT_SECONDS}s. Set reward to 0."
                    )
                    executor.shutdown(wait=False, cancel_futures=True)
                    reward = 0
                except Exception as e:
                    logger.warning(
                        f"Error in computing reward timeout: {e}. Set reward to 0."
                    )
                    executor.shutdown(wait=False, cancel_futures=True)
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
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    seeding.set_random_seed(config.seed, key=f"trainer{rank}")
    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    parallel_strategy = allocation_mode.train

    actor = MegatronPPOActor(config=config.actor)
    actor.create_process_group(parallel_strategy=parallel_strategy)

    # train_dataset = get_custom_dataset(
    #     path=config.train_dataset.path,
    #     rank=mpu.get_data_parallel_rank(),
    #     world_size=mpu.get_data_parallel_world_size(),
    #     split="train",
    #     type=config.train_dataset.type,
    #     tokenizer=tokenizer,
    # )

    train_dataloader = StatefulDataLoader(
        get_boba_math_dataset(config.train_dataset.path, tokenizer, mpu.get_data_parallel_rank(), mpu.get_data_parallel_world_size()),
        batch_size=config.train_dataset.batch_size // mpu.get_data_parallel_world_size(),
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
    rollout.initialize(train_data_parallel_size=parallel_strategy.dp_size)

    # Initialize train engine
    actor.initialize(
        None, ft_spec, parallel_strategy=parallel_strategy, seed=config.seed
    )
    ref = None

    # NOTE: megatron engine currently does not support nccl weight update
    weight_update_meta = WeightUpdateMeta.from_disk(
        config.experiment_name,
        config.trial_name,
        config.cluster.fileroot,
    )

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
            # NOTE: Megatron will ignore inputs from non data parallel heads.
            # However, methods such as `compute_advantages` and `ppo_update`
            # still requires data integrity to be called on every training process.
            # Therefore, we still need to broadcast batch across context+model parallel group.
            batch = None
            if actor.is_data_parallel_head():
                if config.async_training:
                    batch = rollout.prepare_batch(train_dataloader, workflow=workflow)
                else:
                    batch = rollout.rollout_batch(
                        next(data_generator), workflow=workflow
                    )
                batch = tensor_container_to(batch, actor.device)
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
            # eval_rollout.set_version(global_step + 1)

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
            stats_tracker.export_all(reduce_group=mpu.get_data_parallel_group())
        )
        stats_logger.commit(epoch, step, global_step, stats)

        dist.barrier(device_ids=[actor.device.index])
        current_platform.synchronize()

        # Resume rollout
        rollout.resume()
    stats_logger.close()
    rollout.destroy()
    if ref is not None:
        ref.destroy()
    actor.destroy()
    actor.destroy_process_groups()


if __name__ == "__main__":
    main(sys.argv[1:])
