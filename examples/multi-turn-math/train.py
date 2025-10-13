import asyncio
import os
import sys
from dataclasses import dataclass, field

import torch.distributed as dist
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizerFast

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import GenerationHyperparameters, GRPOConfig, load_expr_config
from areal.api.io_struct import FinetuneSpec, StepInfo, WeightUpdateMeta
from areal.api.reward_api import AsyncRewardWrapper
from areal.api.workflow_api import RolloutWorkflow
from areal.dataset import get_custom_dataset
from areal.engine.ppo.actor import FSDPPPOActor
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.experimental.openai import ArealOpenAI
from areal.platforms import current_platform
from areal.utils import seeding, stats_tracker
from areal.utils.data import (
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


@dataclass
class AgentRLConfig(GRPOConfig):
    n_trajs: int = field(
        default=1,
        metadata={
            "help": "We could collect multiple trajectories for a single query. By default n_trajs=1."
        },
    )
    max_turns: int = field(
        default=8,
        metadata={
            "help": "Maximum number of turns per trajectory. By default max_turns=32."
        },
    )
    max_tokens_per_trajectory: int = field(
        default=32768,
        metadata={
            "help": "Maximum number of tokens per trajectory. By default max_tokens_per_trajectory=32768."
        },
    )


def gsm8k_reward_fn(result, answer):
    from areal.reward.math_parser import process_results

    return int(process_results(result, answer)[0])


class MultiTurnMathAgent:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        max_tokens_per_turn: int = 1024,
        max_turns: int = 8,
        max_total_tokens: int = 32768,
    ):
        self.tokenizer = tokenizer
        self.max_tokens_per_turn = max_tokens_per_turn
        self.max_turns = max_turns
        self.max_total_tokens = max_total_tokens
        self.async_reward_fn = AsyncRewardWrapper(gsm8k_reward_fn)

    async def run_agent(self, data, client: ArealOpenAI):
        messages = data["messages"].copy()
        num_turns_left = self.max_turns
        completions = []
        while num_turns_left > 0:
            response = await client.chat.completions.create(
                messages=messages,
                temperature=1.0,
                max_completion_tokens=self.max_tokens_per_turn,
            )
            completions.append(response)
            content = response.choices[0].message.content
            messages.append({"role": "assistant", "content": content})
            reward = await self.async_reward_fn(result=content, answer=data["answer"])
            client.set_reward(response.id, reward)
            if reward == 1:
                break
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": "Your answer is either wrong or not parsable to the reward function. You may misunderstand the original question. "
                        "Please carefully read the original question, check the preivous errors, and try to answer it again.",
                    }
                )
            num_turns_left -= 1
        return reward


class MultiturnRLVRWorkflow(RolloutWorkflow):
    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        dump_dir: str | None = None,
        rollout_stat_scope: str = "rollout",
        n_trajs: int = 1,
        max_tokens: int = 32768,
        max_turns: int = 8,
    ):
        self.gconfig = gconfig
        self.gconfig.n_samples = 1
        self.tokenizer = tokenizer
        self.dump_dir = dump_dir
        self.max_tokens = max_tokens
        self.rollout_stat_scope = rollout_stat_scope
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

        # Search hyper-parameters
        self.n_trajs = n_trajs
        self.agent = MultiTurnMathAgent(
            tokenizer=self.tokenizer,
            max_tokens_per_turn=self.gconfig.max_new_tokens,
            max_turns=max_turns,
            max_total_tokens=max_tokens,
        )

    async def arun_episode(self, engine, data):
        clients = [
            ArealOpenAI(engine=engine, tokenizer=self.tokenizer)
            for _ in range(self.n_trajs)
        ]

        # Collect trajectories
        rewards = await asyncio.gather(
            *[
                self.agent.run_agent(
                    data=data,
                    client=clients[i],
                )
                for i in range(self.n_trajs)
            ]
        )
        for reward in rewards:
            stats_tracker.get(self.rollout_stat_scope).scalar(reward=reward)

        completions_with_reward = {}
        for client in clients:
            client.apply_reward_discount(turn_discount=0.9)
            completions = client.export_completions(style="individual")
            completions_with_reward.update(completions)
        return completions_with_reward


def main(args):
    config, _ = load_expr_config(args, AgentRLConfig)
    config: AgentRLConfig

    rank = int(os.getenv("RANK"))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    seeding.set_random_seed(config.seed, key=f"trainer{rank}")
    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    parallel_strategy = allocation_mode.train
    assert parallel_strategy is not None

    # Initialize train engine
    actor = FSDPPPOActor(config=config.actor)
    actor.create_process_group(parallel_strategy=parallel_strategy)

    train_dataset = get_custom_dataset(
        path=config.train_dataset.path,
        rank=actor.data_parallel_rank,
        world_size=actor.data_parallel_world_size,
        split="train",
        max_length=config.train_dataset.max_length,
        type=config.train_dataset.type,
        tokenizer=tokenizer,
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
    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(train_dataloader) * config.train_dataset.batch_size,
        train_batch_size=config.train_dataset.batch_size,
    )

    # Initialize inference engine
    rollout = RemoteSGLangEngine(config.rollout)
    rollout.initialize(train_data_parallel_size=parallel_strategy.dp_size)

    weight_update_meta = WeightUpdateMeta.from_fsdp_xccl(allocation_mode)

    actor.initialize(None, ft_spec)
    actor.connect_engine(rollout, weight_update_meta)

    # Create rollout workflow
    if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)
    workflow = MultiturnRLVRWorkflow(
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        n_trajs=config.n_trajs,
        max_tokens=config.max_tokens_per_trajectory,
        max_turns=config.max_turns,
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated"
        ),
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

        dist.barrier(device_ids=[actor.device.index])
        current_platform.synchronize()

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
    rollout.destroy()
    actor.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])
