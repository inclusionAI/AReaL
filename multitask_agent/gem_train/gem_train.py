import os
import sys
import re
from dataclasses import dataclass, field
import uuid
import asyncio
import json
import random
import copy

import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from datasets import load_dataset
import numpy as np
import gem

from areal.api.cli_args import GRPOConfig, load_expr_config, GenerationHyperparameters
from areal.api.io_struct import (
    AllocationMode,
    FinetuneSpec,
    StepInfo,
    WeightUpdateMeta,
    ModelRequest,
)
from areal.api.engine_api import InferenceEngine
from areal.api.workflow_api import RolloutWorkflow
from areal.engine.ppo.actor import MegatronPPOActor
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.engine.vllm_remote import RemotevLLMEngine
from areal.platforms import current_platform
from areal.utils import logging, perf_tracer, seeding, stats_tracker
from areal.utils.dataloader import create_dataloader
from areal.utils.data import concat_padded_tensors
from areal.utils.device import log_gpu_stats
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.perf_tracer import Category, trace_perf
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger
from areal.workflow.rlvr import RLVRWorkflow

import multitask_agent.gem_train.minesweeper_with_template
import multitask_agent.gem_train.sudoku_with_template
import multitask_agent.gem_train.mastermind_with_template
from multitask_agent.gem_train.minesweeper.cpp_solver import CPPSolver

logger = logging.getLogger("GEM grpo")


class GEMWorkflow(RolloutWorkflow):
    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer,
        env_name: str,
        dump_dir: str | None = None,
        n_trajs: int = 1,
        max_tokens: int = 32000,
        dynamic_filtering: bool = True,
        reward_type: str = "sum",
        traj_gamma: float = 1.0,
        minesweeper_solver_path: str | None = None,
        minesweeper_solver_reward_coeff: float | None = None,
        minesweeper_random_first_move: bool = False,
    ):
        self.gconfig = gconfig
        self.tokenizer = tokenizer
        self.env_name = env_name
        self.dump_dir = dump_dir
        self.n_trajs = n_trajs
        self.max_tokens = max_tokens
        self.dynamic_filtering = dynamic_filtering
        assert reward_type in ["sum", "return"]
        self.reward_type = reward_type
        self.traj_gamma = traj_gamma
        self.minesweeper_solver_path = minesweeper_solver_path
        self.minesweeper_solver_reward_coeff = minesweeper_solver_reward_coeff
        self.minesweeper_random_first_move = minesweeper_random_first_move

        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

        if self.minesweeper_solver_reward_coeff is not None:
            self.minesweeper_solver = CPPSolver(
                solver_path=self.minesweeper_solver_path
            )

    async def collect_agent_trajectory(self, engine: InferenceEngine, template=None):
        env = gem.make(self.env_name)
        if template is not None:
            # print(f"debug====set template")
            env.set_template(template)
        obs, info = env.reset()
        _obs = obs + info.get("suffix", "")
        messages = [{"role": "user", "content": _obs}]
        rewards = [0.0]
        versions = [[]]
        tokens = [None]

        # print(f"debug====obs: {_obs}")
        traj_rid = uuid.uuid4().hex
        while True:
            input_ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
            req = ModelRequest(
                rid=traj_rid,
                input_ids=input_ids,
                gconfig=self.gconfig.new(
                    n_samples=1,
                ),
            )
            if len(input_ids) + self.gconfig.max_new_tokens >= self.max_tokens:
                break
            resp = await engine.agenerate(req)
            completion_str: str = self.tokenizer.decode(resp.output_tokens)
            tmp = completion_str.split("</think>")
            thinking = tmp[0] + "</think>"
            if len(tmp) < 2:
                action = ""
            else:
                action = tmp[1]
            flag = True
            while flag:
                action = action.rstrip()
                flag = False
                for stop in ["<|im_end|>", self.tokenizer.eos_token]:
                    if action.endswith(stop):
                        action = action[: -len(stop)]
                        flag = True
                        break
            messages.append(
                {"role": "assistant", "content": action, "thinking": thinking}
            )
            # print(f"debug===action: {completion_str}")

            next_obs, reward, terminated, truncated, info = env.step(action)
            if self.minesweeper_solver_reward_coeff is not None:
                solver_reward = self.minesweeper_solver.compute_reward(
                    env.rows, env.cols, _obs, action, debug=reward > 0
                )
                reward += self.minesweeper_solver_reward_coeff * solver_reward
                messages[-1]["aux_reward"] = (
                    self.minesweeper_solver_reward_coeff * solver_reward
                )
            _obs = next_obs + info.get("suffix", "")

            # TODO: tool
            messages.append({"role": "user", "content": _obs})
            rewards.extend([reward, 0.0])
            versions.extend([resp.output_versions, []])
            tokens.extend(
                [
                    {
                        "input_tokens": input_ids,
                        "output_tokens": resp.output_tokens,
                        "output_logprobs": resp.output_logprobs,
                    },
                    None,
                ]
            )

            obs = next_obs
            if terminated or truncated:
                break

        assert len(messages) == len(rewards), (len(messages), len(rewards))
        return messages, rewards, versions, tokens

    async def arun_episode(self, engine, data):
        if self.env_name == "game:Minesweeper-v0-easy-with-template":
            while True:
                template = gem.make("game:Minesweeper-v0-easy")
                if self.minesweeper_random_first_move:
                    r = random.randint(0, template.rows - 1)
                    c = random.randint(0, template.cols - 1)
                else:
                    r = template.rows // 2
                    c = template.cols // 2
                action = f"\\boxed{{reveal {r} {c}}}"
                obs, reward, terminated, truncated, info = template.step(action)
                if not (terminated or truncated):
                    break
        elif re.match(
            r"^game:(Sudoku|Mastermind)-v0-(.*)-with-template$", self.env_name
        ):
            print(f"debug===create template")
            template = gem.make(self.env_name[: -len("-with-template")])
        else:
            template = None
        trajs = await asyncio.gather(
            *[
                self.collect_agent_trajectory(engine, template=template)
                for _ in range(self.n_trajs)
            ]
        )

        results = []
        traj_returns = []
        env_returns = []
        successes = []
        for i, (messages, rewards, versions, tokens) in enumerate(trajs):
            R = 0.0
            # R = sum(rewards)
            for j in range(len(rewards) - 1, -1, -1):
                # reward = rewards[j]
                # R += reward
                if messages[j]["role"] != "assistant":
                    continue
                if self.reward_type == "sum":
                    R = sum(rewards)
                else:
                    R = rewards[j] + R * self.traj_gamma
                # prompt_ids = self.tokenizer.apply_chat_template(
                #     messages[:j], add_generation_prompt=True
                # )
                # input_ids = self.tokenizer.apply_chat_template(
                #     messages[:j]
                #     + [
                #         {
                #             "role": "assistant",
                #             "content": messages[j]["thinking"] + messages[j]["content"],
                #         }
                #     ]
                # )[
                #     :-1
                # ]  # TODO: write in a more elegant way
                # assert len(prompt_ids) < len(input_ids), (
                #     len(prompt_ids),
                #     len(input_ids),
                # )
                # for k in range(len(prompt_ids)):
                #     assert prompt_ids[k] == input_ids[k]
                # assert len(versions[j]) == len(input_ids) - len(prompt_ids), (
                #     len(versions[j]),
                #     len(prompt_ids),
                #     len(input_ids),
                # )
                # res = dict(
                #     input_ids=torch.tensor(input_ids).unsqueeze(0),
                #     logprobs=torch.tensor([0.0] * len(input_ids)).unsqueeze(0),
                #     loss_mask=torch.tensor(
                #         [0] * len(prompt_ids) + [1] * (len(input_ids) - len(prompt_ids))
                #     ).unsqueeze(0),
                #     versions=[-1] * len(prompt_ids) + versions[j],
                #     attention_mask=torch.ones(
                #         len(input_ids), dtype=torch.bool
                #     ).unsqueeze(0),
                #     rewards=torch.tensor([R]),
                # )
                input_tokens = tokens[j]["input_tokens"]
                output_tokens = tokens[j]["output_tokens"]
                input_len = len(input_tokens)
                output_len = len(output_tokens)
                assert len(versions[j]) == output_len, (len(versions[j]), output_len)
                res = dict(
                    input_ids=torch.tensor(input_tokens + output_tokens).unsqueeze(0),
                    logprobs=torch.tensor(
                        [0] * input_len + tokens[j]["output_logprobs"]
                    ).unsqueeze(0),
                    loss_mask=torch.tensor(
                        [0] * input_len + [1] * output_len
                    ).unsqueeze(0),
                    versions=torch.tensor([-1] * input_len + versions[j]).unsqueeze(0),
                    attention_mask=torch.ones(
                        input_len + output_len, dtype=torch.bool
                    ).unsqueeze(0),
                    rewards=torch.tensor([R]),
                )
                results.append(res)
            episode_return = sum(rewards)
            traj_returns.append(episode_return)
            env_return = episode_return - sum(
                [x.get("aux_reward", 0) for x in messages]
            )
            env_returns.append(env_return)
            success = float(np.abs(env_return - 1) < 1e-4)
            if "Minesweeper" in self.env_name:
                success = float("Congratulations!" in messages[-1]["content"])
            successes.append(success)
        stats_tracker.get("rollout").scalar(
            reward=np.mean(traj_returns),
            env_reward=np.mean(env_returns),
            success_rate=np.mean(successes),
        )

        if self.dump_dir is not None:
            version = engine.get_version()
            # print(f"debug====version: {version}")
            os.makedirs(os.path.join(self.dump_dir, str(version)), exist_ok=True)
            with open(
                os.path.join(self.dump_dir, str(version), f"{uuid.uuid4().hex}.jsonl"),
                "w",
            ) as f:
                for i, (messages, rewards, versions, tokens) in enumerate(trajs):
                    print(
                        json.dumps(
                            dict(messages=messages, rewards=rewards, tokens=tokens)
                        ),
                        file=f,
                    )

        results = concat_padded_tensors(results)
        if self.dynamic_filtering:
            if (
                torch.max(results["rewards"]) == 0.0
                or torch.min(results["rewards"]) == 1.0
            ):
                return None
        return results


@dataclass
class GEMConfig(GRPOConfig):
    env_name: str = "game:Minesweeper-v0-easy"
    n_trajs: int = 16
    max_traj_tokens: int = 16384
    reward_type: str = "sum"
    traj_gamma: float = 1.0
    invalid_action_reward: float | None = None

    minesweeper_solver_path: str | None = None
    minesweeper_solver_reward_coeff: float | None = None
    minesweeper_random_first_move: bool = False


def main(args):
    config, _ = load_expr_config(args, GEMConfig)
    config: GEMConfig
    rank = int(os.getenv("RANK"))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    if config.invalid_action_reward is not None:
        from gem.utils.constants import LanguageGameReward

        LanguageGameReward.invalid_action_reward = config.invalid_action_reward

    seeding.set_random_seed(config.seed, key=f"trainer{rank}")
    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    parallel_strategy = allocation_mode.train
    assert parallel_strategy is not None

    # Initialize train engine
    actor = MegatronPPOActor(config=config.actor)
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
    # train_dataset = get_boba_math_dataset(config.train_dataset.path, tokenizer)
    train_dataset = list(range(128 * 100))
    train_dataloader = create_dataloader(
        train_dataset,
        rank=mpu.get_data_parallel_rank(),
        world_size=mpu.get_data_parallel_world_size(),
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

    # Initialize train engine
    actor.initialize(
        None, ft_spec, parallel_strategy=parallel_strategy, seed=config.seed
    )
    weight_update_meta = WeightUpdateMeta.from_megatron_xccl(
        allocation_mode, nccl_group_name=actor.weight_update_group_name
    )
    actor.connect_engine(rollout, weight_update_meta)

    ref = None
    if config.actor.kl_ctl > 0 and config.ref is not None:
        ref = MegatronPPOActor(config=config.ref)
        ref.initialize(
            None, ft_spec, parallel_strategy=parallel_strategy, seed=config.seed
        )

    # Create rollout workflow
    if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)
    workflow = GEMWorkflow(
        config.gconfig,
        tokenizer,
        config.env_name,
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated"
        ),
        n_trajs=config.n_trajs,
        max_tokens=config.max_traj_tokens,
        reward_type=config.reward_type,
        traj_gamma=config.traj_gamma,
        minesweeper_solver_path=config.minesweeper_solver_path,
        minesweeper_solver_reward_coeff=config.minesweeper_solver_reward_coeff,
        minesweeper_random_first_move=config.minesweeper_random_first_move,
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

        dist.barrier(device_ids=[actor.device.index])
        current_platform.synchronize()

        # Upload statistics to the logger (e.g., wandb)
        with perf_tracer.trace_scope(
            "train.log_stats",
            category=Category.INSTR,
            args={"global_step": global_step},
        ):
            stats = stats_tracker.export_all(reduce_group=mpu.get_data_parallel_group())
            stats_logger.commit(epoch, step, global_step, stats)

        dist.barrier(device_ids=[actor.device.index])
        current_platform.synchronize()

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
