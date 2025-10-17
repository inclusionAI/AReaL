import asyncio
import hashlib
import json
import os
import sys
import uuid
from dataclasses import dataclass, field

import numpy as np
import torch.distributed as dist
from agents import Agent as OpenAIAgent
from agents import ModelSettings, OpenAIProvider, RunConfig
from agents import Runner as OpenAIRunner
from agents import SQLiteSession
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
from areal.experimental.openai.agent_patch import AReaLOpenAIClientContext
from areal.platforms import current_platform
from areal.utils import logging, seeding, stats_tracker
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

from .agent.search_agent import SearchAgent
from .utils.prompts import (
    INVALID_PROMPT,
    SEARCH_ACCESS_PROMPT_TEMPLATE,
    SEARCH_ONLY_PROMPT_TEMPLATE,
    VALID_PROMPT,
)
from .utils.rewards import correct_format_fn

worker_id = uuid.uuid4().hex[:4]

logger = logging.getLogger(f"ASearcher @ {worker_id}")


def hash(numbers):
    """Hash an entire list of integers as a single string"""
    # Convert list to string representation
    list_str = json.dumps(numbers, sort_keys=True)  # sort_keys for consistency
    return hashlib.sha256(list_str.encode()).hexdigest()


@dataclass
class ASearcherRLConfig(GRPOConfig):
    n_trajs: int = field(
        default=1,
        metadata={
            "help": "We could collect multiple trajectories for a single query. By default n_trajs=1."
        },
    )
    max_turns: int = field(
        default=8,
        metadata={
            "help": "Maximum number of turns per trajectory. By default max_turns=8."
        },
    )
    max_tokens_per_trajectory: int = field(
        default=32768,
        metadata={
            "help": "Maximum number of tokens per trajectory. By default max_tokens_per_trajectory=32768."
        },
    )
    search_client_type: str = field(
        default="async-online-search-access",
        metadata={
            "help": "Type of search client (async-online-search-access/async-search-access)."
        },
    )
    reward_type: str = field(
        default="F1",
        metadata={"help": "The type of reward function for search results."},
    )
    topk: int = field(
        default=5,
        metadata={"help": "Search returns the top-k results. Default top_k=5."},
    )


def search_reward_fn(result, ground_truth):
    """
    Reward function for search agent based on F1 score or other metrics.
    This is a placeholder - you should implement the actual reward logic.
    """
    # Placeholder implementation - replace with actual reward calculation
    if ground_truth is None:
        return 0.0

    # Simple reward based on whether the result contains relevant information
    # You should implement proper F1 score calculation here
    if isinstance(result, str) and isinstance(ground_truth, str):
        # Simple keyword matching as placeholder
        result_lower = result.lower()
        ground_truth_lower = ground_truth.lower()
        if ground_truth_lower in result_lower:
            return 1.0
        else:
            return 0.0

    return 0.0


class ASearcherAgent:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        max_tokens_per_turn: int = 1024,
        max_turns: int = 8,
        max_total_tokens: int = 32768,
        search_client_type: str = "async-online-search-access",
        reward_type: str = "F1",
        topk: int = 5,
        dataset_path: str = None,
    ):
        self.tokenizer = tokenizer
        self.max_tokens_per_turn = max_tokens_per_turn
        self.max_turns = max_turns
        self.max_total_tokens = max_total_tokens
        self.search_client_type = search_client_type
        self.reward_type = reward_type
        self.topk = topk
        self.dataset_path = dataset_path
        self.async_reward_fn = AsyncRewardWrapper(search_reward_fn)

    async def run_agent(self, data, valid_inst, qid, prompt, client: ArealOpenAI):
        base_run_config = RunConfig(
            model_provider=OpenAIProvider(openai_client=client, use_responses=False),
            tracing_disabled=True,
        )

        # Create search agent with appropriate tools
        openai_agent = OpenAIAgent(
            name="ASearcher",
            # Add search tools here - you'll need to implement these
            # tools=[SearchTool(), WebSearchTool(), etc.]
        )

        SQLiteSession("search")

        agent = SearchAgent(prompt)
        score = 0
        ground_truth = None

        async with AReaLOpenAIClientContext(base_run_config):

            while agent.num_turns < self.max_turns and not agent.is_finished:
                # The agent prepares the prompt and sampling params for LLM generation
                sampling_params = agent.prepare_llm_query()

                # if len(input_ids) + self.gconfig.max_new_tokens >= self.max_tokens:
                #     break

                extra_args = {"max_completion_tokens": self.max_tokens_per_turn}
                if "stop" in sampling_params:
                    extra_args["stop"] = sampling_params["stop"]

                run_config = RunConfig(
                    model_settings=ModelSettings(extra_args=extra_args)
                )

                resp = await OpenAIRunner.run(
                    openai_agent, input=prompt, run_config=run_config
                )
                completion_str = resp.final_output

                # agent extracts tool callings from the llm response
                tool_calls = agent.consume_llm_response(resp, completion_str)

                # call tool and compute reward
                if tool_calls is not None and len(tool_calls) > 0:
                    tool_call = tool_calls[0]
                    res = (await self.toolbox.step((qid, [tool_call])))[0]

                    agent.consume_tool_response(res, topk=self.topk)

                    if "score" in res:
                        score = res["score"]
                    if "ground_truth" in res:
                        ground_truth = res["ground_truth"]

                if resp.output_tokens[-1] in [
                    self.tokenizer.eos_token_id,
                    self.tokenizer.pad_token_id,
                ]:
                    break

            llm_gen_records = agent.memory.filter_records("llm_gen")
            format_reward = float(
                all(
                    [
                        correct_format_fn(i, r.text)
                        for i, r in enumerate(llm_gen_records)
                    ]
                )
            )

            # compute rewards
            score = (score or 0) * format_reward
            pred_answer = agent.get_answer()
            judge_q_invalid = False
            if pred_answer is not None:
                judge_q_invalid = any(
                    [
                        _c in pred_answer
                        for _c in ["question", "invalid", "appropriate", "valid"]
                    ]
                )
            if valid_inst and judge_q_invalid:
                score = -0.5

            stats = agent.memory.logging_stats()
            stats.update(
                dict(
                    score=score,
                    judge_q_invalid=judge_q_invalid,
                    format_reward=format_reward,
                )
            )

            client.set_final_reward(score)

            return ground_truth, score, agent.memory, stats


class ASearcherWorkflow(RolloutWorkflow):
    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        dump_dir: str | None = None,
        rollout_stat_scope: str = "rollout",
        n_trajs: int = 1,
        max_tokens: int = 32768,
        max_turns: int = 8,
        search_client_type: str = "async-online-search-access",
        reward_type: str = "F1",
        topk: int = 5,
        dataset_path: str = None,
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
        self.agent = ASearcherAgent(
            tokenizer=self.tokenizer,
            max_tokens_per_turn=self.gconfig.max_new_tokens,
            max_turns=max_turns,
            max_total_tokens=max_tokens,
            search_client_type=search_client_type,
            reward_type=reward_type,
            topk=topk,
            dataset_path=dataset_path,
        )

    async def arun_episode(self, engine, data):
        # Get the unique identifier for this prompt
        qid = None
        for key in ["query_id", "id", "qid"]:
            qid = data.get(key, None)
            if qid is not None:
                break
        qid = str(qid) or uuid.uuid4().hex

        # check for generated qid when resuming
        if self.dump_dir is not None:
            import glob

            _pattern = os.path.join(self.dump_dir, "*", f"{qid}.jsonl")
            if len(glob.glob(_pattern)) > 0:
                logger.info(f"{qid} is already trained on")
                return None

        # Initialize and Prepare the prompt
        engine.get_version()
        prompt_template = (
            SEARCH_ONLY_PROMPT_TEMPLATE
            if self.search_only
            else SEARCH_ACCESS_PROMPT_TEMPLATE
        )
        prompt = prompt_template.format(question=data["question"])
        valid_inst: bool = np.random.uniform(0, 1) <= self.valid_inst_ratio
        if valid_inst:
            prompt = prompt.replace(INVALID_PROMPT, VALID_PROMPT)
        # prompt_token_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

        clients = [
            ArealOpenAI(engine=engine, tokenizer=self.tokenizer)
            for _ in range(self.n_trajs)
        ]

        # Collect trajectories
        trajs = await asyncio.gather(
            *[
                self.agent.run_agent(
                    data=data,
                    valid_inst=valid_inst,
                    qid=qid,
                    prompt=prompt,
                    client=clients[i],
                )
                for i in range(self.n_trajs)
            ]
        )

        # ground_truth, scores, results, stats = None, [], [], []
        # for gt, score, traj, traj_stats in trajs:
        #     if gt is not None:
        #         ground_truth = gt
        #     scores.append(score)
        #     stats.append(traj_stats)

        # raw_scores = scores
        # score_mean = np.asarray(scores).mean()
        # scores = [s-score_mean for s in scores]
        # # logger.info(f"Scores @ qid={qid}: {raw_scores} -> {scores}")
        # if all([s==0 for s in scores]):
        #     return None

        # trajs = [traj for _, _, traj, _ in trajs]
        # for i, traj_memory in enumerate(trajs):
        #     seqs = []
        #     for j, record in enumerate(traj_memory.memory):
        #         if record.type != "llm_gen":
        #             continue

        #         # Check whether any previous seq is equivalent to input tokens
        #         success = False
        #         for seq in seqs:
        #             if record.input_len  < len(seq["input_ids"]):
        #                 continue
        #             h_cur = hash(record.input_tokens[:len(seq["input_ids"])])
        #             h_seq = hash(seq["input_ids"])
        #             if h_cur == h_seq:
        #                 seq_len = len(seq["input_ids"])
        #                 seq["input_ids"] = record.input_tokens + record.output_tokens
        #                 seq["logprobs"] += [0.0] * (record.input_len - seq_len) + record.output_logprobs
        #                 seq["loss_mask"] += [0] * (record.input_len - seq_len) + [1] * record.output_len
        #                 seq["versions"] += [-1] * (record.input_len - seq_len) + record.output_versions
        #                 success = True
        #                 break
        #         if not success:
        #             seq = dict(
        #                 input_ids = record.input_tokens + record.output_tokens,
        #                 logprobs = [0.0] * record.input_len + record.output_logprobs,
        #                 loss_mask = [0] * record.input_len + [1] * record.output_len,
        #                 versions = [-1] * record.input_len + record.output_versions,
        #             )
        #             seqs.append(seq)

        #     traj_stats = stats.pop(0)
        #     first_llm_gen = True

        #     for seq in seqs:
        #         res = dict(
        #             # unsqueeze to add an additional batch dimension
        #             input_ids=torch.tensor(seq["input_ids"]).unsqueeze(0),
        #             loss_mask=torch.tensor(seq["loss_mask"]).unsqueeze(0),
        #             logprobs=torch.tensor(seq["logprobs"]).unsqueeze(0),
        #             versions=torch.tensor(seq["versions"]).unsqueeze(0),
        #             attention_mask=torch.ones(len(seq["input_ids"]), dtype=torch.bool).unsqueeze(0),
        #             # reward
        #             rewards=torch.tensor([float(scores[i])]),
        #         )

        #         res.update(dict(begin_of_trajectory=torch.tensor([int(first_llm_gen)]),))
        #         res.update({k: torch.tensor([v]) for k, v in traj_stats.items()})
        #         first_llm_gen = False

        #         results.append(TensorDict(res, batch_size=[1]))

        # if self.dump_dir is not None:
        #     os.makedirs(os.path.join(self.dump_dir, str(version)), exist_ok=True)

        #     # Dump rollout to file
        #     with open(
        #         os.path.join(self.dump_dir, str(version), f"{qid}.jsonl"), "w"
        #     ) as f:
        #         for i, (traj_memory, raw_score) in enumerate(zip(trajs, raw_scores)):
        #             f.write(json.dumps(dict(memory=traj_memory.to_dict(), reward=raw_score, ground_truth=ground_truth, traj_idx=i)) + "\n")

        # results = concat_padded_tensors(results)
        # return results

        # for reward in rewards:
        #     stats_tracker.get(self.rollout_stat_scope).scalar(reward=reward)

        completions_with_reward = {}
        for client in clients:
            client.apply_reward_discount(turn_discount=0.9)
            completions = client.export_completions(style="individual")
            completions_with_reward.update(completions)
        return completions_with_reward


def main(args):
    config, _ = load_expr_config(args, ASearcherRLConfig)
    config: ASearcherRLConfig

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
    workflow = ASearcherWorkflow(
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        n_trajs=config.n_trajs,
        max_tokens=config.max_tokens_per_trajectory,
        max_turns=config.max_turns,
        search_client_type=config.search_client_type,
        reward_type=config.reward_type,
        topk=config.topk,
        dataset_path=config.train_dataset.path,
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
