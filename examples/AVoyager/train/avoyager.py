import asyncio
from calendar import c
from json import tool
import os
from platform import processor
import sys
import uuid
import json
import gc
from click import prompt
import torch
import torch.distributed as dist
from PIL import Image
import numpy as np
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from tensordict import TensorDict
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizerFast, AutoProcessor
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_processor_and_tokenizer
from areal.utils.recover import RecoverHandler
from areal.utils.data import broadcast_tensor_container, cycle_dataloader
from dataclasses import dataclass, field
from typing import List

import hashlib

from areal.api.cli_args import (
    GenerationHyperparameters,
    GRPOConfig,
    load_expr_config,
)
from areal.api.io_struct import (
    AllocationMode,
    FinetuneSpec,
    ModelRequest,
    WeightUpdateMeta,
    StepInfo,
)
from areal.api.workflow_api import RolloutWorkflow
from areal.api.cli_args import GRPOConfig
from areal.platforms import current_platform
from areal.engine.ppo.actor import FSDPPPOActor
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.utils.data import concat_padded_tensors
from areal.utils.device import log_gpu_stats
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger
from areal.utils import seeding, logging, stats_tracker

from AVoyager.train.voyage_agent import VoyageAgent
from AVoyager.utils.voyage_tool import VoyageToolBox
from AVoyager.utils.rewards import acc_reward, calculate_reward

worker_id = uuid.uuid4().hex[:4]

logger = logging.getLogger(f"AVoyager @ {worker_id}")

def hash(numbers):
    """Hash an entire list of integers as a single string"""
    # Convert list to string representation
    list_str = json.dumps(numbers, sort_keys=True)  # sort_keys for consistency
    return hashlib.sha256(list_str.encode()).hexdigest()


class AVoyagerWorkflow(RolloutWorkflow):
    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        processor: AutoProcessor = None,
        dataset_path: str,
        dump_dir: str | None = None,
        max_turns: int = 128,
        n_trajs: int = 1,
        voyager_client_types: list = ["image-grounding"],
        reward_type: str = "F1",
        valid_inst_ratio: float = 1.0,
        max_tokens: int = 32000,
        search_only: bool = True,
        reward_info: dict = None,
    ):
        self.gconfig = gconfig
        self.gconfig.n_samples = 1
        self.tokenizer = tokenizer
        self.processor = processor
        self.dump_dir = dump_dir
        self.max_tokens = max_tokens
        self.search_only = search_only
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

        # Search hyper-parameters
        self.max_turns = max_turns
        self.n_trajs = n_trajs
        self.reward_type = reward_type
        self.valid_inst_ratio = valid_inst_ratio
        self.voyager_client_types = voyager_client_types    
        self.reward_info=reward_info

        self.toolbox = VoyageToolBox(self.voyager_client_types)
    
    async def collect_agent_trajectory(self, valid_inst, qid, input_data, engine):
        '''
        input_data:{
            messages
            input_ids
            answer
            images (optional) 
            image_paths (optional)
        }
        '''
        agent = VoyageAgent(input_data)
        score = 0
        ground_truth = input_data["answer"] if "answer" in input_data else None
        # a unique trajectory rid to ensure all requests goes to the same sglang server
        traj_rid = uuid.uuid4().hex
        while agent.num_turns < self.max_turns and not agent.is_finished:
            # The agent prepares the prompt and sampling params for LLM generation
            input_ids, sampling_params = agent.prepare_llm_query()
            
            # Send request to inference engine and get response
            req = ModelRequest(
                rid=traj_rid,
                input_ids=input_ids,
                image_data=input_data.get("image_paths", None),
                gconfig=self.gconfig.new(n_samples=1),
            )
            if "stop" in sampling_params:
                req.gconfig.stop = sampling_params["stop"]
            if len(input_ids) + self.gconfig.max_new_tokens >= self.max_tokens:
                break
            resp = await engine.agenerate(req)
            completion_str = self.tokenizer.decode(resp.output_tokens)

            # agent extracts tool callings from the llm response
            tool_calls = agent.consume_llm_response(resp, completion_str)

            # call tool and compute reward
            if tool_calls is not None and len(tool_calls) > 0:
                tool_call = tool_calls[0]
                res = (await self.toolbox.step((qid, [tool_call])))[0]
                
                agent.consume_tool_response(res)

                # if "score" in res:
                #     score = res["score"]
                # if "ground_truth" in res:
                #     ground_truth = res["ground_truth"]

            if resp.output_tokens[-1] in [self.tokenizer.eos_token_id, self.tokenizer.pad_token_id]:
                break

        llm_gen_records = agent.memory.filter_records("llm_gen")
        pred_answer = agent.get_answer()
        
        # compute reward:{"score","acc_reward"，"format_reward","tool_call_reward"}
        # XXX: Mini-o3 attempted to call GPT API inside reward function and built a reward manager for this multithreading scenario. But in the final implementation this setting is not used. So we keep a direct call to compute_score here.
        reward_dict=compute_score(predict_str_list=llm_gen_records, ground_truth=ground_truth, extra_info=self.reward_info)
        
        score = reward_dict["score"]
        
        judge_q_invalid = False
        if pred_answer is not None:
            judge_q_invalid = any([_c in pred_answer for _c in ["question", "invalid", "appropriate", "valid"]])
        if valid_inst and judge_q_invalid:
            score = -0.5
            logger.warning(f"Invalid answer detected: {pred_answer}. Force to -0.5 score.")
        
        stats = agent.memory.logging_stats()
        stats.update(dict(
            score=score,
            judge_q_invalid=judge_q_invalid,
            acc_reward=reward_dict["acc_reward"],
            format_reward=reward_dict["format_reward"],
            tool_reward=reward_dict["tool_reward"],
        ))

        return ground_truth, score, agent.memory, stats       
    
    async def arun_episode(self, engine, data):
        '''
        data:{
            messages
            qid
            answer
            images (optional)
            image_paths (optional)
        }
        '''
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
        version = engine.get_version()
        input_data = dict()
        # always use grounding tool if image is provided
        if "image" in data.keys():
            self.toolbox.init_grounding_client(original_image=data["image"], processor=self.processor)
            
            processed_input = self.processor(
                images=[data["image"]],
                messages=data["messages"],
                padding=False,
                return_tensors="pt",
            )
            #BUG: Sglang forces to convert input_ids to prompt when processing multimodal data, which leads to slightly different input_ids in rollout and training.
            input_ids = processed_input["input_ids"].tolist()[0]
            input_data["images"] = data["image"]
            input_data["messages"] = data["messages"]
            input_data["input_ids"] = input_ids
            input_data["image_paths"]= data.get("image_path", None)
        else:
            input_ids = self.tokenizer.encode(data["messages"], add_special_tokens=False)
            input_data["messages"] = data["messages"]
            input_data["input_ids"] = input_ids
            
            logger.warning(f"No image provided for qid={qid}. The agent will not be able to use image grounding tool. This option is for more general text-based tools in future.")

        # for adjust train data ratio
        # valid_inst: bool = np.random.uniform(0, 1) <= self.valid_inst_ratio
        # if valid_inst:
        #     prompt = prompt.replace(INVALID_PROMPT, VALID_PROMPT)
        # prompt_token_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

        # Collect trajectories 
        trajs = await asyncio.gather(*[self.collect_agent_trajectory(valid_inst, qid, input_data, engine) for _ in range(self.n_trajs)])

        ground_truth, scores, results, stats = None, [], [], []
        for gt, score, traj, traj_stats in trajs:
            if gt is not None:
                ground_truth = gt
            scores.append(score)
            stats.append(traj_stats)
        
        raw_scores = scores
        score_mean = np.asarray(scores).mean()
        scores = [s-score_mean for s in scores]
        # logger.info(f"Scores @ qid={qid}: {raw_scores} -> {scores}")
        if all([s==0 for s in scores]):
            return None
        #TODO 9.25 整理image逻辑，sglang使用image path，其他使用PIL.image,对整个arun_episode加一个全家multimodal判断，直接拼接token_ids，标记不同的token_id位置设置loss_mask，logprobs，versions统一构建multimodal
        trajs = [traj for _, _, traj, _ in trajs]
        for i, traj_memory in enumerate(trajs):
            seqs = []
            for j, record in enumerate(traj_memory.memory):
                if record.type != "llm_gen":
                    continue

                # Check whether any previous seq is equivalent to input tokens
                success = False
                for seq in seqs:
                    if record.input_len  < len(seq["input_ids"]):
                        continue
                    h_cur = hash(record.input_tokens[:len(seq["input_ids"])])
                    h_seq = hash(seq["input_ids"])
                    if h_cur == h_seq:
                        seq_len = len(seq["input_ids"])
                        seq["input_ids"] = record.input_tokens + record.output_tokens
                        seq["logprobs"] += [0.0] * (record.input_len - seq_len) + record.output_logprobs
                        seq["loss_mask"] += [0] * (record.input_len - seq_len) + [1] * record.output_len
                        seq["versions"] += [-1] * (record.input_len - seq_len) + record.output_versions
                        success = True
                        break
                if not success:
                    seq = dict(
                        input_ids = record.input_tokens + record.output_tokens,
                        logprobs = [0.0] * record.input_len + record.output_logprobs,
                        loss_mask = [0] * record.input_len + [1] * record.output_len,
                        versions = [-1] * record.input_len + record.output_versions,
                    )
                    seqs.append(seq)

            traj_stats = stats.pop(0)
            first_llm_gen = True
        
            for seq in seqs:
                res = dict(
                    # unsqueeze to add an additional batch dimension
                    input_ids=torch.tensor(seq["input_ids"]).unsqueeze(0),
                    loss_mask=torch.tensor(seq["loss_mask"]).unsqueeze(0),
                    logprobs=torch.tensor(seq["logprobs"]).unsqueeze(0),
                    versions=torch.tensor(seq["versions"]).unsqueeze(0),
                    attention_mask=torch.ones(len(seq["input_ids"]), dtype=torch.bool).unsqueeze(0),
                    # reward
                    rewards=torch.tensor([float(scores[i])]),
                )

                res.update(dict(begin_of_trajectory=torch.tensor([int(first_llm_gen)]),))
                res.update({k: torch.tensor([v]) for k, v in traj_stats.items()})
                first_llm_gen = False

                results.append(TensorDict(res, batch_size=[1]))

        if self.dump_dir is not None:
            os.makedirs(os.path.join(self.dump_dir, str(version)), exist_ok=True)

            # Dump rollout to file
            with open(
                os.path.join(self.dump_dir, str(version), f"{qid}.jsonl"), "w"
            ) as f:
                for i, (traj_memory, raw_score) in enumerate(zip(trajs, raw_scores)):
                    f.write(json.dumps(dict(memory=traj_memory.to_dict(), reward=raw_score, ground_truth=ground_truth, traj_idx=i)) + "\n")

        results = concat_padded_tensors(results)
        return results

@dataclass
class AgentRLConfig(GRPOConfig):
    max_turns: int = field(
        default=128,
        metadata={
            "help": "maximum number of turns for search agent"
        }
    )
    n_trajs: int = field(
        default=1,
        metadata={
            "help": "We could collect multiple trajectories for a single query. By default n_trajs=1."
        }
    )
    search_client_type: str = field(
        default="async-online-search-access",
        metadata={
            "help": "Type of tool (async-online-search-access/async-search-access). By default we use 'async-online-search-access'"
        }
    )
    reward_type: str = field(
        default="F1",
        metadata={
            "help": "The type of reward function"
        }
    )
    valid_inst_ratio: float = field(
        default=1.0,
        metadata={
            "help": "We randomly force a ratio of queries to produce valid anwers. By default valid_inst_ratio=1.0"
        }
    )
    # Logging Agent Trajectories
    log_agent_stats:  bool = field(
        default=False,
        metadata={
            "help": "Log stats for agent trajectories"
        },
    )
    log_agent_stats_keys: List[str] = field(
        default_factory=lambda: ["num_llm_gens"],
        metadata={
            "help": "Keys of log stats for agent trajectories"
        },
    )


def get_voyage_dataset(dataset_path, tokenizer, rank, world_size):
    dataset = load_dataset(
        path=dataset_path,
        split="train",
    )
    # dataset = dataset.filter(lambda x: len(tokenizer.encode(x["question"])) <= 1024)
    return split_dataset_by_node(dataset, rank=rank, world_size=world_size)

def main(args):
    config, _ = load_expr_config(args, AgentRLConfig)
    config: AgentRLConfig

    rank = int(os.getenv("RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))
    processor, tokenizer = load_hf_processor_and_tokenizer(config.tokenizer_name_or_path)

    seeding.set_random_seed(config.seed, key=f"trainer{rank}")
    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    parallel_strategy = allocation_mode.train

    # Initialize train engine
    actor = FSDPPPOActor(config=config.actor)
    actor.create_process_group(parallel_strategy=parallel_strategy)
    ref = None

    # Create dataset and dataloaders
    worker_batch_size = config.train_dataset.batch_size // world_size
    train_dataloader = StatefulDataLoader(
        get_search_dataset(config.train_dataset.path, tokenizer, rank, world_size),
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
    rollout.initialize(train_data_parallel_size=parallel_strategy.dp_size)


    
    actor.initialize(None, ft_spec)
    ref = None

    # NOTE: Weight update meta only requires address and free port of rank 0,
    # but `WeightUpdateMeta.from_fsdp_nccl` has to be executed on all ranks
    # due to `engine.get_param_specs()`.
    # Therefore, we create weight update meta on all ranks, then broadcast the one on rank 0.
    weight_update_meta = [
        # WeightUpdateMeta.from_disk(config.experiment_name, config.trial_name, config.cluster.fileroot, "default")
        WeightUpdateMeta.from_fsdp_nccl(AllocationMode.from_str(config.allocation_mode), actor)
    ]
    dist.broadcast_object_list(weight_update_meta, src=0)
    weight_update_meta = weight_update_meta[0]

    # Create rollout workflow
    if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)
    workflow = AVoyagerWorkflow(
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        processor=processor,
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated"
        ),
        dataset_path=config.train_dataset.path,
        max_turns=config.max_turns,
        n_trajs=config.n_trajs,
        search_client_type=config.search_client_type,
        reward_type=config.reward_type,
        valid_inst_ratio=config.valid_inst_ratio,
        max_tokens=config.actor.mb_spec.max_tokens_per_mb,
    )

    # Run training.
    saver = Saver(config.saver, ft_spec)
    stats_logger = StatsLogger(config.stats_logger, ft_spec)
    evaluator = Evaluator(config.evaluator, ft_spec)

    # Recover
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

        print(f"Epoch {epoch}. Step: {step}/{steps_per_epoch}")

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
        
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

        with (
            stats_tracker.record_timing("train_step"),
            stats_tracker.scope("grpo_actor"),
        ):
            if config.log_agent_stats:
                agent_denominator = (batch["begin_of_trajectory"] > 0).bool()
                stats_tracker.denominator(agent=agent_denominator)
                stats_tracker.stat(
                    **{k: batch[k].float() for k in config.log_agent_stats_keys},
                    denominator="agent",
                )

            stats = actor.ppo_update(batch)
            actor.step_lr_scheduler()
            log_gpu_stats("actor update")

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
        stats[0].update(stats_tracker.export_all(reduce_group=actor.parallelism_group))
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


if __name__ == "__main__":
    main(sys.argv[1:])