import os
import sys
import json
import random
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
print(sys.path)

# from tau2_train.domains.airline.environment import get_environment, get_tasks
from tau2_train.registry import ENV_DIC, TASK_DIC
from tau2_train.data_model.tasks import Task, RewardType
from tau2_train.agent import LLMAgent
from tau2_train.orchestrator import Orchestrator
from tau2_train.user_simulator import UserSimulator
from tau2_train.evaluator.evaluator import evaluate_simulation

import yaml
import numpy as np
from loguru import logger
logger.remove()

from dataclasses import dataclass, field, asdict
from copy import deepcopy
import uuid
import asyncio
import functools
import colorama
from concurrent.futures import ProcessPoolExecutor

import torch

from megatron.core import parallel_state as mpu
from torch import distributed as dist
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.utils.dataloader import create_dataloader
from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import GRPOConfig, load_expr_config, GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import FinetuneSpec, StepInfo, WeightUpdateMeta, ModelRequest
from areal.dataset import get_custom_dataset
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.engine.ppo.actor import MegatronPPOActor
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
from areal.utils import perf_tracer
# from areal.workflow.rlvr import RLVRWorkflow

from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from areal.api.workflow_api import RolloutWorkflow

from transformers import PreTrainedTokenizerFast

class Tau2Workflow(RolloutWorkflow):
    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        user_model: str = "",
        user_api_key: str = "empty",
        user_base_url: str = "",
        user_llm_args: dict | None = None,
        user_models: dict | None = None,  # domain-specific user models
        max_num_turns: int = 128,
        max_context_length: int = 32768,
        n_trajs: int = 1,
        reward_type: str = "all",
        dynamic_filtering: bool = False,
        dynamic_max_acc: float = 0.99,
        reward_norm_type: str | None = None,
        process_payment_history: bool = False,
        is_reasoning_model: bool = True,
        dump_dir: str | None = None,
        eval_model: str | None = None,
        eval_api_key: str | None = None,
        eval_base_url: str | None = None,
    ):
        self.gconfig = gconfig
        self.tokenizer = tokenizer
        self.dump_dir = dump_dir
        self.max_num_turns = max_num_turns
        self.n_trajs = n_trajs
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)
        self.max_context_length = max_context_length

        # Default user model configuration (fallback)
        self.user_model = user_model
        self.user_api_key = user_api_key
        self.user_base_url = user_base_url
        self.user_llm_args = user_llm_args
        
        # Domain-specific user models
        # Format: {domain: {model, api_key, base_url, llm_args}}
        self.user_models = user_models or {}

        self.eval_model = eval_model or user_model
        self.eval_api_key = eval_api_key or user_api_key
        self.eval_base_url = eval_base_url or user_base_url

        self.reward_type = reward_type
        self.dynamic_filtering = dynamic_filtering
        self.dynamic_max_acc = dynamic_max_acc
        self.reward_norm_type = reward_norm_type
        self.process_payment_history = process_payment_history
        self.is_reasoning_model = is_reasoning_model
    
    def get_user_config(self, domain: str) -> dict:
        """
        Get user simulator configuration for a specific domain.
        Falls back to default configuration if domain-specific config is not found.
        
        Args:
            domain: The domain name (airline, retail, telecom)
            
        Returns:
            dict with keys: model, api_key, base_url, llm_args
        """
        if domain in self.user_models:
            domain_config = self.user_models[domain]
            return {
                "model": domain_config.get("model", self.user_model),
                "api_key": domain_config.get("api_key", self.user_api_key),
                "base_url": domain_config.get("base_url", self.user_base_url),
                "llm_args": domain_config.get("llm_args", self.user_llm_args),
            }
        else:
            # Fallback to default configuration
            return {
                "model": self.user_model,
                "api_key": self.user_api_key,
                "base_url": self.user_base_url,
                "llm_args": self.user_llm_args,
            }
    
    def get_reward_type(self, domain: str) -> str:
        """
        Get reward type based on domain.
        Automatically selects the appropriate reward type for each domain.
        
        Args:
            domain: The domain name (airline, retail, telecom)
            
        Returns:
            Reward type string: 'db', 'env', or 'all'
        """
        # Domain-specific reward types
        domain_reward_types = {
            "airline": "db",
            "retail": "db",
            "telecom": "env",  # telecom uses ENV_ASSERTION
        }
        
        reward_type = domain_reward_types.get(domain, self.reward_type)
        return reward_type

    async def collect_agent_trajectory(self, engine, task, db_path):
        
        domain = task.user_scenario.instructions.domain
        assert domain in ["airline", "retail", "telecom"], f"{domain} is not supported"
        task_id = perf_tracer.get_task_id()
        if task_id is not None:
            session_id = perf_tracer.register_session(task_id)
            perf_tracer.set_session_id(session_id)
        else:
            session_id = None

        environment = ENV_DIC[domain](db_path)
        # environment = get_environment(db_path)
        agent = LLMAgent(
            engine, 
            self.tokenizer, 
            self.gconfig, 
            environment.get_policy(), 
            environment.get_tools(),
            max_context_length=self.max_context_length - 100,
            is_reasoning_model=self.is_reasoning_model,
        )
        
        try:
            user_tools = environment.get_user_tools()
        except ValueError:
            user_tools = None
            # print(f"[Workflow DEBUG] domain={domain}, no user_tools available", flush=True)
        
        # Get domain-specific user configuration
        user_config = self.get_user_config(domain)
        
        user = UserSimulator(
            tools=user_tools,
            instructions=task.user_scenario,
            llm=user_config["model"],
            api_key=user_config["api_key"],
            base_url=user_config["base_url"],
            llm_args=user_config["llm_args"],
        )

        orchestrator = Orchestrator(
            domain,
            agent=agent,
            user=user,
            environment=environment,
            task=task,
            perf_session_id=session_id,
            max_steps=self.max_num_turns,
        )

        async with perf_tracer.atrace_session_phase(session_id, "traj"):
            simulation = await orchestrator.run()

        async with perf_tracer.atrace_session_phase(session_id, "reward"):
            reward_info = await evaluate_simulation(
                task=task,
                simulation=simulation,
                evaluation_type="all",
                llm=self.eval_model,
                api_key=self.eval_api_key,
                base_url=self.eval_base_url,
                db_path=db_path,
                domain=domain,
                process_payment_history=self.process_payment_history,
            )
        
        messagaes = orchestrator.get_trajectory()
        traj_records = agent.records
        
        # Calculate reward (use domain-specific reward type)
        actual_reward_type = self.get_reward_type(domain)
        
        if actual_reward_type == "db":
            reward = reward_info.db_check.db_reward if reward_info.db_check is not None else 0
        elif actual_reward_type == "env":
            # For telecom domain
            reward_breakdown = reward_info.reward_breakdown or {}
            if RewardType.ENV_ASSERTION in reward_breakdown:
                reward = reward_breakdown[RewardType.ENV_ASSERTION]
            else:
                reward = reward_info.reward
        elif actual_reward_type == "all":
            try:
                reward = reward_info.db_check.db_reward
                if len(reward_info.action_checks) > 0:
                    reward *= np.mean([x.action_reward for x in reward_info.action_checks])
                if len(reward_info.nl_assertions) > 0:
                    reward *= np.mean([x.met for x in reward_info.nl_assertions])

                print(
                "[debug] reward info: ", task.id,
                reward_info.db_check.db_reward,
                [x.action_reward for x in reward_info.action_checks],
                [x.met for x in reward_info.nl_assertions]
            )

            except Exception as e:
                print("[debug] reward info: ", e, reward_info)
                reward = 0
        else:
            raise NotImplementedError

        return messagaes, traj_records, reward

    async def arun_episode(self, engine: InferenceEngine, raw_data=None):

        data = deepcopy(raw_data)
        if data is None:
            domain = "airline"
            tasks = TASK_DIC[domain]()
            task = random.choice(tasks)
            db_path = None
        else:
            data['evaluation_criteria'] = json.loads(data['evaluation_criteria'])
            db_path = data.get("db_path")
            # Clean up None values in arguments added by HF datasets schema inference
            # HF datasets merges all possible keys across rows, filling missing keys with None
            if 'initial_state' in data and data['initial_state'] is not None:
                init_actions = data['initial_state'].get('initialization_actions')
                if init_actions:
                    for action in init_actions:
                        if 'arguments' in action and action['arguments']:
                            action['arguments'] = {k: v for k, v in action['arguments'].items() if v is not None}
            task = Task.model_validate(data)
        
        trajs = await asyncio.gather(*[self.collect_agent_trajectory(engine, task, db_path) for _ in range(self.n_trajs)])
        version = engine.get_version()

        results = []
        rewards = []
        for i, (messagaes, traj_records, reward) in enumerate(trajs):
            rewards.append(reward)

        for i, (messagaes, traj_records, reward) in enumerate(trajs):
            if self.reward_norm_type == "grpo":
                reward = (reward - np.mean(rewards)) / (np.std(rewards) + 1e-6)
            elif self.reward_norm_type == "grpo-pos":
                reward = (reward - np.mean(rewards)) / (np.std(rewards) + 1e-6)
                reward = reward if reward > 0 else 0
            for j, record in enumerate(traj_records):

                seq = record.input_tokens + record.output_tokens
                logprobs = [0.0] * record.input_len + record.output_logprobs
                loss_mask = [0] * record.input_len + [1] * record.output_len
                versions = [-1] * record.input_len + record.output_versions

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
                if len(loss_mask) <= self.max_context_length:
                    results.append(res)
        
        if self.dump_dir is not None:
            os.makedirs(os.path.join(self.dump_dir, str(version)), exist_ok=True)

            # Dump rollout to file
            cur_uuid = uuid.uuid4().hex
            
            # Save scenario data in rollout files
            scenario_info = None
            if data is not None:
                scenario_info = {
                    "id": data.get("id"),
                    "db_path": data.get("db_path"),
                    "user_scenario": data.get("user_scenario"),
                    "evaluation_criteria": data.get("evaluation_criteria"),
                    "description": data.get("description"),
                }
            
            with open(
                os.path.join(self.dump_dir, str(version), f"{data['id']}_{cur_uuid}.jsonl"), "w"
            ) as f:
                for i, (messages, traj_records, score) in enumerate(trajs):
                    enhanced_messages = []
                    agent_record_idx = 0
                    thinking_count = 0 
                    
                    for msg_idx, msg in enumerate(messages):
                        if isinstance(msg, dict):
                            msg_dict = dict(msg)
                        elif hasattr(msg, 'model_dump'):
                            # Pydantic v2
                            msg_dict = msg.model_dump()
                        elif hasattr(msg, 'dict'):
                            # Pydantic v1
                            msg_dict = msg.dict()
                        else:
                            # 尝试直接转换
                            msg_dict = dict(msg)

                        if msg_dict.get("role") == "assistant" and agent_record_idx < len(traj_records):
                            raw_output = traj_records[agent_record_idx].text

                            thinking = None
                            if raw_output and "</think>" in raw_output:
                                # 提取 </think> 之前的所有内容作为 thinking
                                end = raw_output.find("</think>")
                                thinking = raw_output[:end].strip()
                            if thinking:
                                msg_dict["thinking"] = thinking
                                thinking_count += 1
                            
                            agent_record_idx += 1
                        
                        enhanced_messages.append(msg_dict)
                    
                    traj_data = {
                        "scenario": scenario_info, 
                        "messages": enhanced_messages,
                        "reward": score,
                        "traj_idx": i,
                    }
                    f.write(json.dumps(traj_data, ensure_ascii=False) + "\n")
            
            # Dump records to file
            with open(
                os.path.join(self.dump_dir, str(version), f"{data['id']}_record_{cur_uuid}.jsonl"), "w"
            ) as f:
                for i, (messages, traj_records, score) in enumerate(trajs):
                    cur_records = []
                    for j, record in enumerate(traj_records):
                        cur_records.append(
                            {
                                "input": self.tokenizer.decode(record.input_tokens),
                                "output": record.text,
                            }
                        )
                    f.write(json.dumps(cur_records, ensure_ascii=False) + "\n")
    
        if self.dynamic_filtering:
            if np.max(rewards) == 0:
                return None
            if np.mean(rewards) >= self.dynamic_max_acc:
                return None

        if len(results) == 0:
            return None
        else:
            # print("valid prompt", task.id, rewards)
            return concat_padded_tensors(results)


@dataclass
class SamplingHyperparameters:
    top_p: float = field(
        default=1.0,
        metadata={"help": "Nucleus sampling probability threshold (0.0, 1.0]."},
    )

    temperature: float = field(
        default=1.0,
        metadata={"help": "Sampling temperature. Higher values increase diversity."},
    )

    def new(self, **kwargs):
        args = asdict(self)
        args.update(kwargs)
        return SamplingHyperparameters(**args)

@dataclass
class TauRLConfig(GRPOConfig):
    max_turns: int = field(
        default=32,
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
    user_model: str = field(
        default="",
        metadata={
            "help": "Model name for user simulation."
        }
    )

    user_api_key: str = field(
        default="empty",
        metadata={
            "help": "api_key for user simulation model"
        }
    )

    user_base_url: str = field(
        default="",
        metadata={
            "help": "base_url for user simulation model"
        }
    )

    user_llm_args: SamplingHyperparameters | None = field(
        default=None,
        metadata={
            "help": "llm args for the user simulation model"
        }
    )

    user_models: dict | None = field(
        default=None,
        metadata={
            "help": "Domain-specific user models. Format: {domain: {model, api_key, base_url, llm_args}}"
        }
    )

    max_context_length: int = field(
        default=32768,
        metadata={
            "help": "Maximum context length of the trained model"
        }
    )
    
    reward_type: str = field(
        default="all",
        metadata={
            "help": "Reward type for training or evaluation. Options: db, env, all. 'env' is for telecom domain (ENV_ASSERTION only)."
        }
    )
    dynamic_filtering: bool = field(
        default=False,
        metadata={
            "help": "Whether to filter out the tasks dynamicly"
        }
    )
    dynamic_max_acc: float = field(
        default=0.99,
        metadata={
            "help": "Filtering threshold"
        }
    )
    reward_norm_type: str = field(
        default="none",
        metadata={
            "help": "Method to normalize the reward with in a group"
        }
    )
    process_payment_history: bool = field(
        default=False,
        metadata={
            "help": "Aggregate the payment history in the airline domain"
        }
    )
    is_reasoning_model: bool = field(
        default=True,
        metadata={
            "help": "Whether using reasoning model or active reasoning mode of model"
        }
    )

def get_tau2_dataset(dataset_path, tokenizer, rank=None, world_size=None):
    dataset = load_dataset(
        path="json",
        split="train",
        data_files=dataset_path,
    )
    # dataset = dataset.filter(lambda x: len(tokenizer.encode(x["question"])) <= 1024)
    if rank is None or world_size is None:
        return dataset
    else:
        return split_dataset_by_node(dataset, rank=rank, world_size=world_size)


def main(args):
    config, _ = load_expr_config(args, TauRLConfig)
    config: TauRLConfig

    with open(os.path.join(StatsLogger.get_log_path(config.stats_logger), "config.yaml"), 'w') as file:
        print("save config.yaml in", os.path.join(StatsLogger.get_log_path(config.stats_logger)))
        yaml.dump(asdict(config), file, default_flow_style=False, sort_keys=True)

    rank = int(os.getenv("RANK"))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    seeding.set_random_seed(config.seed, key=f"trainer{rank}")
    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    parallel_strategy = allocation_mode.train
    assert parallel_strategy is not None

    actor = MegatronPPOActor(config=config.actor)
    actor.create_process_group(parallel_strategy=parallel_strategy)

    # Configure performance tracer
    if config.perf_tracer is not None:
        perf_tracer.configure(config.perf_tracer, rank=rank)

    # Create dataset and dataloaders

    train_dataloader = create_dataloader(
        get_tau2_dataset(config.train_dataset.path, tokenizer),
        rank=mpu.get_data_parallel_rank(),
        world_size=mpu.get_data_parallel_world_size(),
        dataset_config=config.train_dataset,
    )

    valid_dataloader = StatefulDataLoader(
        get_tau2_dataset(config.valid_dataset.path, tokenizer, mpu.get_data_parallel_rank(), mpu.get_data_parallel_world_size()),
        batch_size=1,
        shuffle=False,
        num_workers=config.valid_dataset.num_workers,
        collate_fn=lambda x: x,
        drop_last=False,
    )
    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(train_dataloader) * config.train_dataset.batch_size,
        train_batch_size=config.train_dataset.batch_size,
    )

    # Initialize inference engine
    rollout = RemoteSGLangEngine(config.rollout)
    rollout.initialize(train_data_parallel_size=parallel_strategy.dp_size)
    eval_rollout = RemoteSGLangEngine(deepcopy(config.rollout))
    # NOTE: eval does not have any offpolicyness control
    eval_rollout.config.max_head_offpolicyness = int(1e12)
    eval_rollout.initialize()

    # Initialize train engine
    actor.initialize(
        None, ft_spec, parallel_strategy=parallel_strategy, seed=config.seed
    )

    weight_update_meta = WeightUpdateMeta.from_megatron_xccl(
        allocation_mode,
        nccl_group_name=actor.weight_update_group_name,
    )
    actor.connect_engine(rollout, weight_update_meta)

    ref = None

    # Create rollout workflow
    if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)
    
    # Process user_models: convert llm_args if needed
    processed_user_models = None
    if config.user_models is not None:
        processed_user_models = {}
        for domain, domain_config in config.user_models.items():
            processed_config = dict(domain_config)
            if "llm_args" in processed_config and processed_config["llm_args"] is not None:
                # Convert SamplingHyperparameters to dict if needed
                if hasattr(processed_config["llm_args"], "__dict__"):
                    processed_config["llm_args"] = asdict(processed_config["llm_args"])
            processed_user_models[domain] = processed_config
    
    workflow = Tau2Workflow(
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated"
        ),
        max_num_turns=config.max_turns,
        n_trajs=config.n_trajs,
        user_model=config.user_model,
        user_api_key=config.user_api_key,
        user_base_url=config.user_base_url,
        user_llm_args=asdict(config.user_llm_args) if config.user_llm_args is not None else None,
        user_models=processed_user_models,
        max_context_length=config.max_context_length,
        reward_type=config.reward_type,
        dynamic_filtering=config.dynamic_filtering,
        dynamic_max_acc=config.dynamic_max_acc,
        reward_norm_type=config.reward_norm_type,
        process_payment_history=config.process_payment_history,
        is_reasoning_model=config.is_reasoning_model,
    )

    eval_workflow = Tau2Workflow(
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated-eval"
        ),
        max_num_turns=config.max_turns,
        n_trajs=4,
        user_model=config.user_model,
        user_api_key=config.user_api_key,
        user_base_url=config.user_base_url,
        user_llm_args=asdict(config.user_llm_args) if config.user_llm_args is not None else None,
        user_models=processed_user_models,
        max_context_length=config.max_context_length,
        reward_type="db",
        dynamic_filtering=False,
        is_reasoning_model=config.is_reasoning_model,
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
    eval_rollout.set_version(start_step)
    for global_step in range(start_step, max_steps):
        
        if global_step % steps_per_epoch == 0:
            train_dataloader.sampler.set_epoch(global_step // steps_per_epoch)
        
        epoch = global_step // steps_per_epoch
        step = global_step % steps_per_epoch
        step_info = StepInfo(
            global_step=global_step,
            epoch=epoch,
            epoch_step=step,
            steps_per_epoch=steps_per_epoch,
        )

        # rollout.pause()
        # dist.barrier(device_ids=[actor.device.index])
        # current_platform.synchronize()
        # with stats_tracker.record_timing("eval"):

        #     def evaluate_fn():
        #         if actor.is_data_parallel_head():
        #             # Stats are logged in workflow
        #             # and will be exported later
        #             cnt = 0
        #             for data in valid_dataloader:
        #                 for item in data:
        #                     eval_rollout.submit(item, eval_workflow)
        #                     cnt += 1
        #             eval_rollout.wait(cnt, timeout=None)
        #         dist.barrier(device_ids=[actor.device.index])
        #         current_platform.synchronize()

        #     evaluator.evaluate(
        #         evaluate_fn,
        #         epoch,
        #         step,
        #         global_step,
        #     )

        # dist.barrier(device_ids=[actor.device.index])
        # current_platform.synchronize()
        # rollout.resume()

        with stats_tracker.record_timing("rollout"):
            batch = actor.prepare_batch(
                train_dataloader,
                granularity=actor.config.group_size,
                workflow=workflow,
            )

        # if config.actor.recompute_logprob or config.actor.use_decoupled_loss:
        #     with stats_tracker.record_timing("recompute_logp"):
        #         logp = actor.compute_logp(batch)
        #         batch["prox_logp"] = logp
        #         log_gpu_stats("recompute logp")

        if ref is not None:
            with stats_tracker.record_timing("ref_logp"):
                batch["ref_logp"] = ref.compute_logp(batch)
                log_gpu_stats("ref logp")

        with stats_tracker.record_timing("compute_advantage"):
            actor.compute_advantages(batch)
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
            eval_rollout.set_version(global_step + 1)

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
        stats = stats_tracker.export_all(reduce_group=mpu.get_data_parallel_group())
        stats_logger.commit(epoch, step, global_step, stats)

        dist.barrier(device_ids=[actor.device.index])
        current_platform.synchronize()

        # Resume rollout
        rollout.resume()
        perf_tracer.save(step=global_step)
    
    perf_tracer.save(force=True)
    stats_logger.close()
    eval_rollout.destroy()
    rollout.destroy()
    if ref is not None:
        ref.destroy()
    actor.destroy()
    actor.destroy_process_groups()


if __name__ == "__main__":
    main(sys.argv[1:])
