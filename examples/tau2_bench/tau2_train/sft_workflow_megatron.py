import os
import sys
import copy
import asyncio
import uuid
import json
import random
import yaml

import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from tensordict import TensorDict
from torchdata.stateful_dataloader import StatefulDataLoader
from typing import Optional
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from transformers import AutoTokenizer, PreTrainedTokenizerFast
import numpy as np
from dataclasses import dataclass, field, asdict
from copy import deepcopy

# Add tau2 workflow imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from tau2_train.domains.airline.environment import get_environment, get_tasks
from tau2_train.data_model.tasks import Task
from tau2_train.agent import LLMAgent
from tau2_train.orchestrator import Orchestrator
from tau2_train.user_simulator import UserSimulator
from tau2_train.evaluator.evaluator import evaluate_simulation

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import (
    load_expr_config, 
    GenerationHyperparameters, 
    InferenceEngineConfig,
    BaseExperimentConfig,
)
from areal.api.io_struct import FinetuneSpec, StepInfo, WeightUpdateMeta
from areal.api.engine_api import InferenceEngine
from areal.api.workflow_api import RolloutWorkflow
from areal.experimental.megatron_lm_engine import MegatronLMEngine
from areal.experimental.api.cli_args import ExperimentalTrainEngineConfig as TrainEngineConfig
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.platforms import current_platform
from areal.utils import seeding, stats_tracker
from areal.utils.data import broadcast_tensor_container, pad_sequences_to_tensors, concat_padded_tensors
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger
from datetime import datetime

def safe_serialize(obj, field_name=None):
    if isinstance(obj, datetime):
        return obj.isoformat()

    elif isinstance(obj, dict):
        return {k: safe_serialize(v, field_name=k) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [safe_serialize(v, field_name=field_name) for v in obj]

    elif obj is None:
        if field_name == "tool_calls":
            return []
        else:
            return ""

    else:
        return obj


class Tau2Workflow(RolloutWorkflow):
    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        user_model: str = "/storage/openpsi/models/Qwen__Qwen2.5-72B-Instruct/",
        user_api_key: str = "empty",
        user_base_url: str = "",
        max_num_turns: int = 128,
        max_context_length: int = 32768,
        n_trajs: int = 4,
        reward_type: str = "db",
        dump_dir: str | None = None,
        eval_model: str | None = None,
        eval_api_key: str | None = None,
        eval_base_url: str | None = None,
        is_reasoning_model: bool = False,
    ):
        self.gconfig = gconfig
        self.tokenizer = tokenizer
        self.dump_dir = dump_dir
        self.max_num_turns = max_num_turns
        self.n_trajs = n_trajs
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)
        self.max_context_length = max_context_length

        self.user_model = user_model
        self.user_api_key = user_api_key
        self.user_base_url = user_base_url

        self.eval_model = eval_model or user_model
        self.eval_api_key = eval_api_key or user_api_key
        self.eval_base_url = eval_base_url or user_base_url

        self.reward_type = reward_type
        self.is_reasoning_model = is_reasoning_model

    async def collect_agent_trajectory(self, engine, task):
        
        environment = get_environment()
        agent = LLMAgent(
            engine, 
            self.tokenizer, 
            self.gconfig, 
            environment.get_policy(), 
            environment.get_tools(),
            max_context_length=self.max_context_length - 100,
            is_reasoning_model=self.is_reasoning_model
        )
        user = UserSimulator(
            instructions=task.user_scenario,
            llm=self.user_model,
            api_key=self.user_api_key,
            base_url=self.user_base_url,
        )

        orchestrator = Orchestrator(
            "airline",
            agent=agent,
            user=user,
            environment=environment,
            task=task,
        )

        simulation = await orchestrator.run()

        reward_info = evaluate_simulation(
            task=task,
            simulation=simulation,
            evaluation_type="all",
            llm=self.eval_model,
            api_key=self.eval_api_key,
            base_url=self.eval_base_url
        )
        
        messages = orchestrator.get_trajectory()
        traj_records = agent.records
        
        # Calculate reward based on type (same logic as RL workflow)
        if self.reward_type == "db":
            reward = reward_info.db_check.db_reward if reward_info.db_check is not None else 0
        elif self.reward_type == "all":
            try:
                reward = reward_info.db_check.db_reward
                if len(reward_info.action_checks) > 0:
                    reward *= np.mean([x.action_reward for x in reward_info.action_checks])
                if len(reward_info.nl_assertions) > 0:
                    reward *= np.mean([x.met for x in reward_info.nl_assertions])
            except Exception as e:
                print("[debug] reward info: ", e, reward_info)
                reward = 0
        else:
            raise NotImplementedError

        return messages, traj_records, reward

    async def arun_episode(self, engine: InferenceEngine, raw_data=None):
        """
        Run one episode of agent-environment interaction.
        For SFT evaluation mode, this returns None (no training data needed).
        """
        data = copy.deepcopy(raw_data)
        if data is None:
            tasks = get_tasks()
            task = random.choice(tasks)
        else:
            data['evaluation_criteria'] = json.loads(data['evaluation_criteria'])
            task = Task.model_validate(data)

        trajs = await asyncio.gather(*[self.collect_agent_trajectory(engine, task) for _ in range(self.n_trajs)])
        version = engine.get_version()

        # Collect rewards for logging
        rewards = [score for _, _, score in trajs]
        
        # Log evaluation results
        if self.dump_dir is not None:
            os.makedirs(os.path.join(self.dump_dir, str(version)), exist_ok=True)
            file_id = data.get('id', 'random_task') if data is not None else 'random_task'
            with open(
                os.path.join(self.dump_dir, str(version), f"{file_id}_{uuid.uuid4().hex}.jsonl"), "w"
            ) as f:
                for i, (messages, _, score) in enumerate(trajs):
                    eval_result = {
                        "messages": messages,
                        "reward": score,
                        "traj_idx": i,
                        "task_info": {
                            "task_id": str(task.id) if hasattr(task, 'id') else None,
                            "reward_type": self.reward_type,
                        }
                    }
                    f.write(json.dumps(eval_result, ensure_ascii=False) + "\n")
        
        # Log reward statistics
        if len(rewards) > 0:
            avg_reward = np.mean(rewards)
            max_reward = np.max(rewards)
            min_reward = np.min(rewards)
            print(f"[Eval] Task {task.id}: avg_reward={avg_reward:.3f}, max={max_reward:.3f}, min={min_reward:.3f}, n_trajs={len(rewards)}")
        
        # Return None as we don't need training data for SFT evaluation
        return None


def get_tau2_sft_dataset(
    path: str,
    split: str,
    tokenizer,
    rank: int,
    world_size: int,
    max_length: Optional[int] = None,
):
    # Load raw JSON data
    import json
    with open(path, 'r', encoding='utf-8') as f:
        raw_data = [json.loads(line) for line in f]
    
    # Get tools definition for chat template (same as used during inference)
    from tau2_train.domains.airline.environment import get_environment
    environment = get_environment()
    tools = [tool.openai_schema for tool in environment.get_tools()]

    # Process data directly to avoid PyArrow type inference issues
    processed_data = []
    for sample in raw_data:
        try:
            messages = sample["messages"]
            answer = sample["answer"]  # Can be dict or string
            messages = [safe_serialize(m) for m in messages]

            # Answer is already in OpenAI format (dict with role, content, tool_calls)
            # Serialize it if needed
            answer_msg = safe_serialize(answer) if isinstance(answer, dict) else {"role": "assistant", "content": answer}
            
            full_messages = messages + [answer_msg]
            full_sequence = tokenizer.apply_chat_template(
                full_messages,
                tokenize=True,
                add_generation_prompt=False,
                tools=tools,  # Add tools definition for consistent training/inference
            )
            
            # Generate prompt-only tokens (without the final assistant reply)
            prompt_sequence = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,  # Add generation prompt for assistant reply
                tools=tools,  # Add tools definition for consistent training/inference
            )
            
            # Create loss mask: 0 means no loss (prompt part), 1 means compute loss (answer part)
            prompt_length = len(prompt_sequence)
            total_length = len(full_sequence)
            loss_mask = [0] * prompt_length + [1] * (total_length - prompt_length)
            
            # Skip if too long
            if max_length is not None and total_length > max_length:
                continue
            
            processed_data.append({
                "input_ids": full_sequence,
                "loss_mask": loss_mask
            })
        except Exception as e:
            print(f"[Warning] Failed to process sample: {e}")
            continue
    
    # Create dataset from processed data
    from datasets import Dataset
    if len(processed_data) > 0:
        dataset = Dataset.from_dict({
            "input_ids": [item["input_ids"] for item in processed_data],
            "loss_mask": [item["loss_mask"] for item in processed_data]
        })
    else:
        dataset = Dataset.from_dict({"input_ids": [], "loss_mask": []})

    dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
    return dataset


def collate_fn_dict_padding(batch):
    if not batch:
        return {}
    
    # Find max length in this batch
    max_length = max(len(item["input_ids"]) for item in batch)
    
    # Pad input_ids and loss_mask
    padded_input_ids = []
    padded_loss_mask = []
    attention_mask = []
    
    for item in batch:
        input_ids = item["input_ids"]
        loss_mask = item["loss_mask"]
        seq_len = len(input_ids)
        
        # Convert to tensor if not already
        if not torch.is_tensor(input_ids):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        if not torch.is_tensor(loss_mask):
            loss_mask = torch.tensor(loss_mask, dtype=torch.long)
        
        # Pad sequences
        pad_len = max_length - seq_len
        padded_input_ids.append(
            torch.nn.functional.pad(input_ids, (0, pad_len), value=0)
        )
        padded_loss_mask.append(
            torch.nn.functional.pad(loss_mask, (0, pad_len), value=0)
        )
        
        # Create attention mask: 1 for real tokens, 0 for padding
        attn_mask = [1] * seq_len + [0] * pad_len
        attention_mask.append(attn_mask)
    
    # Stack into batched tensors and return as plain dict
    return {
        "input_ids": torch.stack(padded_input_ids),
        "loss_mask": torch.stack(padded_loss_mask),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
    }


def get_tau2_eval_dataset(dataset_path, tokenizer, rank, world_size):
    """Load evaluation dataset and split across nodes (same as workflow_megatron.py)"""
    dataset = load_dataset(
        path="json",
        split="train",
        data_files=dataset_path,
    )
    return split_dataset_by_node(dataset, rank=rank, world_size=world_size)    


@dataclass  
class TAU2SFTMegatronConfig(BaseExperimentConfig):
    # Model config (using experimental Megatron engine)
    model: TrainEngineConfig = field(default_factory=TrainEngineConfig)
    
    # Training config
    async_training: bool = field(default=True, metadata={"help": "Enable async training"})
    
    # Environment evaluation control
    enable_env_eval: bool = field(
        default=False,
        metadata={"help": "Whether to enable environment-based evaluation during training"}
    )
    
    # Rollout and generation config for environment evaluation (only used if enable_env_eval=True)
    rollout: Optional[InferenceEngineConfig] = field(
        default=None,
        metadata={"help": "Rollout inference engine config for environment evaluation"}
    )
    
    gconfig: Optional[GenerationHyperparameters] = field(
        default=None,
        metadata={"help": "Generation hyperparameters for environment evaluation"}
    )
    
    # Environment evaluation specific fields
    reward_type: str = field(
        default="db",
        metadata={"help": "Reward type for environment evaluation (e.g., 'db', 'nl')"}
    )
    
    n_trajs: int = field(
        default=4,
        metadata={"help": "Number of trajectories to collect per task for evaluation"}
    )
    
    user_model: str = field(
        default="/storage/openpsi/models/Qwen__Qwen2.5-72B-Instruct/",
        metadata={"help": "Model name for user simulation."}
    )

    user_api_key: str = field(
        default="empty",
        metadata={"help": "api_key for user simulation model"}
    )

    user_base_url: str = field(
        default="http://33.180.164.231:30000/v1/",
        metadata={"help": "base_url for user simulation model"}
    )

    max_turns: int = field(
        default=100,
        metadata={"help": "maximum number of turns for search agent"}
    )

    max_context_length: int = field(
        default=32768,
        metadata={"help": "Maximum context length of the trained model"}
    )


def main(args):
    config, _ = load_expr_config(args, TAU2SFTMegatronConfig)
    config: TAU2SFTMegatronConfig
    
    # Save config
    log_path = StatsLogger.get_log_path(config.stats_logger)
    with open(os.path.join(log_path, "config.yaml"), 'w') as file:
        yaml.dump(asdict(config), file, default_flow_style=False, sort_keys=True)

    rank = int(os.getenv("RANK"))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    seeding.set_random_seed(config.seed, key=f"trainer{rank}")
    
    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    parallel_strategy = allocation_mode.train

    # Initialize Megatron LM engine
    engine = MegatronLMEngine(config=config.model)
    engine.create_process_group(parallel_strategy=parallel_strategy)

    # Load datasets
    train_dataset = get_tau2_sft_dataset(
        path=config.train_dataset.path,
        split="train",
        tokenizer=tokenizer,
        rank=mpu.get_data_parallel_rank(),
        world_size=mpu.get_data_parallel_world_size(),
        max_length=config.train_dataset.max_length,
    )
    
    # Environment evaluation dataset (raw JSON for task execution)
    valid_dataset_env = get_tau2_eval_dataset(
        config.valid_dataset.path,
        tokenizer,
        mpu.get_data_parallel_rank(),
        mpu.get_data_parallel_world_size(),
    )

    # Create dataloaders
    batch_size = config.train_dataset.batch_size // mpu.get_data_parallel_world_size()
    train_dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=config.train_dataset.shuffle,
        num_workers=config.train_dataset.num_workers,
        collate_fn=collate_fn_dict_padding,
        drop_last=config.train_dataset.drop_last,
    )
    
    # Environment evaluation dataloader (for task-based evaluation)
    valid_dataloader_env = StatefulDataLoader(
        valid_dataset_env,
        batch_size=1,
        shuffle=False,
        num_workers=config.valid_dataset.num_workers,
        collate_fn=lambda x: x,  # Keep raw data format
        drop_last=False,
    )
    
    # Create ft_spec after dataloaders
    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(train_dataloader) * config.train_dataset.batch_size,
        train_batch_size=config.train_dataset.batch_size,
    )

    # Initialize engine
    engine.initialize(
        None, ft_spec, parallel_strategy=parallel_strategy, seed=config.seed
    )

    # Initialize inference engine for evaluation (only if enabled)
    eval_rollout = None
    if config.enable_env_eval and config.rollout is not None:
        try:
            # Configure rollout for evaluation (no offpolicyness control)
            config.rollout.max_head_offpolicyness = int(1e12)
            eval_rollout = RemoteSGLangEngine(config.rollout)
            # Initialize with train_data_parallel_size for proper distributed inference
            eval_rollout.initialize(train_data_parallel_size=parallel_strategy.dp_size)
            print("[Info] Environment evaluation enabled")
        except Exception as e:
            print(f"[Warning] Failed to initialize evaluation rollout: {e}")
            print("[Info] Continuing training without environment evaluation")
    else:
        print("[Info] Environment evaluation disabled")

    # Weight update meta (for eval rollout)
    weight_update_meta = None
    if eval_rollout is not None:
        try:
            weight_update_meta = WeightUpdateMeta.from_disk(
                config.experiment_name,
                config.trial_name,
                config.cluster.fileroot,
            )
        except Exception as e:
            print(f"[Warning] Failed to load weight_update_meta: {e}")

    # Create rollout workflow for environment-based evaluation (only if enabled)
    eval_workflow = None
    if config.enable_env_eval and eval_rollout is not None and config.gconfig is not None:
        try:
            # Configure generation parameters
            if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
                config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
            if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
                config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)
                
            eval_workflow = Tau2Workflow(
                gconfig=config.gconfig,
                tokenizer=tokenizer,
                dump_dir=os.path.join(
                    StatsLogger.get_log_path(config.stats_logger), "generated-eval"
                ),
                max_num_turns=config.max_turns,
                n_trajs=4,  # Use fewer trajectories for SFT eval
                user_model=config.user_model,
                user_api_key=config.user_api_key,
                user_base_url=config.user_base_url,
                max_context_length=config.max_context_length,
                reward_type="db",  # Use simple db reward for SFT eval
            )
            print("[Info] Environment evaluation workflow created")
        except Exception as e:
            print(f"[Warning] Failed to create evaluation workflow: {e}")
            eval_workflow = None

    # Run training
    saver = Saver(config.saver, ft_spec)
    stats_logger = StatsLogger(config.stats_logger, ft_spec)
    evaluator = Evaluator(config.evaluator, ft_spec)

    recover_handler = RecoverHandler(config.recover, ft_spec)
    recover_info = recover_handler.load(
        engine,
        saver,
        evaluator,
        stats_logger,
        train_dataloader,
    )
    start_step = (
        recover_info.last_step_info.next().global_step
        if recover_info is not None
        else 0
    )

    total_epochs = config.total_train_epochs
    steps_per_epoch = len(train_dataloader)
    max_steps = total_epochs * steps_per_epoch

    global_step = 0
    
    # Initialize eval_rollout version if evaluation is enabled
    if eval_rollout is not None:
        eval_rollout.set_version(start_step)
        if mpu.get_data_parallel_rank() == 0:
            print(f"[Info] Initialized eval_rollout version to {start_step}")
    
    for epoch in range(total_epochs):
        for step, data in enumerate(train_dataloader):
            if global_step < start_step:
                global_step += 1
                continue
            
            # data is already a dict with torch.Tensor values from collate_fn
            # No need to convert - AReaL's tensor_container_to handles dict natively
                
            step_info = StepInfo(
                global_step=global_step,
                epoch=epoch,
                epoch_step=step,
                steps_per_epoch=steps_per_epoch,
            )

            # Evaluation (frequency controlled by evaluator config)
            if eval_rollout is not None and eval_workflow is not None:
                dist.barrier(device_ids=[engine.device.index])
                
                with stats_tracker.record_timing("eval"):
                    def evaluate_fn():
                        # Run evaluation with timeout and error handling
                        cnt = 0
                        try:
                            if mpu.get_data_parallel_rank() == 0:
                                print(f"[Info] Starting evaluation at global_step={global_step}")
                            for data_batch in valid_dataloader_env:
                                for item in data_batch:
                                    eval_rollout.submit(item, eval_workflow)
                                    cnt += 1
                            if mpu.get_data_parallel_rank() == 0:
                                print(f"[Info] Submitted {cnt} evaluation tasks")
                            if cnt > 0:
                                if mpu.get_data_parallel_rank() == 0:
                                    print(f"[Info] Waiting for {cnt} evaluation tasks to complete...")
                                eval_rollout.wait(cnt, timeout=600)  # 10 minutes timeout
                                if mpu.get_data_parallel_rank() == 0:
                                    print(f"[Info] Completed evaluation for {cnt} tasks")
                            else:
                                print("[Warning] No evaluation tasks submitted")
                        except TimeoutError:
                            print(f"[Error] Evaluation timeout after 10 minutes for {cnt} tasks")
                        except Exception as e:
                            print(f"[Error] Evaluation failed with exception: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    if engine.is_data_parallel_head():
                        evaluator.evaluate(
                            evaluate_fn,
                            epoch,
                            step,
                            global_step,
                        )
                        
                dist.barrier(device_ids=[engine.device.index])
                current_platform.synchronize()

            # Training step
            # NOTE: data are identical across model+context parallel group
            # data is a dict, use tensor_container_to instead of .to()
            from areal.utils.data import tensor_container_to
            data = tensor_container_to(data, current_platform.current_device())
            data = broadcast_tensor_container(
                data,
                src_rank=engine.current_data_parallel_head(),
                group=engine.context_and_model_parallel_group,
            )

            with (
                stats_tracker.record_timing("train_step"),
                stats_tracker.scope("sft"),
            ):
                stats = engine.train_lm(data)
                engine.step_lr_scheduler()
                stats_tracker.scalar(**stats)

            # Update eval rollout version and weights after training step
            if eval_rollout is not None and weight_update_meta is not None:
                eval_rollout.pause()
                
                with stats_tracker.record_timing("update_weights"):
                    if dist.get_rank() == 0:
                        future = eval_rollout.update_weights(weight_update_meta)
                    engine.upload_weights(weight_update_meta)
                    if dist.get_rank() == 0:
                        future.result()
                    dist.barrier(device_ids=[engine.device.index])
                    current_platform.synchronize()
                    
                    engine.set_version(global_step + 1)
                    eval_rollout.set_version(global_step + 1)
                    
                eval_rollout.resume()

            with stats_tracker.record_timing("save"):
                saver.save(engine, epoch, step, global_step, tokenizer=tokenizer)

            with stats_tracker.record_timing("checkpoint_for_recover"):
                recover_handler.dump(
                    engine,
                    step_info,
                    saver,
                    evaluator,
                    stats_logger,
                    train_dataloader,
                    tokenizer=tokenizer,
                )

            dist.barrier(device_ids=[engine.device.index])
            current_platform.synchronize()

            stats_logger.commit(
                epoch,
                step,
                global_step,
                stats_tracker.export_all(reduce_group=mpu.get_data_parallel_group()),
            )
            global_step += 1

    stats_logger.close()
    
    # Clean up eval rollout resources
    if eval_rollout is not None:
        eval_rollout.destroy()
    
    engine.destroy()
    engine.destroy_process_groups()


if __name__ == "__main__":
    main(sys.argv[1:])

