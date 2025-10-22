import asyncio
import json
import os
import re
import sys
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import colorama
import torch
import torch.distributed as dist
from datasets import Dataset
from litellm import ModelResponse
from loguru import logger
from tau2.agent.llm_agent import LLMAgent, LLMSoloAgent
from tau2.data_model.message import AssistantMessage as Tau2AssistantMessage
from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment
from tau2.environment.tool import Tool
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.registry import registry
from tau2.user.user_simulator import DummyUser, UserSimulator

# from tau2.utils.tools import parse_action_string
# from tensordict import TensorDict
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import (
    DatasetConfig,
    GenerationHyperparameters,
    GRPOConfig,
    load_expr_config,
)
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import FinetuneSpec, ModelRequest, StepInfo, WeightUpdateMeta
from areal.api.workflow_api import RolloutWorkflow
from areal.engine.ppo.actor import FSDPPPOActor
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.experimental.openai.client import ArealOpenAI
from areal.platforms import current_platform
from areal.utils import seeding, stats_tracker
from areal.utils.data import (
    broadcast_tensor_container,
    concat_padded_tensors,
    cycle_dataloader,
    tensor_container_to,
)
from areal.utils.dataloader import create_dataloader
from areal.utils.device import log_gpu_stats
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger

# ================================ prompts ================================
STOP_FUNCTION_NAME = "done"
TAU2_AGENT_INSTRUCTION_SOLO = f"""
You are a customer service agent that helps the user according to the <policy> provided below.
You will be provided with a ticket that contains the user's request.
You will need to plan and call the appropriate tools to solve the ticket.

You cannot communicate with the user, only make tool calls.
Stop when you consider that you have solved the ticket.
To do so, send a message containing a single tool call to the `{STOP_FUNCTION_NAME}` tool. Do not include any other tool calls in this last message.

Always follow the policy.
""".strip()

TAU2_SYSTEM_PROMPT_SOLO = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
<ticket>
{ticket}
</ticket>
""".strip()

TAU2_AGENT_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy.
""".strip()

TAU2_SYSTEM_PROMPT = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
""".strip()

TAU2_FORMAT_INSTRUCTION = """
First, you MUST carefully reflect on the history of interactions. Then, reason about what should be done next, which tool to call, what arguments to use. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reflexion and reasoning, you present the tool call as a valid JSON within <action> </action> tags, for example: <action>{"name": "calculate", "arguments": {"expression": "1+2"}}</action>.
""".strip()


def get_tau2_system_prompt(domain_policy: str, ticket: Optional[str] = None):
    if ticket is not None:
        # solo mode
        agent_instruction = TAU2_AGENT_INSTRUCTION_SOLO
        return TAU2_SYSTEM_PROMPT_SOLO.format(
            agent_instruction=agent_instruction,
            domain_policy=domain_policy,
            ticket=ticket,
        )
    else:
        agent_instruction = TAU2_AGENT_INSTRUCTION
        return TAU2_SYSTEM_PROMPT.format(
            agent_instruction=agent_instruction, domain_policy=domain_policy
        )


# ================================ config ================================
# Customized config for tau2, add env config
@dataclass
class Tau2EnvConfig:
    domain: str = field(
        default="telecom",
        metadata={
            "help": "The tau2 domain name, e.g., 'retail', 'airline', 'telecom'."
        },
    )
    max_steps: int = field(
        default=100, metadata={"help": "Maximum number of steps per episode."}
    )
    solo_mode: bool = field(
        default=False, metadata={"help": "Whether to use solo mode."}
    )
    user_llm: Optional[str] = field(
        default=None,
        metadata={"help": "The user LLM to use, default to the gpt-4.1 model."},
    )
    user_llm_args: Optional[dict] = field(
        default=None, metadata={"help": "The arguments for the user LLM."}
    )
    turn_discount: float = field(
        default=1.0, metadata={"help": "Discount factor for turn-based learning."}
    )
    invalid_format_penalty: float = field(
        default=0.1, metadata={"help": "Penalty for invalid format in completions."}
    )


@dataclass
class Tau2GRPOConfig(GRPOConfig):
    econfig: Tau2EnvConfig = field(default_factory=Tau2EnvConfig)


# ================================ dataset ================================
def get_tau2_dataset(
    domain: str,
    type: str = "rl",
    split: str = "train",
    tokenizer: Optional["PreTrainedTokenizerFast"] = None,
) -> Dataset:
    """Create a HuggingFace Dataset from tau2 task IDs.

    Args:
        domain: The tau2 domain name, e.g., 'retail', 'airline', 'telecom'
        split: Dataset split (e.g., 'train', 'test')
        type: Dataset type (e.g., 'rl', 'sft'), only 'rl' is supported for now
        tokenizer: Tokenizer (currently unused, for future compatibility)

    Returns:
        Dataset: HuggingFace Dataset containing task_id entries
    """
    assert type == "rl", "Only RL dataset is supported for now"
    # TODO: support SFT dataset

    splits_loader_fn = registry.get_task_splits_loader(domain)
    if splits_loader_fn is None:
        raise ValueError(f"No task splits loader found for domain {domain}")
    splits = splits_loader_fn()
    if split not in splits:
        raise ValueError(
            f"Split {split} not found in {splits}, available splits: {splits.keys()}"
        )
    task_ids = splits[split]
    # print(f"domain: {domain}, split: {split}, task_ids: {task_ids}")

    dataset_items = [{"task_id": task_id} for task_id in task_ids]
    dataset = Dataset.from_list(dataset_items)
    return dataset


# ================================ workflow ================================
# utils
COLOR_MAP = {
    "yellow": colorama.Fore.YELLOW,
    "red": colorama.Fore.RED,
    "green": colorama.Fore.GREEN,
    "blue": colorama.Fore.BLUE,
    "magenta": colorama.Fore.MAGENTA,
    "cyan": colorama.Fore.CYAN,
    "white": colorama.Fore.WHITE,
}


def colored_text(text: str, color: str):
    if color not in COLOR_MAP:
        raise ValueError(f"Invalid color: {color}")
    return f"{COLOR_MAP[color] + colorama.Style.DIM}{text}{colorama.Style.RESET_ALL}"


def to_dict(message: Tau2AssistantMessage) -> dict:
    data = {
        "role": "assistant",
        "content": message.content,
    }
    if message.tool_calls is not None:
        data["tool_calls"] = [
            {
                "id": tool_call.id,
                "name": tool_call.name,
                "arguments": tool_call.arguments,
            }
            for tool_call in message.tool_calls
        ]
    return data


def get_tool_call_parser_name(model_name: str) -> str:
    if "qwen" in model_name.lower():
        return "qwen25"
    elif "llama" in model_name.lower():
        return "llama"
    elif "gemma" in model_name.lower():
        return "gemma"
    elif "deepseek" in model_name.lower():
        return "deepseek"
    else:
        raise ValueError(f"Unknown model name: {model_name}")


class Tau2Workflow(RolloutWorkflow):
    def __init__(
        self,
        econfig: Tau2EnvConfig,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        actor_model_name: str,
        dump_dir: str | None = None,
        rollout_stat_scope: str = "rollout",
        n_samples: int = 1,
        max_total_tokens: int = 32768,
        max_turns: int = 8,
    ):
        self.econfig = econfig
        self.gconfig = gconfig
        self.tokenizer = tokenizer
        self.actor_model_name = actor_model_name
        self.tool_call_parser_name = get_tool_call_parser_name(actor_model_name)
        self.dump_dir = dump_dir
        self.max_total_tokens = max_total_tokens
        self.rollout_stat_scope = rollout_stat_scope
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

    async def _run_one_episode(self, client: ArealOpenAI, data: Dict[str, Any]):
        domain = self.econfig.domain
        solo_mode = self.econfig.solo_mode

        task_id = data["task_id"]

        def _get_task(task_id: str) -> Task:
            tasks = registry.get_tasks_loader(domain)()
            for task in tasks:
                if task.id == task_id:
                    return task
            raise ValueError(f"No task found with id {task_id} for domain {domain}")

        task = _get_task(task_id)
        logger.info(
            f"STARTING SIMULATION: Domain: {domain}, Task: {task.id}, Solo Mode: {solo_mode}"
        )

        responses: list[ModelResponse] = []

        async def acompletion_areal(
            model: str,
            messages: list[dict],
            tools: list | None = None,
            tool_choice: str | dict | None = None,
            num_retries: int | None = None,
            **kwargs: Any,
        ):
            # ignore `model` and `num_retries` due to this client does not need them
            response = await client.chat.completions.create(
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=self.gconfig.temperature,
                max_completion_tokens=self.gconfig.max_new_tokens,
                top_p=self.gconfig.top_p,
                stop=self.gconfig.stop,
                frequency_penalty=self.gconfig.frequency_penalty,
                **kwargs,
            )
            response.model = f"areal/{self.actor_model_name}"
            responses.append(response)
            return response

        environment_constructor = registry.get_env_constructor(domain)
        environment: Environment = environment_constructor(solo_mode=solo_mode)
        agent_policy_doc = environment.get_policy()
        tools: list[Tool] = environment.get_tools()
        try:
            user_tools = environment.get_user_tools()
        except Exception:
            user_tools = []

        if solo_mode:
            agent = LLMSoloAgent(
                tools=tools + user_tools,
                domain_policy=agent_policy_doc,
                llm=self.actor_model_name,
                llm_args={},
                task=task,
                completion_fn=acompletion_areal,
            )
            user = DummyUser()
        else:
            agent = LLMAgent(
                tools=tools,
                domain_policy=agent_policy_doc,
                llm=self.actor_model_name,
                llm_args={},
                completion_fn=acompletion_areal,
            )
            user = UserSimulator(
                tools=user_tools,
                instructions=str(task.user_scenario),
                llm=self.econfig.user_llm,
                llm_args=self.econfig.user_llm_args,
            )

        orchestrator = Orchestrator(
            domain=domain,
            agent=agent,
            user=user,
            environment=environment,
            task=task,
            max_steps=self.econfig.max_steps,
            # max_errors=self.econfig.max_errors,
            # seed=self.econfig.seed,
            solo_mode=solo_mode,
        )
        simulation = await orchestrator.arun()

        reward_info = evaluate_simulation(
            domain=domain,
            task=task,
            simulation=simulation,
            evaluation_type=EvaluationType.ALL,
            solo_mode=solo_mode,
        )

        simulation.reward_info = reward_info
        for i, response in enumerate(responses):
            if i + 1 < len(responses):
                client.set_reward(response.id, 0)
            else:
                client.set_reward(response.id, reward_info.reward)

        logger.info(
            f"FINISHED SIMULATION: Domain: {domain}, Task: {task.id}, Agent: {agent.__class__.__name__}, User: {user.__class__.__name__}. Reward: {reward_info.reward}"
        )
        return simulation.messages, responses, reward_info.reward

    async def arun_episode(self, engine: InferenceEngine, data: Dict[str, Any]):
        clients = [
            ArealOpenAI(
                engine=engine,
                tokenizer=self.tokenizer,
                tool_call_parser=self.tool_call_parser_name,
            )
            for _ in range(self.gconfig.n_samples)
        ]

        # Collect trajectories
        results = await asyncio.gather(
            *[
                self._run_one_episode(client=clients[i], data=data)
                for i in range(self.gconfig.n_samples)
            ]
        )
        for result in results:
            messages, responses, reward = result
            stats_tracker.get(self.rollout_stat_scope).scalar(
                reward=reward, num_steps=len(responses)
            )
            for msg in messages:
                logger.info(f"Role: {msg['role']}, Content: {msg['content']}")

        completions_with_reward = {}
        for client in clients:
            client.apply_reward_discount(turn_discount=1.0)
            completions = client.export_completions(style="individual")
            completions_with_reward.update(completions)
        return completions_with_reward


def main(args):
    config, _ = load_expr_config(args, Tau2GRPOConfig)
    config: Tau2GRPOConfig
    domain = config.econfig.domain

    logger.remove()
    logger.add(sys.stdout, level="WARNING")
    logger.add(
        os.path.join(StatsLogger.get_log_path(config.stats_logger), "tau2.log"),
        level="INFO",
    )

    rank = int(os.getenv("RANK"))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    seeding.set_random_seed(config.seed, key=f"trainer{rank}")
    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    parallel_strategy = allocation_mode.train
    assert parallel_strategy is not None

    # Initialize train engine
    actor = FSDPPPOActor(config=config.actor)
    actor.create_process_group(parallel_strategy=parallel_strategy)

    def _get_dataset(dataset_config: DatasetConfig) -> Dataset:
        return get_tau2_dataset(
            domain=domain,
            type=dataset_config.type,
            split=dataset_config.path.split("/")[-1],
            tokenizer=tokenizer,
        )

    # Create dataset and dataloaders
    train_dataset = _get_dataset(dataset_config=config.train_dataset)
    valid_dataset = _get_dataset(dataset_config=config.valid_dataset)

    train_dataloader = create_dataloader(
        train_dataset,
        rank=actor.data_parallel_rank,
        world_size=actor.data_parallel_world_size,
        dataset_config=config.train_dataset,
    )
    valid_dataloader = create_dataloader(
        valid_dataset,
        rank=actor.data_parallel_rank,
        world_size=actor.data_parallel_world_size,
        dataset_config=config.valid_dataset,
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

    weight_update_meta = WeightUpdateMeta.from_fsdp_xccl(allocation_mode)

    actor.initialize(None, ft_spec)
    actor.connect_engine(rollout, weight_update_meta)

    ref = None
    if config.actor.kl_ctl > 0 and config.ref is not None:
        ref = FSDPPPOActor(config=config.ref)
        ref.create_process_group(parallel_strategy=parallel_strategy)
        ref.initialize(None, ft_spec)

    # Create rollout workflow
    if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)
    workflow = Tau2Workflow(
        econfig=config.econfig,
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        actor_model_name=config.actor.path.split("/")[-1],
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated"
        ),
    )
    eval_workflow = Tau2Workflow(
        econfig=config.econfig,
        gconfig=config.gconfig.new(temperature=0.6),
        tokenizer=tokenizer,
        actor_model_name=config.actor.path.split("/")[-1],
        rollout_stat_scope="eval-rollout",
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated-eval"
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

        with stats_tracker.record_timing("eval"):

            def evaluate_fn():
                if actor.is_data_parallel_head():
                    cnt = 0
                    for data in valid_dataloader:
                        for item in data:
                            eval_rollout.submit(item, eval_workflow)
                            cnt += 1
                    eval_rollout.wait(cnt, timeout=None)
                dist.barrier(device_ids=[actor.device.index])
                current_platform.synchronize()

            evaluator.evaluate(
                evaluate_fn,
                epoch,
                step,
                global_step,
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
    eval_rollout.destroy()
    rollout.destroy()
    if ref is not None:
        ref.destroy()
    actor.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])
