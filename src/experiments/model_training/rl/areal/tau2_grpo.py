import asyncio
import itertools
import json
import os
import re
import sys
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import colorama
import torch
import torch.distributed as dist
from datasets import Dataset
from datasets.distributed import split_dataset_by_node
from loguru import logger
from tau2.data_model.message import AssistantMessage as Tau2AssistantMessage
from tau2.data_model.tasks import Task
from tau2.environment.tool import Tool
from tau2.gym.gym_agent import AgentGymEnv
from tau2.registry import registry
from tau2.utils.tools import parse_action_string
from tensordict import TensorDict
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters, GRPOConfig, load_expr_config
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import (
    AllocationMode,
    FinetuneSpec,
    ModelRequest,
    StepInfo,
    WeightUpdateMeta,
)
from areal.api.workflow_api import RolloutWorkflow
from areal.engine.ppo.actor import FSDPPPOActor
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.utils.data import concat_padded_tensors
from areal.utils.device import log_gpu_stats
from areal.utils.evaluator import Evaluator
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger
from realhf.api.core.data_api import load_hf_tokenizer
from realhf.base import seeding, stats_tracker

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
    rank: int,
    world_size: int,
    split: str = "train",
    type: str = "rl",
    tokenizer: Optional["PreTrainedTokenizerFast"] = None,
) -> Dataset:
    """Create a HuggingFace Dataset from tau2 task IDs.

    Args:
        domain: The tau2 domain name, e.g., 'retail', 'airline', 'telecom'
        rank: Current process rank for distributed training
        world_size: Total number of processes
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
    dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
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


class Tau2Workflow(RolloutWorkflow):
    def __init__(
        self,
        econfig: Tau2EnvConfig,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        enable_thinking: bool = False,
        dump_dir: str | None = None,
    ):
        super().__init__()
        self.econfig = econfig
        self.gconfig = gconfig
        self.tokenizer = tokenizer
        self.enable_thinking = enable_thinking
        self.dump_dir = dump_dir

    async def _run_one_episode(
        self, engine: InferenceEngine, data: Dict[str, Any], rid: str
    ):
        task_id = data["task_id"]
        env = AgentGymEnv(
            domain=self.econfig.domain,
            task_id=task_id,
            max_steps=self.econfig.max_steps,
            solo_mode=self.econfig.solo_mode,
            user_llm=self.econfig.user_llm,
            user_llm_args=self.econfig.user_llm_args,
        )
        obs, info = env.reset()
        task: Task = info["task"]
        tools: list[Tool] = info["tools"]
        tools_schema = [tool.openai_schema for tool in tools]
        agent_policy_doc = info["policy"]

        system_prompt = get_tau2_system_prompt(
            agent_policy_doc, task.ticket if self.econfig.solo_mode else None
        )
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        if obs:
            messages.append({"role": "user", "content": obs})
        # Convert the prompt into input_ids
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tools=tools_schema,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            # TODO: shall we support thinking? Or add a thinking tool?
        )
        dummy_message = {"role": "assistant", "content": "dummy message"}
        dummy_input_ids = self.tokenizer.apply_chat_template(
            [dummy_message],
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )

        seq, logprobs, loss_mask, versions, rewards = [], [], [], [], []
        discount = 1.0
        for step in range(self.econfig.max_steps):
            # Send generate request to get the response.
            req = ModelRequest(
                rid=rid,
                input_ids=input_ids,
                gconfig=self.gconfig.new(n_samples=1),
                tokenizer=self.tokenizer,
            )
            resp = await engine.agenerate(req)
            # extract action from response
            prompt_str = self.tokenizer.decode(input_ids)
            completions_str = self.tokenizer.decode(resp.output_tokens)
            # print("-" * 100)
            # print(prompt_str)
            # print("completions_str:")
            # print(completions_str)

            index = completions_str.rfind(self.tokenizer.eos_token)
            if index != -1:
                completions_str = completions_str[:index]

            action = Tau2AssistantMessage(role="assistant", content=completions_str)
            action_str = completions_str
            tool_call_err_msg = None
            tool_call_default_err_msg = "Your tool call format is invalid, please use <tool_call>...</tool_call> to call a tool, and include both name and arguments."
            if "<tool_call>" in completions_str or "</tool_call>" in completions_str:
                # is a tool call
                # try to find the first <tool_call> and </tool_call>
                pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
                match = pattern.search(completions_str)
                if match:
                    # take the first tool call
                    tool_call_str = match.group(1).strip()
                    # FIXME: this leads to the message and the outputs are mismatched.
                    try:
                        parsed_action = parse_action_string(tool_call_str)
                    except Exception as e:
                        logger.error(f"Error parsing tool call: {e}")
                        tool_call_err_msg = str(e)
                        continue

                    if parsed_action.tool_calls is not None:
                        action = parsed_action
                        action_str = tool_call_str
                    else:
                        tool_call_err_msg = tool_call_default_err_msg
                else:
                    tool_call_err_msg = tool_call_default_err_msg
            else:
                # TODO: directly treat it as a message, not a tool call
                pass

            if tool_call_err_msg is not None:
                obs = tool_call_err_msg
                # TODO: add reward penalty for invalid format
                # reward = self.econfig.invalid_format_penalty
                reward = 0.0
                done = False
            else:
                # print("-" * 100)
                # print(f"step: {step}")
                # print(f"action: {action_str}")
                obs, reward, done, _, _ = env.step(action_str)

            # Amend results
            input_len = len(resp.input_tokens) - len(seq)
            prefix_match = len(seq) == 0 or resp.input_tokens[: len(seq)] == seq
            msg = "\n".join(
                [
                    f"step: {step}",
                    f"[decoded seq]\n{self.tokenizer.decode(seq)[-1000:]}",
                    f"[decoded input_tokens after seq]\n{self.tokenizer.decode(resp.input_tokens[len(seq) :])}",
                    f"[decoded resp.input_tokens]\n{self.tokenizer.decode(resp.input_tokens)[-1000:]}",
                    f"[decoded resp.output_tokens]\n{self.tokenizer.decode(resp.output_tokens)[-1000:]}",
                    f"len(seq): {len(seq)}",
                    f"len(resp.input_tokens): {len(resp.input_tokens)}",
                    f"len(resp.output_tokens): {len(resp.output_tokens)}",
                ]
            )
            assert prefix_match, msg
            seq += resp.input_tokens[len(seq) :] + resp.output_tokens
            logprobs += [0.0] * input_len + resp.output_logprobs
            loss_mask += [0] * input_len + [1] * resp.output_len
            versions += [-1] * input_len + resp.output_versions
            # rewards += [reward * discount] * (input_len + resp.output_len)

            messages.append(to_dict(action))
            if obs:
                new_msg_role = "tool" if action.tool_calls is not None else "user"
                new_msg = {"role": new_msg_role, "content": obs}
                messages.append(new_msg)

                input_ids += resp.output_tokens
                if resp.output_tokens[-1] != self.tokenizer.eos_token_id:
                    input_ids += [self.tokenizer.eos_token_id]
                new_obs_ids = self.tokenizer.apply_chat_template(
                    [dummy_message, new_msg],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                # TODO: remove system message from new_obs_ids
                input_ids += new_obs_ids[len(dummy_input_ids) :]

            if done:
                break
            discount *= self.econfig.turn_discount

        res = dict(
            input_ids=torch.tensor(seq),
            logprobs=torch.tensor(logprobs),
            loss_mask=torch.tensor(loss_mask),
            versions=torch.tensor(versions),
            rewards=torch.tensor(float(reward * discount)),  # TODO: use rewards
            attention_mask=torch.ones(len(seq), dtype=torch.bool),
        )
        res = {k: v.unsqueeze(0) for k, v in res.items()}
        return (
            TensorDict(res, batch_size=[1]),
            messages,
            prompt_str,
            completions_str,
            reward,
            len(seq),
        )

    async def arun_episode(self, engine: InferenceEngine, data):
        rid = uuid.uuid4().hex
        tasks = [
            self._run_one_episode(engine, data, rid)
            for _ in range(self.gconfig.n_samples)
        ]
        results = await asyncio.gather(*tasks)

        if self.dump_dir is not None:
            version = engine.get_version()
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
                for i, (
                    _,
                    messages,
                    prompt_str,
                    completions_str,
                    reward,
                    seq_len,
                ) in enumerate(results):
                    info = f"idx: {i + 1} / {n_samples}, seqlen: {seq_len}, reward is {reward}.\n"
                    info += colored_text(f"[prompt]\n{prompt_str}\n", "cyan")
                    info += colored_text(f"[completions]\n{completions_str}\n", "white")
                    for j, message in enumerate(messages):
                        role = message["role"]
                        content = message["content"] or ""
                        if "tool_calls" in message:
                            content += "\n[TOOL_CALLS]\n" + json.dumps(
                                message["tool_calls"]
                            )
                        content = f"[{j}][{role}]: {content}"
                        if message["role"] == "system":
                            info += f"{colored_text(content, 'yellow')}\n"
                        elif message["role"] == "user":
                            info += f"{colored_text(content, 'green')}\n"
                        elif message["role"] == "assistant":
                            info += f"{colored_text(content, 'blue')}\n"
                        elif message["role"] == "tool":
                            info += f"{colored_text(content, 'magenta')}\n"
                        else:
                            raise ValueError(f"Invalid role: {message['role']}")
                    f.write(info)

        data = [res[0] for res in results]
        return concat_padded_tensors(data)


# ================================ main ================================
def main(args):
    config, _ = load_expr_config(args, Tau2GRPOConfig)

    logger.remove()
    logger.add(sys.stdout, level="WARNING")
    logger.add(
        os.path.join(StatsLogger.get_log_path(config.stats_logger), "tau2.log"),
        level="INFO",
    )

    rank = int(os.getenv("RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    seeding.set_random_seed(config.seed, key=f"trainer{rank}")

    train_dataset = get_tau2_dataset(
        domain=config.econfig.domain,
        rank=rank,
        world_size=world_size,
        split=config.train_dataset.path.split("/")[-1],
        type=config.train_dataset.type,
        tokenizer=tokenizer,
    )
    valid_dataset = get_tau2_dataset(
        domain=config.econfig.domain,
        rank=rank,
        world_size=world_size,
        split=config.valid_dataset.path.split("/")[-1],
        type=config.valid_dataset.type,
        tokenizer=tokenizer,
    )

    # print(f"train_dataset: {len(train_dataset)}")
    # print(f"valid_dataset: {len(valid_dataset)}")

    # Create dataset and dataloaders
    train_dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=config.train_dataset.batch_size // world_size,
        shuffle=config.train_dataset.shuffle,
        num_workers=config.train_dataset.num_workers,
        collate_fn=lambda x: x,
        drop_last=config.train_dataset.drop_last,
    )
    valid_dataloader = StatefulDataLoader(
        valid_dataset,
        batch_size=config.valid_dataset.batch_size // world_size,
        shuffle=config.valid_dataset.shuffle,
        num_workers=config.valid_dataset.num_workers,
        collate_fn=lambda x: x,
        drop_last=config.valid_dataset.drop_last,
    )
    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(train_dataloader) * config.train_dataset.batch_size,
        train_batch_size=config.train_dataset.batch_size,
    )

    # Initialize inference engine
    rollout = RemoteSGLangEngine(config.rollout)
    rollout.initialize(None, ft_spec)
    eval_rollout = RemoteSGLangEngine(config.rollout)
    eval_rollout.initialize(None, ft_spec)
    # NOTE: set a large version such that eval does not have any offpolicyness control
    eval_rollout.set_version(int(1e12))

    # Initialize train engine
    actor = FSDPPPOActor(config=config.actor)
    actor.initialize(None, ft_spec)
    ref = None
    if config.actor.kl_ctl > 0 and config.ref is not None:
        ref = FSDPPPOActor(config=config.ref)
        ref.initialize(None, ft_spec)

    # NOTE: Weight update meta only requires address and free port of rank 0,
    # but `WeightUpdateMeta.from_fsdp_nccl` has to be executed on all ranks
    # due to `engine.get_param_specs()`.
    # Therefore, we create weight update meta on all ranks, then broadcast the one on rank 0.
    weight_update_meta = [
        WeightUpdateMeta.from_fsdp_nccl(
            AllocationMode.from_str(config.allocation_mode), actor
        )
    ]
    dist.broadcast_object_list(weight_update_meta, src=0)
    weight_update_meta = weight_update_meta[0]

    # Create rollout workflow
    if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)

    # Customized workflow for tau2
    workflow = Tau2Workflow(
        econfig=config.econfig,
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

    data_generator = itertools.cycle(train_dataloader)
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
            if config.async_training:
                batch = rollout.prepare_batch(train_dataloader, workflow=workflow)
            else:
                batch = rollout.rollout_batch(next(data_generator), workflow=workflow)

        batch = batch.to(actor.device)
        # Create barrier to synchronize all rollout processes.
        dist.barrier(device_ids=[actor.device.index])
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
            saver.save(actor, epoch, step, global_step, tokenizer=tokenizer)

        with stats_tracker.record_timing("eval"):

            def evaluate_fn():
                rollout.pause()
                cnt = 0
                for data in valid_dataloader:
                    for item in data:
                        eval_rollout.submit(item, workflow)
                        cnt += 1
                batch = eval_rollout.wait(cnt, timeout=None)
                rewards = batch["rewards"].float().to(actor.device)
                with stats_tracker.scope("grpo-eval"):
                    stats_tracker.denominator(
                        n_seqs=torch.ones(
                            rewards.shape[0],
                            device=rewards.device,
                            dtype=torch.bool,
                        )
                    )
                    stats_tracker.stat(task_reward=rewards, denominator="n_seqs")
                rollout.resume()

            evaluator.evaluate(
                evaluate_fn,
                epoch,
                step,
                global_step,
            )

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
    eval_rollout.destroy()
    rollout.destroy()
    if ref is not None:
        ref.destroy()
    actor.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])
