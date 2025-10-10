import asyncio
import copy
import json
import os
import random
import uuid

import aiofiles
import aiofiles.os
import numpy as np
import torch
from tau2_train.agent import LLMAgent
from tau2_train.data_model.tasks import Task
from tau2_train.domains.airline.environment import get_environment, get_tasks
from tau2_train.evaluator.evaluator import evaluate_simulation
from tau2_train.orchestrator import Orchestrator
from tau2_train.user_simulator import UserSimulator
from tensordict import TensorDict
from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.workflow_api import RolloutWorkflow
from areal.utils import logging
from areal.utils.data import concat_padded_tensors

logger = logging.getLogger("Tau2 workflow")


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
        n_trajs: int = 1,
        reward_type: str = "all",
        dynamic_filtering: bool = False,
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

        self.user_model = user_model
        self.user_api_key = user_api_key
        self.user_base_url = user_base_url

        self.eval_model = eval_model or user_model
        self.eval_api_key = eval_api_key or user_api_key
        self.eval_base_url = eval_base_url or user_base_url

        self.reward_type = reward_type
        self.dynamic_filtering = dynamic_filtering

    async def collect_agent_trajectory(self, engine, task):

        environment = get_environment()
        agent = LLMAgent(
            engine,
            self.tokenizer,
            self.gconfig,
            environment.get_policy(),
            environment.get_tools(),
            max_context_length=self.max_context_length - 100,
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
            base_url=self.eval_base_url,
        )

        messagaes = orchestrator.get_trajectory()
        traj_records = agent.records

        # calculate reward
        if self.reward_type == "db":
            reward = (
                reward_info.db_check.db_reward
                if reward_info.db_check is not None
                else 0
            )
        elif self.reward_type == "all":
            try:
                reward = reward_info.db_check.db_reward
                if len(reward_info.action_checks) > 0:
                    reward *= np.mean(
                        [x.action_reward for x in reward_info.action_checks]
                    )
                if len(reward_info.nl_assertions) > 0:
                    reward *= np.mean([x.met for x in reward_info.nl_assertions])

                print(
                    "[debug] reward info: ",
                    task.id,
                    reward_info.db_check.db_reward,
                    [x.action_reward for x in reward_info.action_checks],
                    [x.met for x in reward_info.nl_assertions],
                )

            except Exception as e:
                print("[debug] reward info: ", e, reward_info)
                reward = 0
        else:
            raise NotImplementedError

        return messagaes, traj_records, reward

    async def arun_episode(self, engine: InferenceEngine, raw_data=None):

        data = copy.deepcopy(raw_data)
        if data is None:
            tasks = get_tasks()
            task = random.choice(tasks)
        else:
            data["evaluation_criteria"] = json.loads(data["evaluation_criteria"])
            task = Task.model_validate(data)

        trajs = await asyncio.gather(
            *[self.collect_agent_trajectory(engine, task) for _ in range(self.n_trajs)]
        )
        version = engine.get_version()

        results = []
        rewards = []
        for i, (messagaes, traj_records, reward) in enumerate(trajs):
            rewards.append(reward)
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
                    results.append(TensorDict(res, batch_size=[1]))

        if self.dump_dir is not None:
            dump_path = os.path.join(self.dump_dir, str(version))
            await aiofiles.os.makedirs(dump_path, exist_ok=True)

            # Dump rollout to file
            file_path = os.path.join(
                self.dump_dir, str(version), f"{data['id']}_{uuid.uuid4().hex}.jsonl"
            )
            async with aiofiles.open(file_path, "a") as f:
                for i, (messages, _, score) in enumerate(trajs):
                    await f.write(
                        json.dumps(dict(messages=messages, reward=score, traj_idx=i))
                        + "\n"
                    )

        if self.dynamic_filtering:
            if np.max(rewards) == 0:
                # print("max_reward = 0", task.id, rewards)
                return None
            if np.min(rewards) == 1:
                # print("min_reward = 1", task.id, rewards)
                return None

        if len(results) == 0:
            return None
        else:
            # print("valid prompt", task.id, rewards)
            return concat_padded_tensors(results)
