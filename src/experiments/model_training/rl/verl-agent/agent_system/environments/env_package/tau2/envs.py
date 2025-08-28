from typing import Optional
import time
import logging

import gymnasium as gym
import numpy as np
import ray
from typing import Literal

from tau2.gym.gym_agent import AgentGymEnv
from tau2.registry import registry

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=0.25)
class Tau2Worker:
    """
    Ray remote actor that replaces the worker function.
    Each actor holds its own independent instance of AgentGymEnv.
    """

    def __init__(self, domain: str, env_kwargs: Optional[dict] = None):
        """Initialize the Tau2 environment in this worker"""
        self.domain = domain
        self.env_kwargs = env_kwargs
        self.env: Optional[AgentGymEnv] = None

    def step(self, action: str):
        """Execute a step in the environment"""
        assert self.env is not None, "Environment is not initialized"

        obs, reward, done, truncated, info = self.env.step(action)
        info["truncated"] = truncated

        # Redefine reward. We use rule-based reward
        info["won"] = False
        if done and reward == 1.0:
            info["won"] = True
        reward *= 10

        return obs, reward, done, info

        # tools and policy info only provided in the reset.
        # simplified_info = {"simulation_run": info["simulation_run"], "truncated": truncated}
        # return obs, reward, done, simplified_info

    def reset(self, task_id: str, max_steps: int = 100, solo_mode: bool = False, seed: int = 0):
        """Reset the environment with given seed"""
        if self.env is not None:
            self.env.close()
            time.sleep(2)

        self.env = AgentGymEnv(domain=self.domain, task_id=task_id, max_steps=max_steps, solo_mode=solo_mode, **self.env_kwargs)
        obs, info = self.env.reset(seed=seed)
        logger.info(f"Reset done. Info: {info}")
        return obs, info


class Tau2Envs(gym.Env):
    def __init__(
        self,
        domain: str,
        split: str = "train",
        max_steps: int = 100,
        solo_mode: bool = False,
        seed: int = 0,
        env_num: int = 1,
        group_n: int = 1,
        env_kwargs: Optional[dict] = None,
    ):
        super().__init__()

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        if "telecom" not in domain and solo_mode:
            raise ValueError(f"Solo mode is not supported for domain {domain}")

        splits_loader = registry.get_task_splits_loader(domain)
        if splits_loader is None:
            raise ValueError(f"No task splits loader found for domain {domain}")
        splits = splits_loader()
        self.task_ids = splits[split]
        self.max_steps = max_steps
        self.solo_mode = solo_mode

        self.num_processes = env_num * group_n
        self.env_num = env_num
        self.group_n = group_n
        self.seed = seed
        np.random.seed(self.seed)

        if env_kwargs is None:
            env_kwargs = {}

        # Create Ray remote actors instead of processes
        self.workers: list[Tau2Worker] = []
        for _ in range(self.num_processes):
            worker = Tau2Worker.remote(domain, env_kwargs)
            self.workers.append(worker)

    def step(self, actions: list):
        assert len(actions) == self.num_processes, "The num of actions must be equal to the num of processes"

        # Send step commands to all workers
        futures = []
        for worker, action in zip(self.workers, actions):
            future = worker.step.remote(action)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list

    def reset(self):
        """
        Send the reset command to all workers at once and collect initial obs/info from each environment.
        """
        obs_list = []
        info_list = []

        task_id = np.random.choice(self.task_ids, self.env_num, replace=False)
        # repeat task_id group_n times
        task_id = np.repeat(task_id, self.group_n).tolist()

        # Send reset commands to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.reset.remote(task_id[i], self.max_steps, self.solo_mode, self.seed)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)

        return obs_list, info_list

    def close(self):
        """
        Close all Ray actors
        """
        # Kill all Ray actors
        for worker in self.workers:
            ray.kill(worker)

    def __del__(self):
        self.close()


def build_tau2_envs(domain: str, split: str = "train", max_steps: int = 100, solo_mode: bool = False, seed: int = 0, env_num: int = 1, group_n: int = 1, env_kwargs: Optional[dict] = None):
    return Tau2Envs(domain, split, max_steps, solo_mode, seed, env_num, group_n, env_kwargs)
