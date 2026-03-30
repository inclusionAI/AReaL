# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gymnasium as gym
import ray
import numpy as np
from gym_cards.envs import Point24Env, EZPointEnv, BlackjackEnv, NumberLineEnv


class GymCardsWorker:
    """
    Ray remote actor that replaces the worker function.
    Each actor holds an independent instance of the specified gym environment.
    """
    
    def __init__(self, env_id):
        """Initialize the gym environment in this worker"""
        if env_id == 'gym_cards/Points24-v0':
            self.env = Point24Env()
        elif env_id == 'gym_cards/EZPoints-v0':
            self.env = EZPointEnv()
        elif env_id == 'gym_cards/Blackjack-v0':
            self.env = BlackjackEnv()
        elif env_id == 'gym_cards/NumberLine-v0':
            self.env = NumberLineEnv()
        else:
            raise NotImplementedError(f"Unknown env_id: {env_id}")
    
    def step(self, action):
        """Execute a step in the environment"""
        obs, reward, done, _, info = self.env.step(action)
        return obs, reward, done, info
    
    def reset(self, seed_for_reset=None):
        """Reset the environment with optional seed"""
        if seed_for_reset is not None:
            obs, info = self.env.reset(seed=seed_for_reset)
        else:
            obs, info = self.env.reset()
        return obs, info


class GymMultiProcessEnv(gym.Env):
    """
    Ray-based parallel environment wrapper for gym cards environments.
    - env_id: Gym environment ID
    - env_num: Number of distinct environments
    - group_n: Number of replicas within each group (commonly used for multiple copies with the same seed)
    - env_kwargs: Parameters needed to create a single gym.make(env_id)
    """

    def __init__(self,
                 env_id,
                 seed=0,
                 env_num=1,
                 group_n=1,
                 resources_per_worker={"num_cpus": 0.1},
                 is_train=True):
        super().__init__()

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        self.env_id = env_id
        self.is_train = is_train
        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n

        np.random.seed(seed)
    
        # Create Ray remote actors instead of processes
        env_worker = ray.remote(**resources_per_worker)(GymCardsWorker)
        self.workers = []
        for _ in range(self.num_processes):
            worker = env_worker.remote(self.env_id)
            self.workers.append(worker)

    def step(self, actions):
        """
        Perform step in parallel.
        :param actions: list or numpy array, length must equal self.num_processes.
        :return: obs_list, reward_list, done_list, info_list
        """
        assert len(actions) == self.num_processes

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
        
        if isinstance(obs_list[0], np.ndarray):
            obs_list = np.array(obs_list)
        return obs_list, reward_list, done_list, info_list

    def reset(self):
        """
        Perform reset in parallel.
        Different seeds will be assigned to each environment (or the same seed within a group).
        :return: (obs_list, info_list)
        """
        if self.is_train:
            seeds = np.random.randint(0, 2**16 - 1, size=self.env_num)
        else:
            seeds = np.random.randint(2**16, 2**32 - 1, size=self.env_num)

        # Repeat seed for environments in the same group
        seeds = np.repeat(seeds, self.group_n)
        seeds = seeds.tolist()

        # Send reset commands to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.reset.remote(seeds[i])
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)

        if isinstance(obs_list[0], np.ndarray):
            obs_list = np.array(obs_list)
        return obs_list, info_list

    def close(self):
        """
        Close all Ray actors.
        """
        # Kill all Ray actors
        for worker in self.workers:
            ray.kill(worker)

    def __del__(self):
        self.close()


def build_gymcards_envs(env_name,
                        seed,
                        env_num,
                        group_n,
                        resources_per_worker,
                        is_train=True):
    """
    Externally exposed constructor function to create parallel Gym environments.
    - env_name: [gym_cards/Blackjack-v0, gym_cards/NumberLine-v0, gym_cards/EZPoints-v0, gym_cards/Points24-v0]
    - seed: For reproducible randomness
    - env_num: Number of distinct environments
    - group_n: Number of environment replicas under the same seed
    - is_train: Determines the seed range used (train/test)
    """
    return GymMultiProcessEnv(
        env_id=env_name,
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
    )