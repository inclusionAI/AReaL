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

import os
import numpy as np
import ray
import time

from appworld import AppWorld, load_task_ids

def load_available_ports(port_file="appworld_ports.ports"):
    """
    Load available port list from file
    """
    if not os.path.exists(port_file):
        raise FileNotFoundError(f"Port file {port_file} does not exist. Please run the service startup script first.")
    
    ports = []
    with open(port_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and line.isdigit():
                ports.append(int(line))
    
    if not ports:
        raise ValueError(f"No valid ports found in port file {port_file}.")
    
    return ports

class AppWorldWorker:
    """
    Ray Actor that holds an instance of AppWorld and operates the environment
    based on method calls from the main process.
    """
    def __init__(self, worker_id, max_interactions, port):
        self.env = None
        self.current_step_count = 0
        self.max_interactions = max_interactions
        self.worker_id = worker_id
        
        self.url = f"http://0.0.0.0:{port}"

    def reset(self, task_id):
        """Reset the environment with a new task."""
        if self.env is not None:
            self.env.close()
            time.sleep(2)

        self.current_step_count = 0

        self.env = AppWorld(
            task_id=task_id,
            experiment_name=f'default_{self.worker_id}',
            remote_environment_url=self.url,
        )

        obs = self.env.task.instruction
        info = {
            "task_id": task_id,
            "supervisor": dict(self.env.task.supervisor),
        }
        return obs, info

    def step(self, action):
        """Execute one step in the environment."""
        if self.env is None:
            raise RuntimeError("Environment not reset before step. Please call reset() first.")

        self.current_step_count += 1

        obs = self.env.execute(action)

        done = self.env.task_completed() or (self.current_step_count >= self.max_interactions)

        if done:
            is_success = self.env.evaluate().success

            reward = 10.0 if is_success else 0.0
            info = {"won": is_success, "step_count": self.current_step_count}
        else:
            reward = 0.0
            info = {"won": False, "step_count": self.current_step_count}

        return obs, reward, done, info

    def close(self):
        """Close the environment."""
        if self.env is not None:
            self.env.close()


class AppWorldEnvs:
    """
    A Ray-based distributed wrapper for AppWorld.
    - Creates multiple Ray actors, each holding a separate AppWorld instance.
    - Implements Gym-style interfaces such as step() / reset() / close().
    """
    def __init__(self, 
                 dataset_name,
                 max_interactions,
                 seed,
                 env_num,
                 group_n,
                 start_server_id,
                 resources_per_worker,
                 port_file="appworld_ports.ports"
                 ):
        super().__init__()

        self.dataset_name = dataset_name
        self.max_interactions = max_interactions
        self.env_num = env_num
        self.group_n = group_n
        self.num_processes = env_num * group_n
        self.task_ids = load_task_ids(dataset_name)
   
        if self.env_num > len(self.task_ids):
            raise ValueError(f"Env_num ({self.env_num}) exceeds available task_ids in '{self.dataset_name}' ({len(self.task_ids)}). Please reducing env_num to {len(self.task_ids)}.")
            
        all_ports = load_available_ports(port_file)

        self.available_ports = all_ports[start_server_id:start_server_id + self.num_processes]

        # Check if we have enough ports
        if len(self.available_ports) < self.num_processes:
            raise ValueError(
                f"Need {self.num_processes} ports, but only {len(self.available_ports)} available ports. "
                f"Please ensure enough service instances are started."
            )

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        # Create Ray actors (workers)
        env_worker = ray.remote(**resources_per_worker)(AppWorldWorker)
        self.workers = []
        for i in range(self.num_processes):
            port = self.available_ports[i]
            worker = env_worker.remote(
                worker_id=start_server_id + i,
                max_interactions=self.max_interactions,
                port=port
            )
            self.workers.append(worker)

    def step(self, actions):
        """
        actions: Must be a list with length equal to self.num_processes, 
        each sent to the corresponding worker.
        
        Return format follows Gym's step() convention:
            observations, rewards, dones, infos
        """
        assert len(actions) == self.num_processes, "The length of actions must match the number of processes."

        # Send step commands to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.step.remote(actions[i])
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        
        obs_list = []
        reward_list = []
        done_list = []
        info_list = []

        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list

    def reset(self):
        """
        Reset all worker environments simultaneously, 
        returning each environment's initial observation and info.
        """
        # randomly select self.env_num task_id from self.task_ids
        task_id = np.random.choice(self.task_ids, self.env_num, replace=False)
        # repeat task_id group_n times
        task_id = np.repeat(task_id, self.group_n).tolist()

        # Send reset commands to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.reset.remote(task_id[i])
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        
        obs_list = []
        info_list = []

        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)

        return obs_list, info_list

    def close(self):
        """Close all workers."""
        # Send close commands to all workers
        futures = []
        for worker in self.workers:
            future = worker.close.remote()
            futures.append(future)
        
        # Wait for all workers to close
        ray.get(futures)
        
        # Shutdown Ray actors
        for worker in self.workers:
            ray.kill(worker)

    def render(self):
        """Implement this if visualization is needed."""
        pass

def build_appworld_envs(dataset_name="train",
                        max_interactions=50,
                        seed=0,
                        env_num=1, 
                        group_n=1,
                        start_server_id=0,
                        resources_per_worker={"num_cpus": 0.1},
                        ):

    return AppWorldEnvs(
        dataset_name=dataset_name,
        max_interactions=max_interactions,
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        start_server_id=start_server_id,
        resources_per_worker=resources_per_worker
    )