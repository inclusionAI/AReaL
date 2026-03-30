"""Vectorized environment wrapper for parallel VisionQATask instances.

This module provides a vectorized environment that wraps multiple VisionQATask
instances for efficient parallel training with RL algorithms.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import gymnasium as gym
import numpy as np

from geo_edit.environment.task.vision_qa_task import VisionQATask


class VecVisionQAEnv:
    """Vectorized environment for parallel VisionQATask execution.

    This class wraps multiple VisionQATask instances and provides a vectorized
    interface compatible with common RL training frameworks.

    Example:
        >>> tasks = [task1, task2, task3]
        >>> vec_env = VecVisionQAEnv(tasks)
        >>> obs, infos = vec_env.reset()
        >>> # obs is a list of observations, one per task
        >>> obs, rewards, terminateds, truncateds, infos = vec_env.step(actions)
    """

    def __init__(
        self,
        tasks: List[VisionQATask],
        auto_reset: bool = False,
    ):
        """Initialize vectorized environment.

        Args:
            tasks: List of VisionQATask instances to wrap
            auto_reset: If True, automatically reset terminated environments
        """
        if not tasks:
            raise ValueError("tasks list cannot be empty")

        self.tasks = tasks
        self.num_envs = len(tasks)
        self.auto_reset = auto_reset

        # Track which environments are done
        self._dones = [False] * self.num_envs

        # Use first task's spaces as reference
        self.observation_space = tasks[0].observation_space
        self.action_space = tasks[0].action_space

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Reset all environments.

        Args:
            seed: Optional random seed (distributed to envs as seed+i)
            options: Optional list of option dicts, one per env

        Returns:
            observations: List of observations from each environment
            infos: List of info dicts from each environment
        """
        observations = []
        infos = []

        for i, task in enumerate(self.tasks):
            env_seed = seed + i if seed is not None else None
            env_options = options[i] if options else None
            obs, info = task.reset(seed=env_seed, options=env_options)
            observations.append(obs)
            infos.append(info)

        self._dones = [False] * self.num_envs
        return observations, infos

    def step(
        self,
        actions: List[Any],
        extra_infos: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[
        List[Dict[str, Any]],
        List[float],
        List[bool],
        List[bool],
        List[Dict[str, Any]],
    ]:
        """Execute actions in all environments.

        Args:
            actions: List of actions, one per environment
            extra_infos: Optional list of extra_info dicts for each action

        Returns:
            observations: List of new observations
            rewards: List of rewards
            terminateds: List of terminated flags
            truncateds: List of truncated flags
            infos: List of info dicts
        """
        if len(actions) != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} actions, got {len(actions)}")

        extra_infos = extra_infos or [None] * self.num_envs

        observations = []
        rewards = []
        terminateds = []
        truncateds = []
        infos = []

        for i, (task, action, extra_info) in enumerate(
            zip(self.tasks, actions, extra_infos)
        ):
            if self._dones[i]:
                # Environment already done, return last observation
                obs = task._get_observation()
                observations.append(obs)
                rewards.append(0.0)
                terminateds.append(True)
                truncateds.append(False)
                infos.append({"already_done": True})
            else:
                obs, reward, terminated, truncated, info = task.step(
                    action, extra_info=extra_info
                )
                observations.append(obs)
                rewards.append(reward)
                terminateds.append(terminated)
                truncateds.append(truncated)
                infos.append(info)

                if terminated or truncated:
                    self._dones[i] = True

                    if self.auto_reset:
                        # Auto-reset this environment
                        final_obs = obs
                        final_info = info
                        obs, info = task.reset()
                        observations[i] = obs
                        infos[i] = {
                            **info,
                            "final_observation": final_obs,
                            "final_info": final_info,
                        }
                        self._dones[i] = False

        return observations, rewards, terminateds, truncateds, infos

    def reset_at(
        self,
        index: int,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset a specific environment.

        Args:
            index: Index of environment to reset
            seed: Optional random seed
            options: Optional options dict

        Returns:
            observation: Observation from reset environment
            info: Info dict from reset environment
        """
        if index < 0 or index >= self.num_envs:
            raise IndexError(f"Index {index} out of range [0, {self.num_envs})")

        obs, info = self.tasks[index].reset(seed=seed, options=options)
        self._dones[index] = False
        return obs, info

    def save_states(self) -> List[Dict[str, Any]]:
        """Save states of all environments.

        Returns:
            List of state dicts from each environment
        """
        return [task.save_state() for task in self.tasks]

    def restore_states(self, states: List[Dict[str, Any]]) -> None:
        """Restore states of all environments.

        Args:
            states: List of state dicts to restore
        """
        if len(states) != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} states, got {len(states)}")

        for task, state in zip(self.tasks, states):
            task.restore_state(state)

        # Reset done flags based on restored states
        self._dones = [False] * self.num_envs

    def save_state_at(self, index: int) -> Dict[str, Any]:
        """Save state of a specific environment.

        Args:
            index: Index of environment

        Returns:
            State dict from the environment
        """
        if index < 0 or index >= self.num_envs:
            raise IndexError(f"Index {index} out of range [0, {self.num_envs})")
        return self.tasks[index].save_state()

    def restore_state_at(self, index: int, state: Dict[str, Any]) -> None:
        """Restore state of a specific environment.

        Args:
            index: Index of environment
            state: State dict to restore
        """
        if index < 0 or index >= self.num_envs:
            raise IndexError(f"Index {index} out of range [0, {self.num_envs})")
        self.tasks[index].restore_state(state)
        self._dones[index] = False

    def get_attr(self, attr_name: str) -> List[Any]:
        """Get an attribute from all environments.

        Args:
            attr_name: Name of attribute to get

        Returns:
            List of attribute values from each environment
        """
        return [getattr(task, attr_name) for task in self.tasks]

    def set_attr(self, attr_name: str, values: List[Any]) -> None:
        """Set an attribute on all environments.

        Args:
            attr_name: Name of attribute to set
            values: List of values to set
        """
        if len(values) != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} values, got {len(values)}")
        for task, value in zip(self.tasks, values):
            setattr(task, attr_name, value)

    def close(self) -> None:
        """Close all environments."""
        for task in self.tasks:
            task.close()

    @property
    def contents(self) -> List[Any]:
        """Get contents from all environments (for compatibility)."""
        return [task.contents for task in self.tasks]

    def __len__(self) -> int:
        return self.num_envs

    def __getitem__(self, index: int) -> VisionQATask:
        return self.tasks[index]


def make_vec_env(
    task_class: Type[VisionQATask],
    task_configs: List[Dict[str, Any]],
    auto_reset: bool = False,
) -> VecVisionQAEnv:
    """Create a vectorized environment from task configs.

    Args:
        task_class: Class of task to instantiate
        task_configs: List of config dicts for each task
        auto_reset: If True, automatically reset terminated environments

    Returns:
        VecVisionQAEnv instance wrapping all tasks
    """
    tasks = [task_class(**config) for config in task_configs]
    return VecVisionQAEnv(tasks, auto_reset=auto_reset)
