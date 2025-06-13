import abc
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, List, Optional

import torch

from arealite.api.cli_args import TrainingArgs, TrajCollectorConfig
from arealite.api.io_struct import Trajectory


class TrajCollector(abc.ABC):
    def __init__(self, args: TrainingArgs, config: TrajCollectorConfig):
        self.args = args
        self.config = config

    def run_episode(
        self,
        env_option: Optional[Any] = None,
        seed: Optional[int] = None,
    ) -> Trajectory:
        """Run a single episode and return trajectory."""
        raise NotImplementedError()

    @staticmethod
    def run_episode_batch(
        collectors: List["TrajCollector"],
        env_options: Optional[List[Any]] = None,
        seeds: Optional[List[int]] = None,
    ) -> List[Trajectory]:
        """Run episodes in batch and return list of trajectories."""
        if env_options is None:
            env_options = [None] * len(collectors)
        if seeds is None:
            seeds = [None] * len(collectors)

        trajectories = []
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(collector.run_episode, env_option, seed)
                for collector, env_option, seed in zip(collectors, env_options, seeds)
            ]
            for future in futures:
                trajectories.append(future.result())
        return trajectories


@dataclass
class TrajCollectorFactory:
    args: TrainingArgs

    def make_collector(self, config: TrajCollectorConfig) -> TrajCollector:
        if config.type == "llm":
            from arealite.impl.collector.llm_collector import LLMTrajCollector

            return LLMTrajCollector(self.args, config)
        else:
            raise NotImplementedError(f"Unknown collector type: {config.type}")
