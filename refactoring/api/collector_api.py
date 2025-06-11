import abc
from dataclasses import dataclass
from typing import Any, Optional

import torch

from refactoring.api.cli_args import TrainingArgs, TrajCollectorConfig
from refactoring.api.io_struct import Trajectory


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


@dataclass
class TrajCollectorFactory:
    args: TrainingArgs

    def make_collector(self, config: TrajCollectorConfig) -> TrajCollector:
        if config.type == "llm":
            from refactoring.impl.collector.llm_collector import LLMTrajCollector

            return LLMTrajCollector(self.args, config)
        else:
            raise NotImplementedError(f"Unknown collector type: {config.type}")
