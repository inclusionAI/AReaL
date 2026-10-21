# SPDX-License-Identifier: Apache-2.0

"""CPU-only, composable actor timing model. No GPU collection or model API calls."""

from .actor import ActorComponents, ActorCostModel
from .catalog import hardware_profile, qwen3_30b_a3b, trial_config
from .components import Efficiency, Parallel, Scale, Serial
from .config import ActorConfig, Hardware, ModelShape, ParallelLayout
from .workload import WorkloadStats

__all__ = [
    "ActorComponents",
    "ActorConfig",
    "ActorCostModel",
    "Efficiency",
    "Hardware",
    "ModelShape",
    "Parallel",
    "ParallelLayout",
    "Scale",
    "Serial",
    "WorkloadStats",
    "hardware_profile",
    "qwen3_30b_a3b",
    "trial_config",
]
