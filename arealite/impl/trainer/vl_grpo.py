# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0

import functools
import os
import time
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
from datasets import Dataset

from arealite import ppo_functional
from arealite.api.cli_args import (
    GRPOTrainerConfig,
    MicroBatchSpec,
    TrainerConfig,
    TrainingArgs,
)
from arealite.api.engine_api import EngineFactory
from arealite.api.io_struct import FinetuneSpec, Trajectory
from arealite.api.llm_client_api import LLMClientFactory
from arealite.api.trainer_api import Trainer
from arealite.system.rollout_controller import RolloutController
from arealite.utils import (
    calc_entropy,
    close_wandb_tensorboard,
    compute_varlen_position_indices,
    concat_padded_tensors,
    gather_logprobs,
    init_stats_logging,
    log_wandb_tensorboard,
    masked_normalization,
    record_timing,
    split_dict_tensor_with_cu_seqlens,
    to_device,
    unpad_input,
)
from realhf.api.core.data_api import load_hf_tokenizer, tabulate_stats,load_hf_processor_and_tokenizer
from realhf.base import constants, logging, name_resolve, names, stats_tracker, timeutil
from .grpo import SpmdGRPOTrainer
logger = logging.getLogger("VL GRPO Trainer", "system")


class VL_SpmdGRPOTrainer(SpmdGRPOTrainer):
    def __init__(
        self,
        args: TrainingArgs,
        trainer_config: TrainerConfig,
        train_dataset: Dataset,
        valid_dataset: Optional[Dataset] = None,
        rollout_controller: Optional[RolloutController] = None,
    ):
        super().__init__(args, trainer_config, train_dataset, valid_dataset, rollout_controller)
