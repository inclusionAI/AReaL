import torch

from areal.api.cli_args import TrainEngineConfig
from areal.api.engine_api import InferenceEngine, TrainEngine
from areal.core.dist_rollout import DistRolloutCoordinator
from areal.engine.megatron import (
    MegatronComputeMixin,
    MegatronDistMixin,
    MegatronStateMixin,
    _MegatronModelList,
)


class MegatronEngine(
    MegatronDistMixin,
    MegatronStateMixin,
    MegatronComputeMixin,
    TrainEngine,
):
    def __init__(self, config: TrainEngineConfig):
        # === Configuration ===
        self.config = config
        self.dtype = getattr(torch, self.config.dtype)
        self.device = None
        self.optimizer_config = config.optimizer
        self.mcore_config = config.megatron

        # === Model & Optimizer (initialized in initialize()) ===
        self.model: _MegatronModelList | None = None
        self.hf_config = None  # PretrainedConfig, set in initialize()
        self.tf_config = None  # TransformerConfig, set in initialize()
        self.optimizer = None
        self.lr_scheduler = None
        self.checkpointer = None
        self.bridge = None
        self.tokenizer = None

        # === Parallelism ===
        self.parallel_strategy = None
        self.rank: int | None = None
        self.world_size: int | None = None
        self.is_pp_head: bool = False
        self.own_global_group: bool = False

        # === Process Group State ===
        self.process_group_initialized = False
        self._context_and_model_parallel_group = None
        self._cpu_group = None

        # === Weight Update ===
        self.weight_update_group_initialized: bool = False
        self.weight_update_group_name: str = ""
        self.weight_update_group = None

        # === Rollout ===
        self.rollout_engine: InferenceEngine | None = None
        self.rollout_coordinator: DistRolloutCoordinator | None = None

        # === Versioning & State ===
        self._version: int = 0
        self.seed: int = 0
        self.is_offload: bool = False
        self.engine_lock = None

        # === Logger (initialized in create_process_group()) ===
        self.logger = None
