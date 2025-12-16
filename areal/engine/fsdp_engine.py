import os

from transformers import AutoConfig

from areal.api.cli_args import TrainEngineConfig
from areal.api.engine_api import TrainEngine
from areal.engine.fsdp import (
    FSDPComputeMixin,
    FSDPDistMixin,
    FSDPStateMixin,
)
from areal.utils.model import is_valid_vision_model


class FSDPEngine(FSDPDistMixin, FSDPStateMixin, FSDPComputeMixin, TrainEngine):
    """FSDP Training Engine.

    Type hints for all attributes are defined in FSDPEngineProtocol.
    """

    def __init__(self, config: TrainEngineConfig):
        self.config = config
        self.optimizer_config = config.optimizer

        self.processor = None
        self._version = 0

        self.initialized = False
        self.own_global_group = False
        self.weight_update_group_initialized = False

        self.model_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=self.config.path,
            trust_remote_code=True,
        )
        self.is_vision_model = is_valid_vision_model(self.model_config.model_type)
        self.world_size = int(os.environ["WORLD_SIZE"])

        self.cpu_offload = None
        self.rollout_engine = None
        self.rollout_coordinator = None
        self.is_offload = False
