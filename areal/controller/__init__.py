"""Controller components for managing distributed training and inference."""

from areal.controller.rollout_controller import RolloutController
from areal.controller.train_controller import TrainController

__all__ = [
    "RolloutController",
    "TrainController",
]
