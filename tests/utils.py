"""Test utilities for AReaL tests.

This module provides utilities shared across tests. Common utilities are
imported from areal.utils.testing_utils to avoid code duplication with
profiling tools.
"""

import asyncio
import random
from typing import Any

import torch

from areal.api.engine_api import InferenceEngine
from areal.api.workflow_api import RolloutWorkflow
from areal.experimental.openai.types import InteractionWithTokenLogpReward

# Re-export common utilities from production code
from areal.utils.testing_utils import get_dataset_path, get_model_path

__all__ = [
    "get_model_path",
    "get_dataset_path",
    "TestWorkflow",
]


class TestWorkflow(RolloutWorkflow):
    """Simple test workflow for testing RolloutWorkflow functionality."""

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, Any] | None | dict[str, InteractionWithTokenLogpReward]:
        await asyncio.sleep(0.1)
        prompt_len = random.randint(2, 8)
        gen_len = random.randint(2, 8)
        seqlen = prompt_len + gen_len
        return dict(
            input_ids=torch.randint(
                0,
                100,
                (
                    1,
                    seqlen,
                ),
            ),
            attention_mask=torch.ones(1, seqlen, dtype=torch.bool),
            loss_mask=torch.tensor(
                [0] * prompt_len + [1] * gen_len, dtype=torch.bool
            ).unsqueeze(0),
            rewards=torch.randn(1),
        )
