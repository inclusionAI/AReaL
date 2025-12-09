import asyncio
import os
import random
from typing import Any

import torch
from huggingface_hub import snapshot_download

from areal.api.engine_api import InferenceEngine
from areal.api.workflow_api import RolloutWorkflow
from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.utils import logging

logger = logging.getLogger("areal.tests.utils")

_warned_bool_env_var_keys = set()


def get_bool_env_var(name: str, default: str = "false") -> bool:
    value = os.getenv(name, default)
    value = value.lower()

    truthy_values = ("true", "1")
    falsy_values = ("false", "0")

    if (value not in truthy_values) and (value not in falsy_values):
        if value not in _warned_bool_env_var_keys:
            logger.warning(
                f"get_bool_env_var({name}) see non-understandable value={value} and treat as false"
            )
        _warned_bool_env_var_keys.add(value)

    return value in truthy_values


def is_in_ci():
    return get_bool_env_var("AREAL_IS_IN_CI")


def get_model_path(local_path: str, hf_id: str) -> str:
    """Get model path, preferring local storage over HuggingFace Hub.

    If local_path exists, returns it directly. Otherwise downloads
    the model from HuggingFace Hub using snapshot_download.
    """
    if os.path.exists(local_path):
        logger.info(f"Model found at local path: {local_path}")
        return local_path

    try:
        logger.info(f"Downloading model from HuggingFace Hub: {hf_id}")
        downloaded_path = snapshot_download(
            repo_id=hf_id,
            # Allow partial downloads for faster testing
            ignore_patterns=["*.gguf", "*.ggml", "consolidated*"],
        )
        logger.info(f"Model downloaded to: {downloaded_path}")
        return downloaded_path
    except Exception as e:
        logger.error(f"Failed to download model {hf_id}: {e}")
        raise e


def get_dataset_path(local_path: str, hf_id: str) -> str:
    """Get dataset path, preferring local storage over HuggingFace Hub.

    If local_path exists, returns it directly. Otherwise downloads
    the dataset from HuggingFace Hub using snapshot_download.
    """
    if os.path.exists(local_path):
        logger.info(f"Dataset found at local path: {local_path}")
        return local_path

    try:
        logger.info(f"Downloading dataset from HuggingFace Hub: {hf_id}")
        downloaded_path = snapshot_download(
            repo_id=hf_id,
            repo_type="dataset",
        )
        logger.info(f"Dataset downloaded to: {downloaded_path}")
        return downloaded_path
    except Exception as e:
        logger.error(f"Failed to download dataset {hf_id}: {e}")
        raise e


class TestWorkflow(RolloutWorkflow):
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
