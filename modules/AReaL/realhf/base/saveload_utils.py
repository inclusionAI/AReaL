# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import os
import shutil
from typing import Dict

import torch
import tqdm
from safetensors import safe_open

from realhf.base import logging

logger = logging.getLogger("SaveLoad")


def split_state_dict_into_shards(state_dict: Dict, n_shards: int) -> Dict:
    if n_shards == 1:
        return [state_dict]

    keys = list(state_dict.keys())
    if len(keys) < n_shards:
        raise ValueError(f"state_dict has {len(keys)} keys, but n_shards={n_shards}")

    shard_size = len(keys) // n_shards
    extra = len(keys) % n_shards
    shard_size_list = [shard_size for _ in range(n_shards)]
    shard_size_list[-1] = shard_size + extra
    start, shards = 0, []
    for i, size in enumerate(
        tqdm.tqdm(
            shard_size_list,
            desc=f"Splitting state dict into {len(shard_size_list)} shards...",
        )
    ):
        shard = {}
        for j in range(start, start + size):
            shard[keys[j]] = state_dict[keys[j]]
        start += size
        shards.append(shard)
    return shards


HF_MODEL_CONFIG_FILES = [
    "generation_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
]


def copy_hf_configs(src_model_dir, dst_model_dir):
    for file in HF_MODEL_CONFIG_FILES:
        try:
            shutil.copy(
                os.path.join(src_model_dir, file),
                os.path.join(dst_model_dir, file),
            )
            logger.info(f"copied {file} from {src_model_dir} to {dst_model_dir}")
        except FileNotFoundError:
            logger.info(f"{file} not exist in {src_model_dir} skipping.")
    # Copy remote codes
    for file in os.listdir(src_model_dir):
        for prefix in ["chat_format", "configuration_", "modeling_", "tokenization_"]:
            if file.startswith(prefix) and file.endswith(".py"):
                shutil.copy(
                    os.path.join(src_model_dir, file),
                    os.path.join(dst_model_dir, file),
                )
                logger.info(f"copied {file} from {src_model_dir} to {dst_model_dir}")


def load_safetensor(fn: str) -> Dict[str, torch.Tensor]:
    assert fn.endswith(".safetensors")
    state_dict = {}
    with safe_open(fn, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    return state_dict
