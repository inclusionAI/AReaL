# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").
import os
from typing import List

from huggingface_hub import snapshot_download

from realhf.api.cli_args import ModelTrainEvalConfig, SGLangConfig, vLLMConfig
from realhf.api.quickstart.device_mesh import RPCAllocation
from realhf.base import logging

logger = logging.getLogger(__name__)


def check_is_realhf_native_impl(_cls):
    return _cls.__module__.startswith("realhf")


def check_is_realhf_native_model_interface(name):
    # NOTE: we should not import iterfaces here,
    # such that we can avoid CUDA initialization.
    return name in ["ppo_actor", "ppo_critic", "sft", "rw-math-code", "fused-threading"]


def check_valid_vllm(role: str, vllm: vLLMConfig, rpc_allocs: List[RPCAllocation]):
    rpcs = [alloc.rpc for alloc in rpc_allocs if alloc.rpc.role == role]
    if vllm.hybrid_train and not any(rpc.is_train() for rpc in rpcs):
        logger.warning("vLLM hybrid_train is enabled, but no training RPCs are found.")
    if vllm.hybrid_train and not vllm.enforce_eager:
        logger.warning(
            "For version < 0.7.0, vLLM hybrid_train requires eager mode to be enabled. "
            "The user has the responsibility to ensure the version is correct."
        )


def check_valid_sglang(
    role: str, sglang: SGLangConfig, rpc_allocs: List[RPCAllocation]
):
    rpcs = [alloc.rpc for alloc in rpc_allocs if alloc.rpc.role == role]
    if sglang.hybrid_train and not any(rpc.is_train() for rpc in rpcs):
        logger.warning(
            "SGLang hybrid_train is enabled, but no training RPCs are found."
        )
    if sglang.hybrid_train and not sglang.disable_cuda_graph:
        raise ValueError("SGLang hybrid_train requires CUDA graph to be disabled.")


def check_valid_optimizer(model: ModelTrainEvalConfig):
    if model.optimizer.min_lr_ratio < 0.0 or model.optimizer.min_lr_ratio > 1.0:
        raise ValueError(f"Invalid min_lr_ratio: {model.optimizer.min_lr_ratio}")
    if (
        model.optimizer.warmup_steps_proportion < 0.0
        or model.optimizer.warmup_steps_proportion > 1.0
    ):
        raise ValueError(
            f"Invalid warmup_steps_proportion: {model.optimizer.warmup_steps_proportion}"
        )


def check_valid_model_and_path(role: str, model: ModelTrainEvalConfig, fileroot):
    """
    Check if model path exists locally, download from HuggingFace Hub if not.

    Args:
        role: The role identifier for the model
        model: ModelTrainEvalConfig object containing model configuration

    Returns:
        str: The local path to the model (either existing or newly downloaded)

    Raises:
        Exception: If download fails or other errors occur
    """
    if os.path.exists(model.path):
        return

    logger.info(f"Model path `{model.path}` for `{role}` does not exist locally.")

    # Extract model name from path or use the path as model identifier
    # Adjust this logic based on how your ModelTrainEvalConfig stores the model identifier
    model_name = model.path

    target_path = os.path.join(fileroot, "models", model_name.replace("/", "__"))
    if not os.path.exists(target_path):
        # If not in cache, download to /models/ directory
        download_path = snapshot_download(
            repo_id=model_name,
            local_dir=target_path,  # Replace '/' to avoid path issues
        )
    logger.info(f"Model downloaded successfully to: {target_path}")
    # Update the model object's path to point to the downloaded location
    model.path = target_path


def check_valid_parallel_batch_size(rpc_alloc: RPCAllocation):
    try:
        rpc = rpc_alloc.rpc
        mb_spec = rpc.mb_spec

        dp_size = rpc_alloc.parallel.data_parallel_size
        tp_size = rpc_alloc.parallel.tensor_parallel_size
        pp_size = rpc_alloc.parallel.pipeline_parallel_size

        factor = 1
        if rpc.is_train() and rpc_alloc.parallel.pipeline_parallel_size > 1:
            factor = 2

        assert (
            rpc.n_seqs
            >= factor * dp_size * pp_size * rpc.min_n_seqs_per_pass * mb_spec.n_mbs
        ), (
            rpc.name,
            rpc.n_seqs,
            mb_spec,
            rpc.min_n_seqs_per_pass,
            factor,
            dp_size,
            pp_size,
        )
    except AssertionError as e:
        raise ValueError(
            f"Invalid parallel batch size and batch size configuration."
        ) from e
