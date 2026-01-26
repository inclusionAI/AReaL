from __future__ import annotations

import os
import shutil
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)

from areal.utils.fsdp.checkpoint import DCPState

if TYPE_CHECKING:
    from transformers import AutoProcessor, PreTrainedTokenizerFast

    from areal.experimental.engine.archon_engine import ArchonEngine


def save_model_to_hf(
    engine: ArchonEngine,
    path: str,
    tokenizer: PreTrainedTokenizerFast | None,
    processor: AutoProcessor | None = None,
) -> None:
    """Save model in HuggingFace format using DCP infrastructure."""
    from torch.distributed.checkpoint import HuggingFaceStorageWriter

    if engine.model is None:
        raise RuntimeError("Model not initialized")
    if engine.state_dict_adapter is None:
        raise RuntimeError("state_dict_adapter is required for HF format")

    engine.logger.info(f"Saving HF checkpoint to {path}")
    os.makedirs(path, exist_ok=True)

    # Get distributed state dict
    options = StateDictOptions(full_state_dict=False, cpu_offload=True)
    state_dict = get_model_state_dict(engine.model, options=options)

    # Convert to HF format using adapter
    hf_state_dict = engine.state_dict_adapter.to_hf(state_dict)

    fqn_to_index_mapping = engine.state_dict_adapter.fqn_to_index_mapping

    # NOTE: HuggingFaceStorageWriter always creates a sharded/ subdirectory when
    # save_distributed=True. With enable_consolidation=True, it saves shards to
    # path/sharded/, then consolidates to path/. The sharded/ directory is NOT
    # automatically cleaned up by PyTorch, so we must remove it manually.
    sharded_dir = os.path.join(path, "sharded")

    if fqn_to_index_mapping:
        # Multi-file output: save to sharded/, then consolidate with all ranks
        from torch.distributed.checkpoint._consolidate_hf_safetensors import (
            consolidate_safetensors_files_on_every_rank,
        )

        hf_writer = HuggingFaceStorageWriter(
            path=sharded_dir,
            save_distributed=True,
            fqn_to_index_mapping=fqn_to_index_mapping,
            enable_consolidation=False,
        )
        dcp.save(hf_state_dict, storage_writer=hf_writer)

        # NOTE: consolidate_safetensors_files_on_every_rank() has internal barrier
        consolidate_safetensors_files_on_every_rank(
            input_dir=sharded_dir,
            output_dir=path,
            fqn_to_index_mapping=fqn_to_index_mapping,
            num_threads=8,
        )
    else:
        # Single-file output: auto consolidation
        hf_writer = HuggingFaceStorageWriter(
            path=path,
            save_distributed=True,
            enable_consolidation=True,
        )
        dcp.save(hf_state_dict, storage_writer=hf_writer)

    # Clean up sharded/ directory after consolidation
    if dist.get_rank() == 0 and os.path.exists(sharded_dir):
        shutil.rmtree(sharded_dir)

    dist.barrier(group=engine.cpu_group)

    if dist.get_rank() == 0:
        engine.model_config.save_pretrained(path)
        if tokenizer is not None:
            tokenizer.save_pretrained(path)
        if processor is not None:
            processor.save_pretrained(path)
    dist.barrier(group=engine.cpu_group)


def load_model_from_hf(engine: ArchonEngine, path: str) -> None:
    """Load model from HuggingFace format using DCP infrastructure."""
    if engine.model is None:
        raise RuntimeError("Model not initialized")
    if engine.state_dict_adapter is None:
        raise RuntimeError("state_dict_adapter is required for HF format")

    engine.logger.info(f"Loading HF checkpoint from {path}")

    # Get model state dict structure (distributed)
    options = StateDictOptions(full_state_dict=False, cpu_offload=True)
    state_dict = get_model_state_dict(engine.model, options=options)

    # Convert to HF format to match checkpoint keys
    hf_state_dict = engine.state_dict_adapter.to_hf(state_dict)

    # Load using DCP with HuggingFaceStorageReader
    hf_reader = engine.state_dict_adapter.get_hf_storage_reader(path)
    dcp.load(hf_state_dict, storage_reader=hf_reader)

    # Convert back to Archon format
    archon_state_dict = engine.state_dict_adapter.from_hf(hf_state_dict)

    # Load into FSDP model (same as DCPState.load_state_dict)
    set_model_state_dict(
        engine.model,
        model_state_dict=archon_state_dict,
        options=StateDictOptions(strict=False),
    )

    dist.barrier(group=engine.cpu_group)


def save_to_dcp(engine: ArchonEngine, path: str, with_optim: bool) -> None:
    """Save model (and optionally optimizer) using DCP format."""
    if engine.model is None:
        raise RuntimeError("Model not initialized")

    os.makedirs(path, exist_ok=True)

    dcp_state = DCPState(engine.model, engine.optimizer if with_optim else None)
    state_dict = {"dcp": dcp_state}
    dcp.save(state_dict, checkpoint_id=path)


def load_from_dcp(engine: ArchonEngine, path: str, with_optim: bool) -> None:
    """Load model (and optionally optimizer) from DCP format."""
    if engine.model is None:
        raise RuntimeError("Model not initialized")

    dcp_state = DCPState(engine.model, engine.optimizer if with_optim else None)
    state_dict = {"dcp": dcp_state}
    dcp.load(state_dict=state_dict, checkpoint_id=path)


def save_optimizer_state(engine: ArchonEngine, path: str) -> None:
    """Save optimizer state to disk (sharded by rank)."""
    assert engine.optimizer is not None
    assert dist.is_initialized()
    rank = dist.get_rank()
    shard_path = os.path.join(
        path, f"optim_world_size_{engine.world_size}_rank_{rank}.pt"
    )
    state_dict = engine.optimizer.state_dict()
    torch.save(state_dict, shard_path)
    dist.barrier(group=engine.cpu_group)


def load_optimizer_state(engine: ArchonEngine, path: str) -> None:
    """Load optimizer state from disk (sharded by rank)."""
    assert engine.optimizer is not None
    assert dist.is_initialized()
    rank = dist.get_rank()
    shard_path = os.path.join(
        path, f"optim_world_size_{engine.world_size}_rank_{rank}.pt"
    )
    optimizer_state_dict = torch.load(shard_path, weights_only=False)
    engine.optimizer.load_state_dict(optimizer_state_dict)
    dist.barrier(group=engine.cpu_group)
