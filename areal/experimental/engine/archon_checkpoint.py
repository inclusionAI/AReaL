from __future__ import annotations

import os
import shutil
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful

if TYPE_CHECKING:
    from transformers import AutoProcessor, PreTrainedTokenizerFast

    from areal.experimental.engine.archon_engine import ArchonEngine


class DCPState(Stateful):
    """DCP wrapper for archon models.

    Key design decisions:
    - Uses flatten_optimizer_state_dict=True to avoid param_group index collisions
      (without flatten, each optimizer uses indices 0, 1, 2... which collide across
      PP stages; with flatten, keys become parameter FQNs which are unique)
    - For PP (len(model_parts) > 1): uses strict=False when loading because each
      PP stage only has subset of keys
    - For non-PP (len(model_parts) == 1): uses strict=True to catch real issues
    """

    def __init__(
        self,
        model_parts: list[nn.Module] | nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
    ):
        """Initialize DCPState.

        Args:
            model_parts: Single model or list of model parts from pipeline_llm
            optimizer: Optimizer for the model(s)
        """
        if isinstance(model_parts, nn.Module):
            self.model_parts = [model_parts]
        else:
            self.model_parts = model_parts
        self.optimizer = optimizer
        # PP mode uses non-strict loading since each stage only has subset of keys
        self._is_pp = len(self.model_parts) > 1

    def state_dict(self) -> dict[str, Any]:
        """Get state dict for model parts and optimizer using DCP utilities."""
        # Merge model state dicts from all parts
        # cpu_offload=True ensures tensors are on CPU for DCP filesystem writer
        model_state: dict[str, Any] = {}
        model_options = StateDictOptions(cpu_offload=True)
        for model_part in self.model_parts:
            part_state = get_model_state_dict(model_part, options=model_options)
            model_state.update(part_state)

        state: dict[str, Any] = {"model": model_state}

        if self.optimizer is not None:
            optim_options = StateDictOptions(
                flatten_optimizer_state_dict=True,
                cpu_offload=True,
            )

            # Get optimizer state for each model part and merge
            optim_state: dict[str, Any] = {}
            for model_part in self.model_parts:
                part_optim = get_optimizer_state_dict(
                    model_part, self.optimizer, options=optim_options
                )
                optim_state.update(part_optim)

            state["optim"] = optim_state

        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state dicts onto model parts and optimizer."""
        model_state = state_dict["model"]

        model_options = StateDictOptions(strict=not self._is_pp)
        for model_part in self.model_parts:
            set_model_state_dict(model_part, model_state, options=model_options)

        if self.optimizer is not None and "optim" in state_dict:
            optim_state = state_dict["optim"]
            optim_options = StateDictOptions(
                strict=not self._is_pp,
                flatten_optimizer_state_dict=True,
            )
            for model_part in self.model_parts:
                set_optimizer_state_dict(
                    model_part, self.optimizer, optim_state, options=optim_options
                )


def _validate_model_initialized(engine: ArchonEngine) -> None:
    """Validate that model is properly initialized for checkpoint operations."""
    if not engine.model_parts:
        raise RuntimeError("Model parts not initialized")


def _get_merged_state_dict(
    engine: ArchonEngine,
    options: StateDictOptions,
) -> dict[str, Any]:
    """Get merged model state dict, handling PP mode."""
    if engine.parallel_dims.pp_enabled:
        state_dict: dict = {}
        for model_part in engine.model_parts:
            part_state = get_model_state_dict(model_part, options=options)
            state_dict.update(part_state)
        return state_dict
    return get_model_state_dict(engine.model, options=options)


def save_model_to_hf(
    engine: ArchonEngine,
    path: str,
    tokenizer: PreTrainedTokenizerFast | None,
    processor: AutoProcessor | None = None,
) -> None:
    """Save model in HuggingFace format using DCP infrastructure."""
    from torch.distributed.checkpoint import HuggingFaceStorageWriter

    _validate_model_initialized(engine)
    if engine.state_dict_adapter is None:
        raise RuntimeError("state_dict_adapter is required for HF format")

    engine.logger.info(f"Saving HF checkpoint to {path}")
    os.makedirs(path, exist_ok=True)

    # Get distributed state dict
    options = StateDictOptions(full_state_dict=False, cpu_offload=True)
    state_dict = _get_merged_state_dict(engine, options)

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
    _validate_model_initialized(engine)
    if engine.state_dict_adapter is None:
        raise RuntimeError("state_dict_adapter is required for HF format")

    engine.logger.info(f"Loading HF checkpoint from {path}")

    # Get model state dict structure
    options = StateDictOptions(full_state_dict=False, cpu_offload=True)
    state_dict = _get_merged_state_dict(engine, options)

    # Convert to HF format to match checkpoint keys
    hf_state_dict = engine.state_dict_adapter.to_hf(state_dict)

    # PP mode + weight tying fix: last stage needs embed_tokens weight for output layer
    # When tie_word_embeddings=True, HF checkpoint only stores embed_tokens.weight,
    # not lm_head.weight. In PP mode, last stage has output.weight but no tok_embeddings,
    # so we need to explicitly load embed_tokens.weight even though it's not in state_dict.
    pp_weight_tying_fix = (
        engine.parallel_dims.pp_enabled
        and engine.pp_has_last_stage
        and getattr(engine.state_dict_adapter, "enable_weight_tying", False)
        and "output.weight" in state_dict
    )
    if pp_weight_tying_fix:
        # Add a placeholder with embed_tokens key so DCP will load it
        embed_key = "model.embed_tokens.weight"
        if embed_key not in hf_state_dict:
            output_tensor = state_dict["output.weight"]
            hf_state_dict[embed_key] = torch.empty_like(output_tensor)

    # Load using DCP with HuggingFaceStorageReader
    hf_reader = engine.state_dict_adapter.get_hf_storage_reader(path)
    dcp.load(hf_state_dict, storage_reader=hf_reader)

    # Convert back to Archon format
    archon_state_dict = engine.state_dict_adapter.from_hf(hf_state_dict)

    # In PP mode, filter to only keep keys needed by this rank's model_parts
    model_keys = set(state_dict.keys())
    if engine.parallel_dims.pp_enabled:
        archon_state_dict = {
            k: v for k, v in archon_state_dict.items() if k in model_keys
        }
    loaded_keys = set(archon_state_dict.keys())

    # Compute key differences for diagnostics
    missing_keys = model_keys - loaded_keys
    unexpected_keys = loaded_keys - model_keys

    # Filter known expected missing keys
    expected_missing = set()
    for key in list(missing_keys):
        # rotary_emb is computed at runtime, not stored in checkpoint
        if "rotary_emb" in key:
            expected_missing.add(key)
    missing_keys -= expected_missing

    if dist.get_rank() == 0:
        if missing_keys:
            engine.logger.warning(
                f"Unexpected missing keys in checkpoint: {missing_keys}"
            )
        if unexpected_keys:
            engine.logger.warning(
                f"Unexpected extra keys in checkpoint: {unexpected_keys}"
            )

    # Load into model(s)
    load_options = StateDictOptions(strict=False)
    if engine.parallel_dims.pp_enabled:
        for model_part in engine.model_parts:
            set_model_state_dict(
                model_part,
                model_state_dict=archon_state_dict,
                options=load_options,
            )
    else:
        set_model_state_dict(
            engine.model,
            model_state_dict=archon_state_dict,
            options=load_options,
        )

    dist.barrier(group=engine.cpu_group)


def save_to_dcp(engine: ArchonEngine, path: str, with_optim: bool) -> None:
    """Save model (and optionally optimizer) using DCP format."""
    _validate_model_initialized(engine)

    os.makedirs(path, exist_ok=True)

    dcp_state = DCPState(engine.model_parts, engine.optimizer if with_optim else None)

    state_dict = {"dcp": dcp_state}
    dcp.save(state_dict, checkpoint_id=path)

    dist.barrier(group=engine.cpu_group)


def load_from_dcp(engine: ArchonEngine, path: str, with_optim: bool) -> None:
    """Load model (and optionally optimizer) from DCP format."""
    _validate_model_initialized(engine)

    dcp_state = DCPState(engine.model_parts, engine.optimizer if with_optim else None)

    state_dict = {"dcp": dcp_state}
    dcp.load(state_dict=state_dict, checkpoint_id=path)

    dist.barrier(group=engine.cpu_group)


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
