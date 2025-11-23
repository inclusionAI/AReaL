import os
import time
from importlib.metadata import version as get_version
from typing import Any

import pytest
import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import (
    MegatronEngineConfig,
    MicroBatchSpec,
    OptimizerConfig,
    TrainEngineConfig,
)
from areal.api.io_struct import FinetuneSpec, SaveLoadMeta
from areal.engine.megatron_engine import MegatronEngine
from areal.platforms import current_platform
from areal.utils import logging
from areal.utils.device import log_gpu_stats
from areal.utils.functional import gather_logprobs

logger = logging.getLogger("MegatronEngine Test")

VOCAB_SIZE = 100
MODEL_PATH = "/storage/openpsi/models/Qwen__Qwen3-0.6B/"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def mock_input(
    batch_size=5,
    min_seqlen=10,
    max_seqlen=20,
    device=current_platform.device_type,
) -> dict[str, Any]:
    """Create mock padded input data (same format for huggingface) for testing.
    Returns a dict with input_ids, attention_mask, and position_ids.
    """
    pad_token_id = 0
    seqlens = torch.randint(
        min_seqlen, max_seqlen, (batch_size,), dtype=torch.int, device=device
    )
    max_seqlen = int(max(seqlens))
    input_ids = torch.randint(
        0, VOCAB_SIZE, (batch_size, max_seqlen), dtype=torch.long, device=device
    )
    attn_mask = torch.zeros((batch_size, max_seqlen), dtype=torch.bool, device=device)

    attn_mask[
        torch.arange(0, max_seqlen, device=device).unsqueeze(0) < seqlens.unsqueeze(1)
    ] = 1
    input_ids.masked_fill_(~attn_mask, pad_token_id)

    return dict(
        input_ids=input_ids,
        attention_mask=attn_mask,
    )


@pytest.fixture(scope="module")
def mock_tree_input(
    batch_size=5,
    tree_tokens=30,
    total_tokens=60,
    device=current_platform.device_type,
):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if total_tokens < tree_tokens:
        raise ValueError("total_tokens must be >= tree_tokens")
    if total_tokens < batch_size:
        raise ValueError(
            "total_tokens must be >= batch_size to allocate at least one token per sequence"
        )

    device = device if isinstance(device, torch.device) else torch.device(device)
    lengths = [tree_tokens]
    remaining_tokens = total_tokens - tree_tokens
    remaining_slots = batch_size - 1

    if remaining_slots:
        if remaining_tokens < remaining_slots:
            raise ValueError("Not enough tokens available for the requested batch size")
        for index in range(remaining_slots):
            slots_left = remaining_slots - index - 1
            max_assignable = min(tree_tokens, remaining_tokens - slots_left)
            share = max(1, min(max_assignable, remaining_tokens // (slots_left + 1)))
            lengths.append(share)
            remaining_tokens -= share
        if remaining_tokens != 0:
            lengths[-1] += remaining_tokens
            remaining_tokens = 0
    else:
        if total_tokens != tree_tokens:
            raise ValueError("total_tokens must equal tree_tokens when batch_size is 1")

    lengths = [int(lengh) for lengh in lengths]
    if sum(lengths) != total_tokens:
        raise RuntimeError("Token length allocation mismatch")

    base_tokens = torch.arange(1, tree_tokens + 1, dtype=torch.long, device=device)
    max_len = max(lengths)
    input_ids = torch.full((batch_size, max_len), 0, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)

    sequences = []
    for idx, length in enumerate(lengths):
        seq_tokens = base_tokens[:length]
        input_ids[idx, :length] = seq_tokens
        attention_mask[idx, :length] = True
        sequences.append(seq_tokens.tolist())

    def _count_unique_nodes(seqs: list[list[int]]) -> int:
        root: dict[int, dict] = {}
        count = 0
        for seq in seqs:
            node = root
            for token in seq:
                if token not in node:
                    node[token] = {}
                    count += 1
                node = node[token]
        return count

    unique_nodes = _count_unique_nodes(sequences)
    if unique_nodes != tree_tokens:
        raise RuntimeError(
            f"Constructed tree has {unique_nodes} tokens, expected {tree_tokens}"
        )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def mock_loss_fn(logits: torch.Tensor, input_data: dict) -> torch.Tensor:
    """Mock loss function for testing."""
    return torch.mean(logits)


# Cannot use a "module" scope since process groups can only be initialized once.
@pytest.fixture
def engine():
    logger.info(f"megatron.core version={get_version('megatron.core')}")
    os.environ.update(
        {
            "WORLD_SIZE": "1",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "7777",
        }
    )
    config = TrainEngineConfig(
        experiment_name="test",
        trial_name="test",
        path=MODEL_PATH,
        mb_spec=MicroBatchSpec(max_tokens_per_mb=1024),
        optimizer=OptimizerConfig(),
        megatron=MegatronEngineConfig(
            use_deterministic_algorithms=True,
        ),
    )
    alloc_mode = AllocationMode.from_str("d1p1t1")
    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=128, train_batch_size=8)
    engine = MegatronEngine(config)
    engine.create_process_group(alloc_mode.train)
    engine.initialize(addr=None, ft_spec=ft_spec, parallel_strategy=alloc_mode.train)
    logger.info(f"mcore GPTModel initialized: {engine.model}")
    log_gpu_stats("initialize")
    try:
        yield engine
    finally:
        engine.destroy()
        assert not dist.is_initialized()


def test_simple_forward(engine, mock_input):
    engine.eval()
    result = engine.forward(mock_input)
    logger.info(f"Forward done, result: {result}")


def test_simple_train(engine, mock_input):
    engine.train()
    train_result = engine.train_batch(
        mock_input,
        loss_fn=mock_loss_fn,
        loss_weight_fn=lambda x: torch.tensor(1.0, device=engine.device),
    )
    engine.step_lr_scheduler()
    logger.info(f"Train done, result={train_result}")


def test_tree_training_forward(engine, mock_tree_input):
    for k, v in mock_tree_input.items():
        print(f"mock_tree_input[{k}].shape={v.shape}, dtype={v.dtype} v=\n{v}")

    def calc_logprobs(logits, input_data):
        labels = input_data.get(
            "rolled_input_ids",
            torch.roll(input_data["input_ids"], shifts=-1, dims=-1),
        )
        logprobs = gather_logprobs(logits, labels, 1.0)
        return logprobs

    engine.eval()
    logprob_baseline = engine.forward(
        input_=mock_tree_input,
        post_hook=calc_logprobs,
        aggregate_fn=lambda xs: torch.cat(xs, dim=-1),
    )
    config = TrainEngineConfig(
        experiment_name="test",
        trial_name="test",
        path=MODEL_PATH,
        mb_spec=MicroBatchSpec(max_tokens_per_mb=1024),
        optimizer=OptimizerConfig(),
        megatron=MegatronEngineConfig(
            enable_tree_training=True, use_deterministic_algorithms=True
        ),
    )
    tree_engine = MegatronEngine(config)
    alloc_mode = AllocationMode.from_str("d1p1t1")
    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=128, train_batch_size=8)
    tree_engine.create_process_group(alloc_mode.train)
    tree_engine.initialize(
        addr=None, ft_spec=ft_spec, parallel_strategy=alloc_mode.train
    )
    tree_engine.eval()
    logprob_tree = tree_engine.forward(
        input_=mock_tree_input,
        post_hook=calc_logprobs,
        aggregate_fn=lambda xs: torch.cat(xs, dim=-1),
    )

    print(f"logprob_baseline={logprob_baseline}")
    print(f"logprob_tree={logprob_tree}")
    # print where logprob baseline and logprob_tree are zeros
    print(
        f"logprob_baseline == 0 at positions: {(logprob_baseline == 0).nonzero(as_tuple=True)}"
    )
    print(
        f"logprob_tree == 0 at positions: {(logprob_tree == 0).nonzero(as_tuple=True)}"
    )

    # print where logprob_baseline and logprob_tree differ
    diff_positions = (logprob_baseline - logprob_tree).abs() > 1e-6
    print(
        f"Positions where logprob_baseline and logprob_tree differ: {diff_positions.nonzero(as_tuple=True)}"
    )
    print(f"diff = {logprob_baseline - logprob_tree}")
    assert torch.allclose(logprob_baseline, logprob_tree, atol=1e-6)


@torch.no_grad()
def test_hf_save_load_weights(tmp_path_factory, engine, mock_input):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    path = tmp_path_factory.mktemp("hf_engine_test")
    save_load_meta = SaveLoadMeta(
        path=path,
        weight_format="hf",
        tokenizer=tokenizer,
        with_optim=False,
        base_model_path=None,
    )

    old = engine.forward(input_=mock_input)
    start = time.perf_counter()
    engine.save(save_load_meta)
    logger.info(f"Save done, time cost: {time.perf_counter() - start:.4f} seconds.")
    for name, param in engine.model.named_parameters():
        param.zero_()

    start = time.perf_counter()
    engine.load(save_load_meta)
    logger.info(f"Load done, time cost: {time.perf_counter() - start:.4f} seconds.")
    new = engine.forward(input_=mock_input)
    assert torch.allclose(old, new)


@torch.no_grad()
@pytest.mark.slow
def test_dcp_save_load_weights(tmp_path_factory, engine, mock_input):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    path = tmp_path_factory.mktemp("megatron_engine_dcp_test")
    save_load_meta = SaveLoadMeta(
        path=path,
        weight_format="dcp",
        tokenizer=tokenizer,
        with_optim=True,
        base_model_path=None,
    )

    old = engine.forward(input_=mock_input)
    start = time.perf_counter()
    engine.save(save_load_meta)
    logger.info(f"Save done, time cost: {time.perf_counter() - start:.4f} seconds.")
    for name, param in engine.model.named_parameters():
        param.zero_()

    start = time.perf_counter()
    engine.load(save_load_meta)
    logger.info(f"Load done, time cost: {time.perf_counter() - start:.4f} seconds.")
    new = engine.forward(input_=mock_input)
    assert torch.allclose(old, new)
