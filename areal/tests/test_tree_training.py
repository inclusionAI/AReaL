import os
from importlib.metadata import version as get_version

import pytest
import torch
import torch.distributed as dist

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import (
    FSDPEngineConfig,
    MegatronEngineConfig,
    MicroBatchSpec,
    OptimizerConfig,
    TrainEngineConfig,
)
from areal.api.io_struct import FinetuneSpec
from areal.engine.fsdp_engine import FSDPEngine
from areal.engine.megatron_engine import MegatronEngine
from areal.models.tree_attn.tree import build_packed_tree_batch
from areal.platforms import current_platform
from areal.tests.utils import get_model_path
from areal.utils import logging

logger = logging.getLogger("MegatronEngine Test")


MODEL_PATH = get_model_path(
    "/storage/openpsi/models/Qwen__Qwen3-0.6B/", "Qwen/Qwen3-0.6B"
)


@pytest.fixture(scope="module")
def mock_tree_input(
    batch_size=4,
    tree_tokens=128,
    total_tokens=256,
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

    lengths = [int(length) for length in lengths]
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
    try:
        yield engine
    finally:
        engine.destroy()
        assert not dist.is_initialized()


def test_tree_training_forward(engine, mock_tree_input):
    engine.eval()
    logprob_baseline = engine.forward(
        input_=mock_tree_input,
        aggregate_fn=lambda xs: torch.cat(xs, dim=-1),
    )
    config = TrainEngineConfig(
        experiment_name="test",
        trial_name="test",
        path=MODEL_PATH,
        mb_spec=MicroBatchSpec(max_tokens_per_mb=1024),
        optimizer=OptimizerConfig(),
        megatron=MegatronEngineConfig(
            use_deterministic_algorithms=True
        ),
        enable_tree_training=True,
    )
    tree_engine = MegatronEngine(config)
    alloc_mode = AllocationMode.from_str("d1p1t1")
    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=128, train_batch_size=8)
    tree_engine.create_process_group(alloc_mode.train)
    tree_engine.initialize(
        addr=None, ft_spec=ft_spec, parallel_strategy=alloc_mode.train
    )
    tree_engine.eval()
    logprob_tree = tree_engine.forward(input_=mock_tree_input)

    # Check if results match with detailed error reporting
    # The tolenrance values are high due to precision problems introduced
    # by flex attention with customized attention masks.
    rtol, atol = 0.2, 0.2
    is_close = torch.isclose(logprob_tree, logprob_baseline, rtol=rtol, atol=atol)
    if not is_close.all():
        mismatched_mask = ~is_close
        num_mismatched = mismatched_mask.sum().item()
        total_elements = mismatched_mask.numel()
        mismatch_percentage = 100.0 * num_mismatched / total_elements

        # Get mismatched positions
        mismatched_indices = torch.nonzero(mismatched_mask, as_tuple=False)

        # Get values at mismatched positions (limit to first 10 for readability)
        num_to_show = min(10, num_mismatched)
        logger.error(
            f"Assertion failed: {num_mismatched}/{total_elements} elements mismatched ({mismatch_percentage:.2f}%)"
        )
        logger.error(f"First {num_to_show} mismatched positions and values:")
        for i in range(num_to_show):
            idx = tuple(mismatched_indices[i].tolist())
            tree_val = logprob_tree[idx].item()
            baseline_val = logprob_baseline[idx].item()
            abs_diff = abs(tree_val - baseline_val)
            rel_diff = abs_diff / (abs(baseline_val) + 1e-8)
            logger.error(
                f"  Position {idx}: tree={tree_val:.6f}, baseline={baseline_val:.6f}, "
                f"abs_diff={abs_diff:.6f}, rel_diff={rel_diff:.6f}"
            )

        # Summary statistics
        abs_diff_all = (logprob_tree - logprob_baseline).abs()
        logger.error(
            f"Overall abs diff: max={abs_diff_all.max().item():.6f}, "
            f"mean={abs_diff_all.mean().item():.6f}, median={abs_diff_all.median().item():.6f}"
        )

    assert is_close.all(), (
        f"logprob_tree and logprob_baseline differ: "
        f"{(~is_close).sum().item()}/{is_close.numel()} elements mismatched "
        f"({100.0 * (~is_close).sum().item() / is_close.numel():.2f}%)"
    )


def _collect_gradients(engine) -> dict[str, torch.Tensor]:
    grads = {}
    for model in engine.model:
        for name, param in model.named_parameters():
            # Megatron stores gradients in main_grad attribute
            if hasattr(param, "main_grad") and param.main_grad is not None:
                grads[name] = param.main_grad.clone()
            elif param.grad is not None:
                grads[name] = param.grad.clone()
    return grads


def _collect_parameters(engine) -> dict[str, torch.Tensor]:
    params = {}
    for model in engine.model:
        for name, param in model.named_parameters():
            params[name] = param.data.clone()
    return params


def _check_nan_params(params: dict[str, torch.Tensor], label: str) -> list[str]:
    nan_params = []
    for name, param in params.items():
        if torch.isnan(param).any():
            nan_count = torch.isnan(param).sum().item()
            total_count = param.numel()
            nan_params.append(name)
            print(f"  {name}: {nan_count}/{total_count} NaN values")
    if nan_params:
        print(f"\n⚠ NaN parameters in {label} ({len(nan_params)}):")
    return nan_params


def test_tree_training_forward_backward(mock_tree_input):
    def loss_fn(logprobs, entropy, input_data):
        return logprobs.sum()

    # ========== Setup baseline engine ==========
    os.environ.update(
        {
            "WORLD_SIZE": "1",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "7778",
        }
    )
    baseline_config = TrainEngineConfig(
        experiment_name="test_baseline",
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

    baseline_engine = MegatronEngine(baseline_config)
    baseline_engine.create_process_group(alloc_mode.train)
    baseline_engine.initialize(
        addr=None, ft_spec=ft_spec, parallel_strategy=alloc_mode.train
    )
    baseline_engine.train()

    # Run baseline forward-backward
    _ = baseline_engine.train_batch(
        mock_tree_input,
        loss_fn=loss_fn,
        loss_weight_fn=lambda x: torch.tensor(1.0, device=baseline_engine.device),
    )

    # Collect baseline gradients and updated parameters
    baseline_grads = _collect_gradients(baseline_engine)
    baseline_params = _collect_parameters(baseline_engine)

    logger.info(f"Collected {len(baseline_grads)} gradients from baseline engine")
    logger.info(f"Collected {len(baseline_params)} parameters from baseline engine")
    baseline_engine.destroy()

    # ========== Setup tree training engine ==========
    os.environ.update(
        {
            "WORLD_SIZE": "1",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "7779",
        }
    )
    tree_config = TrainEngineConfig(
        experiment_name="test_tree",
        trial_name="test",
        path=MODEL_PATH,
        mb_spec=MicroBatchSpec(max_tokens_per_mb=1024),
        optimizer=OptimizerConfig(),
        megatron=MegatronEngineConfig(
            use_deterministic_algorithms=True,
        ),
        enable_tree_training=True,
    )

    tree_engine = MegatronEngine(tree_config)
    tree_engine.create_process_group(alloc_mode.train)
    tree_engine.initialize(
        addr=None, ft_spec=ft_spec, parallel_strategy=alloc_mode.train
    )
    tree_engine.train()

    # Run tree training forward-backward
    _ = tree_engine.train_batch(
        mock_tree_input,
        loss_fn=loss_fn,
        loss_weight_fn=lambda x: torch.tensor(1.0, device=tree_engine.device),
    )

    # Collect tree training gradients and updated parameters
    tree_grads = _collect_gradients(tree_engine)
    tree_params = _collect_parameters(tree_engine)

    logger.info(f"Collected {len(tree_grads)} gradients from tree training engine")
    logger.info(f"Collected {len(tree_params)} parameters from tree training engine")
    tree_engine.destroy()

    # ========== Compare gradients ==========
    baseline_keys = set(baseline_grads.keys())
    tree_keys = set(tree_grads.keys())

    # Check for missing keys
    only_in_baseline = baseline_keys - tree_keys
    only_in_tree = tree_keys - baseline_keys

    if only_in_baseline:
        logger.warning(f"Gradients only in baseline: {only_in_baseline}")
    if only_in_tree:
        logger.warning(f"Gradients only in tree training: {only_in_tree}")

    common_keys = baseline_keys & tree_keys
    logger.info(f"Comparing {len(common_keys)} common gradient tensors")
    # Check for NaN gradients
    nan_in_baseline = []
    nan_in_tree = []
    # Check for zero gradients
    zero_in_baseline = []
    zero_in_tree = []
    for name in sorted(common_keys):
        if torch.isnan(baseline_grads[name]).any():
            nan_in_baseline.append(name)
        if torch.isnan(tree_grads[name]).any():
            nan_in_tree.append(name)
        # Check if all gradients are zero
        if (baseline_grads[name] == 0).all():
            zero_in_baseline.append(name)
        if (tree_grads[name] == 0).all():
            zero_in_tree.append(name)

    if nan_in_baseline:
        logger.info(f"\n⚠ NaN gradients in BASELINE ({len(nan_in_baseline)}):")
        for name in nan_in_baseline:
            nan_count = torch.isnan(baseline_grads[name]).sum().item()
            total_count = baseline_grads[name].numel()
            logger.info(f"  {name}: {nan_count}/{total_count} NaN values")

    if nan_in_tree:
        logger.info(f"\n⚠ NaN gradients in TREE TRAINING ({len(nan_in_tree)}):")
        for name in nan_in_tree:
            nan_count = torch.isnan(tree_grads[name]).sum().item()
            total_count = tree_grads[name].numel()
            logger.info(f"  {name}: {nan_count}/{total_count} NaN values")

    if zero_in_baseline:
        logger.info(f"\n⚠ Zero gradients in BASELINE ({len(zero_in_baseline)}):")
        for name in zero_in_baseline:
            logger.info(f"  {name}: all {baseline_grads[name].numel()} values are zero")

    if zero_in_tree:
        logger.info(f"\n⚠ Zero gradients in TREE TRAINING ({len(zero_in_tree)}):")
        for name in zero_in_tree:
            logger.info(f"  {name}: all {tree_grads[name].numel()} values are zero")

    # Check for NaN in updated parameters
    nan_params_baseline = _check_nan_params(baseline_params, "BASELINE PARAMS")
    nan_params_tree = _check_nan_params(tree_params, "TREE TRAINING PARAMS")

    mismatched_params = []
    max_diff_overall = 0.0

    for name in sorted(common_keys):
        baseline_grad = baseline_grads[name]
        tree_grad = tree_grads[name]

        if baseline_grad.shape != tree_grad.shape:
            mismatched_params.append(
                (name, f"shape mismatch: {baseline_grad.shape} vs {tree_grad.shape}")
            )
            continue

        diff = (baseline_grad - tree_grad).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        max_diff_overall = max(max_diff_overall, max_diff)

        if mean_diff > 1e-3:
            mismatched_params.append(
                (name, f"max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")
            )
            logger.info(
                f"Gradient mismatch for {name}:"
                f"Shape: {baseline_grad.shape}"
                f"Baseline grad mean: {baseline_grad.float().mean().item():.6e}"
                f"Tree grad mean: {tree_grad.float().mean().item():.6e}"
                f"Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}"
            )

    assert len(only_in_baseline) == 0, (
        f"Gradients missing in tree training: {only_in_baseline}"
    )
    assert len(only_in_tree) == 0, f"Gradients missing in baseline: {only_in_tree}"
    assert len(nan_in_baseline) == 0, f"NaN gradients in baseline: {nan_in_baseline}"
    assert len(nan_in_tree) == 0, f"NaN gradients in tree training: {nan_in_tree}"
    assert len(zero_in_baseline) == 0, f"Zero gradients in baseline: {zero_in_baseline}"
    assert len(zero_in_tree) == 0, f"Zero gradients in tree training: {zero_in_tree}"
    assert len(nan_params_baseline) == 0, (
        f"NaN parameters in baseline: {nan_params_baseline}"
    )
    assert len(nan_params_tree) == 0, (
        f"NaN parameters in tree training: {nan_params_tree}"
    )
    assert len(mismatched_params) == 0, (
        f"Gradient mismatches found: {mismatched_params}"
    )


# =============================================================================
# Tests for n_mbs and n_mbs_divisor in tree packing
# =============================================================================


def _create_test_input(
    batch_size: int,
    seq_lengths: list[int],
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Create test input data with specified sequence lengths.

    Args:
        batch_size: Number of sequences.
        seq_lengths: List of sequence lengths for each sequence.
        device: Device for tensors.

    Returns:
        Dictionary with 'input_ids' and 'attention_mask' tensors.
    """
    assert len(seq_lengths) == batch_size
    max_len = max(seq_lengths)

    input_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)

    for i, length in enumerate(seq_lengths):
        # Use unique tokens for each sequence to avoid sharing
        input_ids[i, :length] = torch.arange(
            i * 1000, i * 1000 + length, dtype=torch.long, device=device
        )
        attention_mask[i, :length] = True

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def _create_shared_prefix_input(
    batch_size: int,
    prefix_length: int,
    suffix_lengths: list[int],
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Create test input where all sequences share a common prefix.

    Args:
        batch_size: Number of sequences.
        prefix_length: Length of the shared prefix.
        suffix_lengths: List of suffix lengths for each sequence.
        device: Device for tensors.

    Returns:
        Dictionary with 'input_ids' and 'attention_mask' tensors.
    """
    assert len(suffix_lengths) == batch_size
    seq_lengths = [prefix_length + s for s in suffix_lengths]
    max_len = max(seq_lengths)

    input_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)

    # Shared prefix tokens
    prefix_tokens = torch.arange(1, prefix_length + 1, dtype=torch.long, device=device)

    for i, (length, suffix_len) in enumerate(zip(seq_lengths, suffix_lengths)):
        # Shared prefix
        input_ids[i, :prefix_length] = prefix_tokens
        # Unique suffix for each sequence
        if suffix_len > 0:
            input_ids[i, prefix_length:length] = torch.arange(
                1000 + i * 100, 1000 + i * 100 + suffix_len, dtype=torch.long, device=device
            )
        attention_mask[i, :length] = True

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def test_build_packed_tree_batch_n_mbs_minimum():
    """Test that n_mbs enforces minimum number of trees."""
    # Create input with 8 sequences that would naturally pack into fewer trees
    # Each sequence has unique tokens to avoid prefix sharing
    data = _create_test_input(
        batch_size=8,
        seq_lengths=[50, 50, 50, 50, 50, 50, 50, 50],
    )

    # With large max_tokens_per_mb, all sequences would fit in 1 tree
    # But n_mbs=4 should force at least 4 trees
    mb_spec = MicroBatchSpec(
        max_tokens_per_mb=10240,  # 80 * 128
        n_mbs=4,
        n_mbs_divisor=1,
    )

    result = build_packed_tree_batch(data, mb_spec, pad_to_maximum=True)

    assert len(result) >= 4, (
        f"Expected at least 4 trees (n_mbs=4), got {len(result)}"
    )


def test_build_packed_tree_batch_n_mbs_divisor():
    """Test that n_mbs_divisor ensures tree count is divisible."""
    # Create input with 5 sequences that can be grouped together
    # Each sequence has unique tokens to avoid prefix sharing
    data = _create_test_input(
        batch_size=5,
        seq_lengths=[100, 100, 100, 100, 100],
    )

    # With max_tokens_per_mb=512, sequences can be grouped (up to 5 per tree)
    # This would naturally create 1 tree with all 5 sequences
    # n_mbs_divisor=2 should force splitting to get an even number (2 trees)
    mb_spec = MicroBatchSpec(
        max_tokens_per_mb=512,  # 4 * 128
        n_mbs=1,
        n_mbs_divisor=2,
    )

    result = build_packed_tree_batch(data, mb_spec, pad_to_maximum=True)

    assert len(result) % 2 == 0, (
        f"Expected tree count divisible by 2 (n_mbs_divisor=2), got {len(result)}"
    )


def test_build_packed_tree_batch_n_mbs_and_divisor_combined():
    """Test that n_mbs and n_mbs_divisor work together correctly."""
    # Create input with 6 sequences
    data = _create_test_input(
        batch_size=6,
        seq_lengths=[80, 80, 80, 80, 80, 80],
    )

    # n_mbs=5 (minimum 5 trees), n_mbs_divisor=3 (must be divisible by 3)
    # Result should be 6 trees (next multiple of 3 >= 5)
    mb_spec = MicroBatchSpec(
        max_tokens_per_mb=128,  # 1 * 128
        n_mbs=5,
        n_mbs_divisor=3,
    )

    result = build_packed_tree_batch(data, mb_spec, pad_to_maximum=True)

    assert len(result) >= 5, (
        f"Expected at least 5 trees (n_mbs=5), got {len(result)}"
    )
    assert len(result) % 3 == 0, (
        f"Expected tree count divisible by 3 (n_mbs_divisor=3), got {len(result)}"
    )


def test_build_packed_tree_batch_default_values():
    """Test that default n_mbs=1 and n_mbs_divisor=1 work correctly."""
    # Create input that would naturally pack into 1 tree
    data = _create_shared_prefix_input(
        batch_size=4,
        prefix_length=50,
        suffix_lengths=[10, 10, 10, 10],
    )

    mb_spec = MicroBatchSpec(
        max_tokens_per_mb=10240,  # 80 * 128
        # n_mbs and n_mbs_divisor default to 1
    )

    result = build_packed_tree_batch(data, mb_spec, pad_to_maximum=True)

    # With shared prefix, all sequences should pack into 1 tree
    assert len(result) >= 1, f"Expected at least 1 tree, got {len(result)}"


def test_build_packed_tree_batch_cannot_split_raises_error():
    """Test that RuntimeError is raised when trees cannot be split to meet requirements."""
    # Create input with only 2 sequences - can only split to 2 trees max
    data = _create_test_input(
        batch_size=2,
        seq_lengths=[50, 50],
    )

    # Request 4 trees, but only 2 sequences available
    # This should raise RuntimeError since we can't create 4 trees from 2 sequences
    mb_spec = MicroBatchSpec(
        max_tokens_per_mb=128,  # 1 * 128
        n_mbs=4,
        n_mbs_divisor=1,
    )

    with pytest.raises(RuntimeError, match="Cannot split trees to meet n_mbs"):
        build_packed_tree_batch(data, mb_spec, pad_to_maximum=True)


def test_build_packed_tree_batch_cannot_split_divisor_raises_error():
    """Test that RuntimeError is raised when n_mbs_divisor cannot be satisfied."""
    # Create input with 3 sequences, each getting its own tree
    data = _create_test_input(
        batch_size=3,
        seq_lengths=[100, 100, 100],
    )

    # With max_tokens_per_mb=128, each sequence gets its own tree (3 trees)
    # n_mbs_divisor=2 requires even number, but 3 trees can't be split (1 seq each)
    # This should raise RuntimeError
    mb_spec = MicroBatchSpec(
        max_tokens_per_mb=128,  # 1 * 128
        n_mbs=1,
        n_mbs_divisor=2,
    )

    with pytest.raises(RuntimeError, match="Cannot split trees to meet"):
        build_packed_tree_batch(data, mb_spec, pad_to_maximum=True)


def test_build_packed_tree_batch_max_tokens_still_respected():
    """Test that max_tokens_per_mb is still respected when splitting."""
    # Create input with sequences that exceed max_tokens_per_mb individually
    data = _create_test_input(
        batch_size=4,
        seq_lengths=[100, 100, 100, 100],
    )

    # max_tokens_per_mb=128 means at most ~1 sequence per tree
    mb_spec = MicroBatchSpec(
        max_tokens_per_mb=128,  # 1 * 128
        n_mbs=2,
        n_mbs_divisor=1,
    )

    result = build_packed_tree_batch(data, mb_spec, pad_to_maximum=True)

    # Each tree should respect max_tokens_per_mb
    for i, mb in enumerate(result.mbs):
        if "trie_node" in mb:
            tree_tokens = mb["trie_node"].num_tokens
            assert tree_tokens <= 128, (
                f"Tree {i} has {tree_tokens} tokens, exceeds max_tokens_per_mb=128"
            )


# =============================================================================
# Multiprocessing test for dp_group synchronization
# =============================================================================


def _dp_group_worker(
    rank: int,
    world_size: int,
    backend: str,
    result_queue,
    data_per_rank: list[dict[str, torch.Tensor]],
    max_tokens_per_mb: int,
):
    """Worker function for distributed dp_group test.

    Each rank runs build_packed_tree_batch with different input data
    and validates that the number of trees is synchronized across ranks.
    """
    import torch.multiprocessing as mp

    try:
        # Set environment variables for distributed
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)

        # Initialize process group
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
        )

        # Set device
        device = f"cuda:{rank}"
        torch.cuda.set_device(device)

        # Get data for this rank and move to GPU
        data = {
            k: v.to(device) for k, v in data_per_rank[rank].items()
        }

        # Create mb_spec
        mb_spec = MicroBatchSpec(
            max_tokens_per_mb=max_tokens_per_mb,
            n_mbs=1,
            n_mbs_divisor=1,
        )

        # Get the default process group as dp_group
        dp_group = dist.distributed_c10d._get_default_group()

        # Run build_packed_tree_batch with dp_group
        result = build_packed_tree_batch(
            data,
            mb_spec,
            pad_to_maximum=True,
            dp_group=dp_group,
        )

        num_trees = len(result)

        # All-gather to verify all ranks have same number of trees
        local_count = torch.tensor([num_trees], dtype=torch.int64, device=device)
        all_counts = [
            torch.zeros(1, dtype=torch.int64, device=device)
            for _ in range(world_size)
        ]
        dist.all_gather(all_counts, local_count)

        all_tree_counts = [c.item() for c in all_counts]

        # Put result in queue
        result_queue.put({
            "rank": rank,
            "num_trees": num_trees,
            "all_tree_counts": all_tree_counts,
            "success": True,
            "error": None,
        })

    except Exception as e:
        import traceback
        result_queue.put({
            "rank": rank,
            "num_trees": -1,
            "all_tree_counts": [],
            "success": False,
            "error": f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
        })

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Requires at least 2 GPUs"
)
def test_build_packed_tree_batch_dp_group_sync():
    """Test that dp_group synchronizes tree count across ranks.

    This test spawns 2 processes (one per GPU) with different input data:
    - Rank 0: 2 sequences that fit in 1 tree
    - Rank 1: 4 sequences that require 2 trees

    With dp_group synchronization, both ranks should produce 2 trees.
    """
    import torch.multiprocessing as mp

    world_size = 2
    backend = "nccl"
    max_tokens_per_mb = 256  # 2 * 128

    # Create different data for each rank on CPU (will be moved to GPU in worker)
    # Rank 0: 2 sequences, total ~100 tokens -> fits in 1 tree
    data_rank0 = _create_test_input(
        batch_size=2,
        seq_lengths=[50, 50],
        device="cpu",
    )

    # Rank 1: 4 sequences, total ~400 tokens -> needs 2 trees (256 max per tree)
    data_rank1 = _create_test_input(
        batch_size=4,
        seq_lengths=[100, 100, 100, 100],
        device="cpu",
    )

    data_per_rank = [data_rank0, data_rank1]

    # Use spawn context for CUDA
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    processes = []
    for rank in range(world_size):
        p = ctx.Process(
            target=_dp_group_worker,
            args=(rank, world_size, backend, result_queue, data_per_rank, max_tokens_per_mb),
        )
        p.start()
        processes.append(p)

    # Collect results
    results = []
    for _ in range(world_size):
        results.append(result_queue.get(timeout=60))

    # Wait for processes to finish
    for p in processes:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()
            p.join()

    # Sort results by rank
    results.sort(key=lambda r: r["rank"])

    # Check for errors
    for r in results:
        if not r["success"]:
            pytest.fail(f"Rank {r['rank']} failed: {r['error']}")

    # Verify all ranks have the same number of trees
    tree_counts = [r["num_trees"] for r in results]
    assert len(set(tree_counts)) == 1, (
        f"Tree counts should be identical across ranks, got {tree_counts}"
    )

    # Verify the synchronized count is the maximum (rank 1 needed 2 trees)
    assert tree_counts[0] >= 2, (
        f"Expected at least 2 trees after sync, got {tree_counts[0]}"
    )

    # Verify all_tree_counts are consistent
    for r in results:
        assert r["all_tree_counts"] == tree_counts, (
            f"Rank {r['rank']} all_tree_counts mismatch: {r['all_tree_counts']} vs {tree_counts}"
        )


# =============================================================================
# FSDP Engine Tree Training Tests
# =============================================================================

fsdp_logger = logging.getLogger("FSDPEngine Test")


@pytest.fixture
def fsdp_engine():
    fsdp_logger.info(f"torch version={torch.__version__}")
    os.environ.update(
        {
            "WORLD_SIZE": "1",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "7780",
        }
    )
    config = TrainEngineConfig(
        experiment_name="test",
        trial_name="test",
        path=MODEL_PATH,
        mb_spec=MicroBatchSpec(max_tokens_per_mb=1024),
        optimizer=OptimizerConfig(),
        fsdp=FSDPEngineConfig(),
    )
    alloc_mode = AllocationMode.from_str("d1p1t1")
    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=128, train_batch_size=8)
    engine = FSDPEngine(config)
    engine.create_process_group(alloc_mode.train)
    engine.initialize(addr=None, ft_spec=ft_spec, parallel_strategy=alloc_mode.train)
    fsdp_logger.info(f"FSDP Model initialized: {engine.model}")
    try:
        yield engine
    finally:
        engine.destroy()
        assert not dist.is_initialized()


def _collect_fsdp_gradients(engine: FSDPEngine) -> dict[str, torch.Tensor]:
    """Collect gradients from FSDP engine."""
    grads = {}
    for name, param in engine.model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.clone()
    return grads


def _collect_fsdp_parameters(engine: FSDPEngine) -> dict[str, torch.Tensor]:
    """Collect parameters from FSDP engine."""
    params = {}
    for name, param in engine.model.named_parameters():
        params[name] = param.data.clone()
    return params


def test_fsdp_tree_training_forward(fsdp_engine, mock_tree_input):
    """Test FSDP tree training forward pass produces correct logprobs."""
    fsdp_engine.eval()
    logprob_baseline = fsdp_engine.forward_batch(
        input_=mock_tree_input,
        aggregate_fn=lambda xs: torch.cat(xs, dim=-1),
    )

    # Create tree training FSDP engine
    os.environ.update(
        {
            "WORLD_SIZE": "1",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "7781",
        }
    )
    config = TrainEngineConfig(
        experiment_name="test",
        trial_name="test",
        path=MODEL_PATH,
        mb_spec=MicroBatchSpec(max_tokens_per_mb=1024),
        optimizer=OptimizerConfig(),
        fsdp=FSDPEngineConfig(),
        enable_tree_training=True,
    )
    tree_engine = FSDPEngine(config)
    alloc_mode = AllocationMode.from_str("d1p1t1")
    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=128, train_batch_size=8)
    tree_engine.create_process_group(alloc_mode.train)
    tree_engine.initialize(
        addr=None, ft_spec=ft_spec, parallel_strategy=alloc_mode.train
    )
    tree_engine.eval()
    logprob_tree = tree_engine.forward_batch(input_=mock_tree_input)

    # Check if results match with detailed error reporting
    # The tolerance values are high due to precision problems introduced
    # by flex attention with customized attention masks.
    rtol, atol = 0.2, 0.2
    is_close = torch.isclose(logprob_tree, logprob_baseline, rtol=rtol, atol=atol)
    if not is_close.all():
        mismatched_mask = ~is_close
        num_mismatched = mismatched_mask.sum().item()
        total_elements = mismatched_mask.numel()
        mismatch_percentage = 100.0 * num_mismatched / total_elements

        # Get mismatched positions
        mismatched_indices = torch.nonzero(mismatched_mask, as_tuple=False)

        # Get values at mismatched positions (limit to first 10 for readability)
        num_to_show = min(10, num_mismatched)
        fsdp_logger.error(
            f"Assertion failed: {num_mismatched}/{total_elements} elements mismatched ({mismatch_percentage:.2f}%)"
        )
        fsdp_logger.error(f"First {num_to_show} mismatched positions and values:")
        for i in range(num_to_show):
            idx = tuple(mismatched_indices[i].tolist())
            tree_val = logprob_tree[idx].item()
            baseline_val = logprob_baseline[idx].item()
            abs_diff = abs(tree_val - baseline_val)
            rel_diff = abs_diff / (abs(baseline_val) + 1e-8)
            fsdp_logger.error(
                f"  Position {idx}: tree={tree_val:.6f}, baseline={baseline_val:.6f}, "
                f"abs_diff={abs_diff:.6f}, rel_diff={rel_diff:.6f}"
            )

        # Summary statistics
        abs_diff_all = (logprob_tree - logprob_baseline).abs()
        fsdp_logger.error(
            f"Overall abs diff: max={abs_diff_all.max().item():.6f}, "
            f"mean={abs_diff_all.mean().item():.6f}, median={abs_diff_all.median().item():.6f}"
        )

    tree_engine.destroy()

    assert is_close.all(), (
        f"logprob_tree and logprob_baseline differ: "
        f"{(~is_close).sum().item()}/{is_close.numel()} elements mismatched "
        f"({100.0 * (~is_close).sum().item() / is_close.numel():.2f}%)"
    )


def test_fsdp_tree_training_forward_backward(mock_tree_input):
    """Test FSDP tree training forward-backward pass produces correct gradients."""
    def loss_fn(logprobs, entropy, input_data):
        return logprobs.sum()

    # ========== Setup baseline FSDP engine ==========
    os.environ.update(
        {
            "WORLD_SIZE": "1",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "7782",
        }
    )
    baseline_config = TrainEngineConfig(
        experiment_name="test_baseline",
        trial_name="test",
        path=MODEL_PATH,
        mb_spec=MicroBatchSpec(max_tokens_per_mb=1024),
        optimizer=OptimizerConfig(),
        fsdp=FSDPEngineConfig(),
    )
    alloc_mode = AllocationMode.from_str("d1p1t1")
    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=128, train_batch_size=8)

    baseline_engine = FSDPEngine(baseline_config)
    baseline_engine.create_process_group(alloc_mode.train)
    baseline_engine.initialize(
        addr=None, ft_spec=ft_spec, parallel_strategy=alloc_mode.train
    )
    baseline_engine.train()

    # Run baseline forward-backward
    _ = baseline_engine.train_batch(
        mock_tree_input,
        loss_fn=loss_fn,
        loss_weight_fn=lambda x: torch.tensor(1.0, device=baseline_engine.device),
    )

    # Collect baseline gradients and parameters
    baseline_grads = _collect_fsdp_gradients(baseline_engine)
    baseline_params = _collect_fsdp_parameters(baseline_engine)

    fsdp_logger.info(f"Collected {len(baseline_grads)} gradients from baseline FSDP engine")
    fsdp_logger.info(f"Collected {len(baseline_params)} parameters from baseline FSDP engine")
    baseline_engine.destroy()

    # ========== Setup tree training FSDP engine ==========
    os.environ.update(
        {
            "WORLD_SIZE": "1",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "7783",
        }
    )
    tree_config = TrainEngineConfig(
        experiment_name="test_tree",
        trial_name="test",
        path=MODEL_PATH,
        mb_spec=MicroBatchSpec(max_tokens_per_mb=1024),
        optimizer=OptimizerConfig(),
        fsdp=FSDPEngineConfig(),
        enable_tree_training=True,
    )

    tree_engine = FSDPEngine(tree_config)
    tree_engine.create_process_group(alloc_mode.train)
    tree_engine.initialize(
        addr=None, ft_spec=ft_spec, parallel_strategy=alloc_mode.train
    )
    tree_engine.train()

    # Run tree training forward-backward
    _ = tree_engine.train_batch(
        mock_tree_input,
        loss_fn=loss_fn,
        loss_weight_fn=lambda x: torch.tensor(1.0, device=tree_engine.device),
    )

    # Collect tree training gradients and parameters
    tree_grads = _collect_fsdp_gradients(tree_engine)
    tree_params = _collect_fsdp_parameters(tree_engine)

    fsdp_logger.info(f"Collected {len(tree_grads)} gradients from tree training FSDP engine")
    fsdp_logger.info(f"Collected {len(tree_params)} parameters from tree training FSDP engine")
    tree_engine.destroy()

    # ========== Compare gradients ==========
    baseline_keys = set(baseline_grads.keys())
    tree_keys = set(tree_grads.keys())

    # Check for missing keys
    only_in_baseline = baseline_keys - tree_keys
    only_in_tree = tree_keys - baseline_keys

    if only_in_baseline:
        fsdp_logger.warning(f"Gradients only in baseline: {only_in_baseline}")
    if only_in_tree:
        fsdp_logger.warning(f"Gradients only in tree training: {only_in_tree}")

    common_keys = baseline_keys & tree_keys
    fsdp_logger.info(f"Comparing {len(common_keys)} common gradient tensors")

    # Check for NaN and zero gradients
    nan_in_baseline = []
    nan_in_tree = []
    zero_in_baseline = []
    zero_in_tree = []

    for name in sorted(common_keys):
        if torch.isnan(baseline_grads[name]).any():
            nan_in_baseline.append(name)
        if torch.isnan(tree_grads[name]).any():
            nan_in_tree.append(name)
        if (baseline_grads[name] == 0).all():
            zero_in_baseline.append(name)
        if (tree_grads[name] == 0).all():
            zero_in_tree.append(name)

    if nan_in_baseline:
        fsdp_logger.info(f"\n⚠ NaN gradients in BASELINE ({len(nan_in_baseline)}):")
        for name in nan_in_baseline:
            nan_count = torch.isnan(baseline_grads[name]).sum().item()
            total_count = baseline_grads[name].numel()
            fsdp_logger.info(f"  {name}: {nan_count}/{total_count} NaN values")

    if nan_in_tree:
        fsdp_logger.info(f"\n⚠ NaN gradients in TREE TRAINING ({len(nan_in_tree)}):")
        for name in nan_in_tree:
            nan_count = torch.isnan(tree_grads[name]).sum().item()
            total_count = tree_grads[name].numel()
            fsdp_logger.info(f"  {name}: {nan_count}/{total_count} NaN values")

    # Check for NaN in updated parameters
    nan_params_baseline = _check_nan_params(baseline_params, "BASELINE FSDP PARAMS")
    nan_params_tree = _check_nan_params(tree_params, "TREE TRAINING FSDP PARAMS")

    mismatched_params = []
    max_diff_overall = 0.0

    for name in sorted(common_keys):
        baseline_grad = baseline_grads[name]
        tree_grad = tree_grads[name]

        if baseline_grad.shape != tree_grad.shape:
            mismatched_params.append(
                (name, f"shape mismatch: {baseline_grad.shape} vs {tree_grad.shape}")
            )
            continue

        diff = (baseline_grad - tree_grad).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        max_diff_overall = max(max_diff_overall, max_diff)

        if mean_diff > 1e-3:
            mismatched_params.append(
                (name, f"max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")
            )
            fsdp_logger.info(
                f"Gradient mismatch for {name}: "
                f"Shape: {baseline_grad.shape}, "
                f"Baseline grad mean: {baseline_grad.float().mean().item():.6e}, "
                f"Tree grad mean: {tree_grad.float().mean().item():.6e}, "
                f"Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}"
            )

    assert len(only_in_baseline) == 0, (
        f"Gradients missing in tree training: {only_in_baseline}"
    )
    assert len(only_in_tree) == 0, f"Gradients missing in baseline: {only_in_tree}"
    assert len(nan_in_baseline) == 0, f"NaN gradients in baseline: {nan_in_baseline}"
    assert len(nan_in_tree) == 0, f"NaN gradients in tree training: {nan_in_tree}"
    assert len(nan_params_baseline) == 0, (
        f"NaN parameters in baseline: {nan_params_baseline}"
    )
    assert len(nan_params_tree) == 0, (
        f"NaN parameters in tree training: {nan_params_tree}"
    )
    assert len(mismatched_params) == 0, (
        f"Gradient mismatches found: {mismatched_params}"
    )

