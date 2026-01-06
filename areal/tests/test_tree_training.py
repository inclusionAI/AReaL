import os
from importlib.metadata import version as get_version

import pytest
import torch
import torch.distributed as dist

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import (
    MegatronEngineConfig,
    MicroBatchSpec,
    OptimizerConfig,
    TrainEngineConfig,
)
from areal.api.io_struct import FinetuneSpec
from areal.engine.megatron_engine import MegatronEngine
from areal.platforms import current_platform
from areal.utils import logging

logger = logging.getLogger("MegatronEngine Test")


MODEL_PATH = "/storage/openpsi/models/Qwen__Qwen2-1.5B-Instruct/"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "Qwen/Qwen2-1.5B-Instruct"


@pytest.fixture(scope="module")
def mock_tree_input(
    batch_size=8,
    tree_tokens=1024,
    total_tokens=2048,
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
        mb_spec=MicroBatchSpec(max_tokens_per_mb=32768),
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
        mb_spec=MicroBatchSpec(max_tokens_per_mb=32768),
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
    logprob_tree = tree_engine.forward(input_=mock_tree_input)

    # Check if results match with detailed error reporting
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
        mb_spec=MicroBatchSpec(max_tokens_per_mb=32768),
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
        mb_spec=MicroBatchSpec(max_tokens_per_mb=32768),
        optimizer=OptimizerConfig(),
        megatron=MegatronEngineConfig(
            enable_tree_training=True,
            use_deterministic_algorithms=True,
        ),
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
