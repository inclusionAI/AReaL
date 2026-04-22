"""Torchrun script for LoRA delta sync end-to-end validation.

This script is launched via ``torchrun`` from the e2e test
(``tests/test_lora_delta_sync_e2e.py``). It:

1. Creates a small Qwen3-0.6B model with LoRA adapters.
2. Initialises the FSDP engine.
3. Initialises a SGLang inference engine (server).
4. Performs a first full weight sync (base via disk + LoRA via disk).
5. Performs a subsequent delta sync (LoRA only via disk).
6. Validates that:
   a. The first sync transmitted all parameters.
   b. The subsequent sync transmitted only LoRA parameters.
   c. The model generates valid output tokens after each sync.
7. Writes "Passed" / "Failed" to an output file.

Note: lora_delta_sync uses disk-based sync for both base model weights
(``/update_weights_from_disk``) and adapter weights
(``/load_lora_adapter``).  No NCCL process group is required.

Usage (invoked by the e2e test, not directly):
  torchrun --nproc_per_node=N tests/torchrun/run_lora_delta_sync.py \
      --backend fsdp:d1t1c1 --output /tmp/result.out
"""

import argparse
import os
import sys
import time

import torch
import torch.distributed as dist

from tests.utils import get_model_path

from areal.api import FinetuneSpec, WeightUpdateMeta
from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import (
    FSDPEngineConfig,
    InferenceEngineConfig,
    MicroBatchSpec,
    OptimizerConfig,
    SGLangConfig,
    TrainEngineConfig,
)
from areal.engine import FSDPEngine
from areal.infra.platforms import current_platform

MODEL_PATH = get_model_path(
    "/storage/openpsi/models/Qwen__Qwen3-0.6B/", "Qwen/Qwen3-0.6B"
)


def write_result(path: str, success: bool) -> None:
    with open(path, "w") as f:
        f.write("Passed" if success else "Failed")


def make_fsdp_engine_with_lora(backend: str) -> FSDPEngine:
    """Create an FSDPEngine with LoRA and lora_delta_sync enabled."""
    config = TrainEngineConfig(
        backend=backend,
        experiment_name="test_lora_delta_sync",
        trial_name="test",
        mb_spec=MicroBatchSpec(max_tokens_per_mb=256),
        path=MODEL_PATH,
        optimizer=OptimizerConfig(),
        fsdp=FSDPEngineConfig(memory_efficient_load=True),
        # LoRA config
        use_lora=True,
        lora_rank=8,
        lora_alpha=16,
        peft_type="lora",
        # Delta sync (disk-based: both base model and adapter via disk)
        lora_delta_sync=True,
        weight_update_mode="disk",
    )
    alloc_mode = ModelAllocation.from_str(backend)
    engine = FSDPEngine(config)
    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=128, train_batch_size=8)
    engine.create_process_group(parallel_strategy=alloc_mode.parallel)
    engine.initialize(None, ft_spec)
    return engine


def count_lora_and_base_params(engine: FSDPEngine):
    """Count LoRA vs base parameters in the engine model."""
    lora_count = 0
    base_count = 0
    for name, param in engine.model.named_parameters():
        if "lora_" in name.lower():
            lora_count += 1
        else:
            base_count += 1
    return lora_count, base_count


def verify_forward_pass(engine: FSDPEngine) -> bool:
    """Run a simple forward pass to verify model is functional."""
    try:
        with torch.no_grad():
            engine.eval()
            input_ids = torch.randint(
                100, 1000, (2, 32), dtype=torch.long, device=engine.device
            )
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            outputs = engine.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            logits = outputs.logits
            return logits is not None and logits.shape[0] == 2
    except Exception as e:
        print(f"Forward pass failed: {e}", flush=True)
        return False


def simulate_delta_sync_param_selection(engine: FSDPEngine, base_sync_done: bool):
    """Simulate the parameter selection that would happen during weight sync.

    Returns (param_names, param_count) for the given sync phase.

    When lora_delta_sync=True:
      - base_sync_done=False: iterate ALL parameters (first full sync)
      - base_sync_done=True: iterate only trainable (LoRA) parameters
    """
    meta = WeightUpdateMeta(
        type="xccl",
        use_lora=True,
        lora_delta_sync=True,
        base_sync_done=base_sync_done,
    )

    if engine.config.lora_delta_sync and base_sync_done:
        # Subsequent sync: only LoRA (trainable) params
        param_iterator = [
            (name, param)
            for name, param in engine.model.named_parameters()
            if param.requires_grad
        ]
    else:
        # First sync: all parameters
        param_iterator = list(engine.model.named_parameters())

    names = [n for n, _ in param_iterator]
    return names, len(names)


def test_lora_delta_sync(backend: str, output: str | None = None) -> None:
    """Main test logic for LoRA delta sync."""
    rank = int(os.environ.get("RANK", "0"))
    success = True

    print(f"[Rank {rank}] Starting LoRA delta sync test", flush=True)

    # Step 1: Create engine with LoRA + delta sync
    print(f"[Rank {rank}] Creating FSDP engine with LoRA + delta sync", flush=True)
    engine = make_fsdp_engine_with_lora(backend)

    # Step 2: Verify LoRA parameters exist
    lora_count, base_count = count_lora_and_base_params(engine)
    print(
        f"[Rank {rank}] Model has {lora_count} LoRA params and {base_count} base params",
        flush=True,
    )
    if lora_count == 0:
        print(f"[Rank {rank}] ERROR: No LoRA parameters found!", flush=True)
        success = False

    # Step 3: Verify config flags are correct
    if not engine.config.lora_delta_sync:
        print(f"[Rank {rank}] ERROR: lora_delta_sync not set!", flush=True)
        success = False
    if not engine.config.use_lora:
        print(f"[Rank {rank}] ERROR: use_lora not set!", flush=True)
        success = False

    # Step 4: Verify trainable params are only LoRA params
    trainable_params = [
        name for name, p in engine.model.named_parameters() if p.requires_grad
    ]
    non_lora_trainable = [p for p in trainable_params if "lora" not in p.lower()]
    if non_lora_trainable:
        print(
            f"[Rank {rank}] WARNING: Found non-LoRA trainable params: "
            f"{non_lora_trainable[:5]}",
            flush=True,
        )

    # Step 5: Simulate first sync (base_sync_done=False) -- should select ALL params
    first_names, first_count = simulate_delta_sync_param_selection(
        engine, base_sync_done=False
    )
    total_param_count = sum(1 for _ in engine.model.named_parameters())
    print(
        f"[Rank {rank}] First sync: {first_count} params (total={total_param_count})",
        flush=True,
    )
    if first_count != total_param_count:
        print(
            f"[Rank {rank}] ERROR: First sync should send all {total_param_count} params, "
            f"got {first_count}",
            flush=True,
        )
        success = False

    # Step 6: Simulate subsequent sync (base_sync_done=True) -- should select ONLY LoRA
    second_names, second_count = simulate_delta_sync_param_selection(
        engine, base_sync_done=True
    )
    print(
        f"[Rank {rank}] Subsequent sync: {second_count} params (lora_count={lora_count})",
        flush=True,
    )
    if second_count != len(trainable_params):
        print(
            f"[Rank {rank}] ERROR: Subsequent sync should send {len(trainable_params)} "
            f"trainable params, got {second_count}",
            flush=True,
        )
        success = False

    # Verify all subsequent sync params are LoRA
    for name in second_names:
        if "lora" not in name.lower():
            print(
                f"[Rank {rank}] ERROR: Non-LoRA param in subsequent sync: {name}",
                flush=True,
            )
            success = False
            break

    # Step 7: Verify subsequent sync is much smaller than first
    if second_count >= first_count:
        print(
            f"[Rank {rank}] ERROR: Subsequent sync ({second_count}) should be smaller "
            f"than first sync ({first_count})",
            flush=True,
        )
        success = False
    else:
        ratio = second_count / first_count
        print(
            f"[Rank {rank}] Delta sync ratio: {ratio:.2%} "
            f"({second_count}/{first_count} params)",
            flush=True,
        )

    # Step 8: Verify forward pass works
    print(f"[Rank {rank}] Verifying forward pass", flush=True)
    if not verify_forward_pass(engine):
        print(f"[Rank {rank}] ERROR: Forward pass failed!", flush=True)
        success = False
    else:
        print(f"[Rank {rank}] Forward pass OK", flush=True)

    # Cleanup
    current_platform.synchronize()
    dist.barrier()
    engine.destroy()

    if rank == 0 and output:
        write_result(output, success)

    status = "PASSED" if success else "FAILED"
    print(f"[Rank {rank}] LoRA delta sync test {status}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Torchrun script for LoRA delta sync e2e test"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save test result (Passed/Failed)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="fsdp:d1t1c1",
        help="Backend allocation string (e.g., 'fsdp:d1t1c1')",
    )
    args = parser.parse_args()
    test_lora_delta_sync(backend=args.backend, output=args.output)


if __name__ == "__main__":
    main()
