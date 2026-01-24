# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AReaL is a large-scale asynchronous reinforcement learning system for training reasoning and agentic language models. It provides algorithm-first design with native async RL training, supporting both FSDP and Megatron backends for training, and SGLang/vLLM for inference.

## Common Commands

### Running Training

```bash
# Local single-node training
python3 -m areal.launcher.local examples/math/gsm8k_rl.py --config examples/math/gsm8k_grpo.yaml

# Ray cluster (multi-node)
python3 -m areal.launcher.ray examples/math/gsm8k_rl.py --config examples/math/gsm8k_grpo.yaml \
    cluster.n_nodes=2 cluster.n_gpus_per_node=8

# Slurm cluster
python3 -m areal.launcher.slurm examples/math/gsm8k_rl.py --config examples/math/gsm8k_grpo.yaml
```

### Development Setup

```bash
# Install with all development dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks (runs Ruff, mdformat, clang-format, nbstripout)
pip install pre-commit
pre-commit install
```

### Testing

```bash
# Run tests (use -sv --sw --lf for step-wise debugging)
pytest areal/tests/

# Run specific test file
pytest areal/tests/test_utils.py

# CI runs: pytest -m "not slow or ci"
```

Test markers: `slow` (>30s, skipped in CI unless also `ci`), `gpu`, `multi_gpu`

### Code Formatting

```bash
# Run all pre-commit checks
pre-commit run --all-files

# Regenerate CLI docs after changing cli_args.py
python docs/generate_cli_docs.py
```

## Architecture

### Directory Structure

- **`areal/`** - Core package
  - `api/` - Contracts: `engine_api.py` (TrainEngine, InferenceEngine), `workflow_api.py` (RolloutWorkflow), `cli_args.py` (config dataclasses)
  - `engine/` - Backend implementations: `fsdp_engine.py`, `megatron_engine.py`, `sglang_remote.py`, `vllm_remote.py`
  - `engine/ppo/` - PPO/GRPO algorithm implementations (actor.py, critic.py)
  - `workflow/` - Rollout workflows: `rlvr.py` (multi-sample), `multi_turn.py` (agentic), `vision_rlvr.py`
  - `launcher/` - Cluster launchers: `local.py`, `ray.py`, `slurm.py`
  - `dataset/` - Dataset loaders (gsm8k.py, etc.) - extend via `VALID_DATASETS` and `_get_custom_dataset`
  - `reward/` - Reward functions and math parsers
  - `utils/` - Logging, checkpointing (`saver.py`, `recover.py`), stats tracking
- **`examples/`** - Entry point scripts and YAML configs for math, multi-turn, VLM, alignment tasks
- **`realhf/`** - Legacy code (read-only, do not modify or import)
- **`csrc/`** - CUDA/C++ extensions

### Core Abstractions

1. **TrainEngine** (`areal/api/engine_api.py`) - SPMD distributed training interface with `train_batch()`, `forward()`, `update_weights()`, `save()`, `load()`

2. **InferenceEngine** (`areal/api/engine_api.py`) - Async generation interface with `agenerate()`, `update_weights()`

3. **RolloutWorkflow** (`areal/api/workflow_api.py`) - Implement `arun_episode(engine, data)` for trajectory collection. Must stay async, use `concat_padded_tensors` for outputs with shape `[batch, seq_len, ...]`

4. **PPOActor/Critic** (`areal/engine/ppo/`) - Algorithm implementations that wrap TrainEngine via composition

### Training Flow

Entry points (`examples/`) compose: Dataset → RolloutWorkflow (uses InferenceEngine) → PPOActor (uses TrainEngine) → training loop with `prepare_batch()` → `compute_advantages()` → `ppo_update()` → `update_weights()`

## Key Patterns

### Adding a Rollout Workflow

Subclass `RolloutWorkflow` in `areal/workflow/`, implement `arun_episode`. Use `AsyncRewardWrapper` for blocking rewards. Reference: `multi_turn.py`, `rlvr.py`.

### Adding a Reward Function

Create `areal/reward/<name>.py` with signature `(prompt, completions, prompt_ids, completion_ids, **data) -> scalar`. Add to `VALID_REWARD_FN` in `areal/reward/__init__.py`.

### Adding a Dataset

Create `areal/dataset/<name>.py` with `get_<name>_<type>_dataset` helpers. Add to `VALID_DATASETS` and `_get_custom_dataset` dispatch in `areal/dataset/__init__.py`.

### Configuration

Extend dataclasses in `areal/api/cli_args.py`. Use Hydra dotted keys for CLI overrides. YAML configs reference variables with `${...}` syntax.

## Important Notes

- Use `areal.utils.logging.getLogger(__name__)` not `print`
- Keep rollout workflows async-only (use `await`, `aiofiles`)
- Emit metrics via `stats_tracker`/`StatsLogger`
- Prefer composition over inheritance for algorithm implementations
- Leave `TODO(agent)` comments for unresolved constraints
