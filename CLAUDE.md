# CLAUDE.md — AReaL

## WHAT: Project Overview

AReaL is a distributed RL training framework for LLM alignment via reinforcement
learning.

**Tech Stack**: Python 3.12+ · PyTorch · FSDP2/Megatron · SGLang/vLLM

**Core Directories**:

- `areal/` — Core package
  - `api/` — Config dataclasses, workflow/engine contracts
  - `engine/` — FSDP2, Megatron, SGLang/vLLM adapters
  - `workflow/` — RolloutWorkflow implementations
  - `reward/` — Reward functions
  - `dataset/` — Dataset loaders
  - `utils/` — Logging, tensor ops, checkpoints
- `examples/` — Training scripts and configs
- `docs/` — Jupyter Book source

## WHY: Purpose

- Enable efficient RL training for LLM alignment at scale
- Async rollout + distributed training for high throughput
- Modular design: workflows, engines, rewards, and datasets are independently extensible

## HOW: Core Commands

```bash
# Check environment
python --version              # Requires 3.12+
uv --version                  # Install: https://docs.astral.sh/uv/

# Sync dependencies
uv sync --extra cuda          # With CUDA support (or `uv sync` without CUDA)
uv sync --group dev           # Include dev/test packages
uv run python3 areal/tools/validate_installation.py  # Validate installation

# Pre-commit hooks
pre-commit install            # Set up hooks (run once)
pre-commit run --all-files    # Format and lint

# Run tests
uv run pytest areal/tests/test_<topic>.py

# Generate CLI docs
uv run python docs/generate_cli_docs.py
```

## Boundaries

### Constraints

- Designed for distributed GPU clusters; assume containerized execution
- Integration tests require multi-node hardware; explain skips when unavailable
- Secrets and endpoints are managed outside the repo

### Always Do

- Read relevant files before modifying code
- Run `pre-commit run --all-files` before committing
- Follow existing code patterns in the same module
- Add tests for new functionality

### Ask First

- Modifying config structures in `areal/api/cli_args.py`
- Adding new dependencies
- Changing launcher or scheduler logic
- Deleting or renaming public APIs
- Running pytest tests (may require GPU/multi-node)

### Never Do

- Hardcode secrets, paths, or endpoints
- Skip pre-commit hooks
- Guess cluster configs or rebuild CUDA/driver stacks
- Use wildcard imports (`from x import *`)

## Progressive Disclosure: Detailed Guides

| Task                   | Reference                                                     |
| ---------------------- | ------------------------------------------------------------- |
| Add Workflow           | `docs/customization/agent.md`, `areal/workflow/multi_turn.py` |
| Add Dataset            | `docs/customization/`, `areal/dataset/gsm8k.py`               |
| Add Reward             | `areal/api/reward_api.py`, `areal/reward/geometry3k.py`       |
| Algorithm Details      | `docs/algorithms/*.md`                                        |
| Quickstart             | `docs/tutorial/quickstart.md`                                 |
| Architecture Deep Dive | `docs/lite/gsm8k_grpo.md`                                     |
| CLI Reference          | `docs/cli_reference.md`                                       |

## Git Workflow

- **Commits**: Conventional Commits (`feat:`, `fix:`, `docs:`), ~72 chars subject,
  imperative voice, reasoning in body
- **Squash**: Squash WIP commits before opening PR
- **PR requirements**: Run pre-commit, document test coverage, note hardware limitations

## Extended Configuration

See `.claude/agents/`, `.claude/commands/`, and `.claude/skills/` for specialized
instructions.

**Commands** (invoke with `/command`):

- `/pr-review` - Intelligent PR code review with dynamic agent allocation
- `/gen-commit-msg` - Generate commit message from staged changes
