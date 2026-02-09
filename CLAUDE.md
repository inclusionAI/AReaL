# CLAUDE.md - AReaL

## WHAT: Project Overview

AReaL is a distributed RL training framework for LLM alignment via reinforcement
learning.

**Tech Stack**: Python 3.12+ · PyTorch · FSDP2/Megatron · SGLang/vLLM

**Core Directories**:

- `areal/` - Core package
  - `api/` - Config dataclasses, workflow/engine contracts
  - `engine/` - FSDP2, Megatron, SGLang/vLLM adapters
  - `workflow/` - RolloutWorkflow implementations
  - `reward/` - Reward functions
  - `dataset/` - Dataset loaders
  - `utils/` - Logging, tensor ops, checkpoints
- `examples/` - Training scripts and configs
- `docs/` - Jupyter Book source

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
# First check GPU availability (many tests require GPU)
python -c "import torch; print('GPU available:', torch.cuda.is_available())"
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
- Running GPU/distributed tests (check GPU first:
  `python -c "import torch; print('GPU available:', torch.cuda.is_available())"`)

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
| Add Archon Model       | `areal/experimental/models/archon/qwen2/`, `qwen3/`           |
| Algorithm Details      | `docs/algorithms/*.md`                                        |
| Quickstart             | `docs/tutorial/quickstart.md`                                 |
| Architecture Deep Dive | `docs/tutorial/gsm8k_grpo.md`                                 |
| CLI Reference          | `docs/cli_reference.md`                                       |

## Git Workflow

- **Commits**: Conventional Commits (`feat:`, `fix:`, `docs:`), ~72 chars subject,
  imperative voice, reasoning in body
- **Squash**: Squash WIP commits before opening PR
- **PR requirements**: Run pre-commit, document test coverage, note hardware limitations

## Extended Configuration

See `.claude/agents/`, `.claude/skills/`, `.claude/commands/`, and `.claude/rules/` for
specialized instructions.

### Agents

| Agent                       | Purpose                                   | Activation Trigger                                                  |
| --------------------------- | ----------------------------------------- | ------------------------------------------------------------------- |
| `planner`                   | Implementation planning                   | Before multi-file changes, new features, or architectural decisions |
| `simple-code-reviewer`      | Quick code quality checks                 | After code changes, before committing                               |
| `code-verifier`             | Formatting/linting/tests                  | After code changes, before committing                               |
| `fsdp-engine-expert`        | FSDPEngine implementation                 | FSDPEngine code changes or questions                                |
| `archon-engine-expert`      | ArchonEngine implementation               | ArchonEngine code changes or questions                              |
| `megatron-engine-expert`    | MegatronEngine implementation             | MegatronEngine code changes or questions                            |
| `algorithm-expert`          | RL algorithms                             | GRPO/PPO/DAPO questions                                             |
| `launcher-scheduler-expert` | Cluster launching and resource scheduling | Launcher/scheduler code changes or configuration questions          |

**Stage-by-Stage Agent Guidance**:

1. **Planning Stage** (Before coding): Use `planner` for architecture design and
   implementation planning
1. **Code Formatting & Linting** (After coding): Use `code-verifier` to automatically
   run formatting, linting, and tests, catching syntax errors and style issues quickly
1. **Code Quality Check** (After formatting): Use `simple-code-reviewer` for quick code
   quality checks, focusing on logic issues and code smells

### Skills (Guided Development Workflows)

Skills provide step-by-step guides for common development tasks:

- `/add-dataset` - Dataset loader creation guide
- `/add-workflow` - Workflow implementation guide
- `/add-reward` - Reward function guide
- `/add-archon-model` - Archon engine model architecture guide
- `/debug-distributed` - Distributed debugging guide
- `/add-unit-tests` - Test development guide (NEW)

### Commands (User-invoked Actions)

Commands perform specific actions when invoked:

- `/create-pr` - Rebase, squash commits, and create/update PR with intelligent messages
- `/gen-commit-msg` - Generate commit messages from staged changes
- `/pr-review` - Intelligent PR code review with dynamic agent allocation

### Rules (Code Quality Standards)

Project-wide standards enforced across all code changes:

- `api-config.md` - Configuration dataclass design patterns
- `code-style.md` - Coding conventions beyond pre-commit hooks
- `distributed.md` - Distributed training patterns and constraints
- `testing.md` - Testing strategy and coverage requirements
