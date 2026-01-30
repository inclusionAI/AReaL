# Code Style Rules

Rules beyond pre-commit (ruff format/lint).

## Design Patterns

- **Prefer composition over inheritance**: Avoid deep class hierarchies
  - Good: `Engine` holds a `Checkpointer` instance
  - Avoid: `CheckpointableEngine(Engine)` →
    `FSDPCheckpointableEngine(CheckpointableEngine)`
- Keep inheritance shallow (≤2 levels when possible)
- Use mixins sparingly; prefer explicit delegation

## Logging

- Use `areal.utils.logging.getLogger(name)` with PascalCase descriptive name, NOT
  `print` or stdlib `logging`
  - Good: `getLogger("RLVRWorkflow")`, `getLogger("ArchonEngine")`,
    `getLogger("GSM8KReward")`
  - Avoid: `getLogger(__name__)` or dotted paths like `getLogger("areal.engine.fsdp")`
- Log levels:
  - DEBUG: Detailed tracing (avoid in hot paths)
  - INFO: Milestones (training start, checkpoint saved)
  - WARNING: Recoverable issues
  - ERROR: Failures requiring attention

## Performance Patterns

- **Avoid GPU-CPU sync**: `.item()`, `.tolist()`, `print(tensor)` cause sync
- **Prefer batch operations**: Avoid Python loops over tensor elements
- **In-place ops**: Use when safe, but careful with autograd (`.add_()` vs `+`)

## Naming Conventions

| Type             | Pattern       | Example                             |
| ---------------- | ------------- | ----------------------------------- |
| Config dataclass | `XxxConfig`   | `GRPOConfig`, `FSDPConfig`          |
| Engine class     | `XxxEngine`   | `FSDPEngine`, `ArchonEngine`        |
| Workflow class   | `XxxWorkflow` | `RLVRWorkflow`, `MultiTurnWorkflow` |
| Reward function  | `xxx_reward`  | `math_reward`, `code_reward`        |

## Tensor Conventions

- Shape convention: `[batch, seq_len, hidden]` or document clearly
- Use `torch.Size` assertions for shape validation in debug
- Prefer explicit dtype/device over implicit conversion

## Import Style

- Group: stdlib, third-party, areal (ruff handles order)
- Avoid `from x import *` (CLAUDE.md rule)
- Prefer explicit imports over module-level imports for large modules
