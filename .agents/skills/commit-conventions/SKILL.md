---
name: commit-conventions
description: AReaL Conventional Commit guidance with repository-specific scope inference. Load before every `git commit`.
---

# Commit Conventions

Load this skill before creating any commit in AReaL.

## Format

```text
<type>(<scope>): <subject>

<body>

[optional]
Key changes:
- change 1
- change 2

Refs: #123
```

## Type Selection

| Type       | Use for                                                  |
| ---------- | -------------------------------------------------------- |
| `feat`     | New feature or capability                                |
| `fix`      | Bug fix                                                  |
| `docs`     | Documentation-only changes                               |
| `refactor` | Structural change without feature or bug behavior change |
| `test`     | New or corrected tests                                   |
| `chore`    | Tooling, config, workflow, or dependency changes         |
| `perf`     | Performance improvements                                 |

## Scope Inference

Infer scope from the primary changed paths:

| Path                                                         | Scope      |
| ------------------------------------------------------------ | ---------- |
| `areal/workflow/`                                            | `workflow` |
| `areal/engine/`                                              | `engine`   |
| `areal/reward/`                                              | `reward`   |
| `areal/dataset/`                                             | `dataset`  |
| `areal/api/`                                                 | `api`      |
| `areal/utils/`                                               | `utils`    |
| `areal/infra/`                                               | `infra`    |
| `areal/trainer/`                                             | `trainer`  |
| `areal/models/`                                              | `models`   |
| `areal/experimental/`                                        | `archon`   |
| `docs/`                                                      | `docs`     |
| `examples/`                                                  | `examples` |
| `AGENTS.md`, `.agents/`, `.claude/`, `.codex/`, `.opencode/` | `agents`   |

If the commit spans multiple unrelated areas, omit the scope instead of inventing one.

## Rules

- Keep the subject imperative.
- Keep the subject under about 72 characters.
- Do not end the subject with a period.
- Use the body to explain why the change exists.
- Add `Key changes:` only when it materially improves readability.

## Example

```text
chore(agents): add Codex harness for AReaL

Register Codex subagents and migrate reusable workflows into
project skills so the repository has a native Codex setup.

Key changes:
- add .codex/config.toml and custom agent prompts
- add .agents/skills for repo-local Codex workflows
- align AGENTS.md and docs with Codex naming
```
