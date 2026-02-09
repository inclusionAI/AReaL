# AI-Assisted Development

AReaL provides AI-assisted development configurations to help developers work more
efficiently. The current configs are written for
[Claude Code](https://docs.anthropic.com/en/docs/claude-code), but you can use other LLM
providers that support similar configuration formats.

## Getting Started

Start Claude Code in the AReaL directory:

```bash
cd /path/to/AReaL
claude
```

The AI assistant automatically loads project context from `CLAUDE.md` and understands
AReaL's architecture, conventions, and constraints.

## Development Workflow

A typical development session in AReaL follows this flow:

1. **Plan**: Describe your task. The `planner` agent creates an implementation plan
1. **Implement**: Follow the plan, using skills like `/add-dataset` if creating new
   components
1. **Verify**: The `code-verifier` agent runs pre-commit and tests automatically
1. **Review**: Use `simple-code-reviewer` or manually review your changes
1. **Submit**: Run `/create-pr` to rebase, squash, and create a PR with auto-generated
   description

Example session:

```
> I want to add a new reward function for code execution

Claude: [planner agent activates, creates implementation plan]

> /add-reward code_execution

Claude: [guides through creating areal/reward/code_execution.py]

> Can you verify my changes?

Claude: [code-verifier runs pre-commit and tests]

> /create-pr

Claude: [rebases, squashes commits, creates PR]
```

## Agents

Agents are specialized AI assistants that activate automatically based on context. They
provide domain expertise without requiring explicit invocation.

**General agents** help with common development tasks:

| Agent                  | Purpose                                                        |
| ---------------------- | -------------------------------------------------------------- |
| `planner`              | Creates implementation plans before complex multi-file changes |
| `code-verifier`        | Runs pre-commit hooks and tests after code changes             |
| `simple-code-reviewer` | Performs quick code quality checks before commits              |

**Domain expert agents** provide deep knowledge in specific areas:

| Agent                       | Expertise                                |
| --------------------------- | ---------------------------------------- |
| `fsdp-engine-expert`        | FSDP2 configuration, memory optimization |
| `archon-engine-expert`      | MoE training, expert parallelism         |
| `megatron-engine-expert`    | Pipeline parallelism, large models       |
| `algorithm-expert`          | GRPO, PPO, DAPO algorithms               |
| `launcher-scheduler-expert` | Slurm, Ray, Kubernetes configuration     |

When you ask a question like "How do I configure expert parallelism?", the AI
automatically routes to the appropriate expert agent.

## Commands

Commands are automated workflows invoked with `/` prefix. They handle multi-step
operations that would otherwise require manual execution.

| Command           | Purpose                                     |
| ----------------- | ------------------------------------------- |
| `/create-pr`      | Rebase, squash commits, and create PR       |
| `/gen-commit-msg` | Generate commit message from staged changes |
| `/pr-review`      | Intelligent code review with risk analysis  |

**`/pr-review`** is particularly powerful. It uses dynamic templates to analyze PR
changes, detect risk levels (CRITICAL/HIGH/MEDIUM/LOW), and spawn minimal targeted
subagents for review. This approach keeps reviews focused and efficient - only spawning
the specific expertise needed for each change type.

## Skills

Skills provide step-by-step guided workflows for creating new components. They ensure
you follow AReaL's conventions and don't miss required steps.

| Skill                | When to Use                                                          |
| -------------------- | -------------------------------------------------------------------- |
| `/add-dataset`       | Adding a new dataset loader to `areal/dataset/`                      |
| `/add-workflow`      | Creating a new RolloutWorkflow implementation                        |
| `/add-reward`        | Implementing a new reward function                                   |
| `/add-archon-model`  | Adding a new model architecture to the Archon engine                 |
| `/add-unit-tests`    | Adding tests for new or existing functionality                       |
| `/debug-distributed` | Troubleshooting distributed training issues (hang, OOM, NCCL errors) |

Each skill walks through the complete process: file creation, registration, testing, and
common pitfalls to avoid.

## Configuration Files

The AI-assisted development system is configured in:

```
AReaL/
├── CLAUDE.md              # Project context and constraints
└── .claude/
    ├── agents/            # Specialized AI assistants
    ├── skills/            # Guided workflows
    ├── commands/          # Automated actions
    └── rules/             # Code quality standards
```

See these files directly for configuration details.

## Contributing

We welcome contributions to both the codebase and AI development configurations:

- **Code contributions**: New features, bug fixes, documentation improvements
- **AI config contributions**: New skills, agents, commands, or improvements to existing
  ones

See [CONTRIBUTING.md](https://github.com/inclusionAI/AReaL/blob/main/CONTRIBUTING.md)
for guidelines.
