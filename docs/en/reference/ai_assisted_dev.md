# AI-Assisted Development

AReaL ships with [OpenCode](https://opencode.ai/) configurations out of the box, giving
you domain expert agents, guided skills, automated commands, and a plugin system -- all
tailored to the AReaL codebase.
[Claude Code](https://docs.anthropic.com/en/docs/claude-code) is also supported through
a compatible configuration set in `.claude/`.

## Getting Started

### Install OpenCode and oh-my-opencode

```bash
curl -fsSL https://opencode.ai/install | bash
```

Alternative methods: `brew install anomalyco/tap/opencode`,
`npm install -g opencode-ai`, or grab a binary from the
[releases page](https://github.com/anomalyco/opencode/releases). See the
[official install guide](https://opencode.ai/docs/) for details.

We recommend installing [oh-my-opencode](https://github.com/code-yeongyu/oh-my-opencode)
-- a batteries-included OpenCode plugin that provides multi-agent orchestration,
parallel background agents, LSP/AST-aware tools, and curated MCP integrations.

```bash
bunx oh-my-opencode install   # interactive installer (recommended)
npx oh-my-opencode install    # alternative
```

Key capabilities:

- **Agent orchestration** -- Sisyphus (orchestrator), Oracle (read-only consultant),
  Librarian (external reference search), Explore (codebase grep), and more
- **Background agents** -- Fire multiple agents in parallel for research and exploration
- **Crafted tools** -- LSP diagnostics, AST-grep search/replace, session management
- **Claude Code compatibility** -- Full compatibility layer so `.claude/` configs work
  seamlessly in OpenCode

### Start a session

```bash
cd /path/to/AReaL
opencode
```

OpenCode loads project context from `AGENTS.md` and discovers all agents, commands,
skills, and plugins from the `.opencode/` directory. It also reads `.claude/skills/` for
shared skill definitions.

## Development Workflow

A typical development session follows this flow:

1. **Describe your task** -- OpenCode reads `AGENTS.md` and understands AReaL's
   architecture, conventions, and constraints
1. **Use skills** -- Load guided workflows like `add-dataset` or `add-reward` for
   creating new components
1. **Consult experts** -- Domain expert subagents are fired automatically when the task
   touches engines, algorithms, or infrastructure
1. **Commit** -- The `commit-conventions` skill auto-triggers to ensure consistent
   commit messages
1. **Submit** -- Run `/create-pr` to rebase, squash, and create a PR with auto-generated
   description

Example session:

```
> I want to add a new reward function for code execution

OpenCode: [loads add-reward skill, guides implementation]

> Can you review my changes?

OpenCode: [fires expert subagents for relevant domains]

> /create-pr

OpenCode: [rebases, squashes commits, creates PR]
```

## Domain Expert Agents

Expert subagents provide deep, read-only knowledge in specific areas. They are defined
in `.opencode/agents/` and fired via `task()` delegation:

| Expert             | Expertise                                | Invocation                                    |
| ------------------ | ---------------------------------------- | --------------------------------------------- |
| `fsdp-expert`      | FSDP2 configuration, memory optimization | `task(subagent_type="fsdp-expert", ...)`      |
| `archon-expert`    | MoE training, expert parallelism         | `task(subagent_type="archon-expert", ...)`    |
| `megatron-expert`  | Pipeline parallelism, large models       | `task(subagent_type="megatron-expert", ...)`  |
| `algorithm-expert` | GRPO, PPO, DAPO algorithms               | `task(subagent_type="algorithm-expert", ...)` |
| `launcher-expert`  | Slurm, Ray, Kubernetes configuration     | `task(subagent_type="launcher-expert", ...)`  |

When you ask a question like "How do I configure expert parallelism?", OpenCode
automatically routes to the appropriate expert.

## Commands

Commands are automated workflows invoked with the `/` prefix:

| Command             | Purpose                                    |
| ------------------- | ------------------------------------------ |
| `/create-pr`        | Rebase, squash commits, and create PR      |
| `/review-pr`        | Intelligent code review with risk analysis |
| `/translate-doc-zh` | Translate English documentation to Chinese |

**`/review-pr`** is particularly powerful -- it uses dynamic templates to analyze PR
changes, detect risk levels (CRITICAL/HIGH/MEDIUM/LOW), and spawn minimal targeted
subagents for review.

## Skills

Skills are on-demand guided workflows that the agent can load via the built-in `skill`
tool. They are discovered from multiple locations:

- **Project**: `.opencode/skills/<name>/SKILL.md`
- **Shared**: `.claude/skills/<name>/SKILL.md` (also read by OpenCode)
- **Global**: `~/.config/opencode/skills/<name>/SKILL.md`

### Implementation skills

These skills live in `.opencode/skills/` and guide you through creating new AReaL
components:

| Skill                | When to Use                                                          |
| -------------------- | -------------------------------------------------------------------- |
| `add-dataset`        | Adding a new dataset loader to `areal/dataset/`                      |
| `add-workflow`       | Creating a new RolloutWorkflow implementation                        |
| `add-reward`         | Implementing a new reward function                                   |
| `add-archon-model`   | Adding a new model architecture to the Archon engine                 |
| `add-unit-tests`     | Adding tests for new or existing functionality                       |
| `debug-distributed`  | Troubleshooting distributed training issues (hang, OOM, NCCL errors) |
| `commit-conventions` | Commit message conventions (auto-triggers on every commit)           |

Each skill walks through the complete process: file creation, registration, testing, and
common pitfalls to avoid. The same skills are also available via `.claude/skills/` for
Claude Code users.

### External skills (skills-lock.json)

OpenCode supports installing skills from external sources (e.g., GitHub repositories).
Installed external skills are tracked in `skills-lock.json` at the project root:

```json
{
  "version": 1,
  "skills": {
    "agent-browser": {
      "source": "vercel-labs/agent-browser",
      "sourceType": "github",
      "computedHash": "6325a9ba..."
    },
    "find-skills": {
      "source": "vercel-labs/skills",
      "sourceType": "github",
      "computedHash": "6412eb4e..."
    }
  }
}
```

To install or sync external skills locally:

```bash
npx skills experimental_install
```

Commit `skills-lock.json` to version control so everyone gets the same skill versions.

## Plugins

Plugins extend OpenCode's behavior by hooking into events (file edits, tool execution,
session lifecycle, etc.). AReaL declares plugin dependencies in
`.opencode/package.json`, and OpenCode installs them via Bun at startup. For more on
writing custom plugins, see the OpenCode
[plugin docs](https://opencode.ai/docs/plugins/).

## Configuration Files

```
AReaL/
|-- AGENTS.md                # Project context (loaded automatically)
|-- skills-lock.json         # External skill lockfile
+-- .opencode/
    |-- agents/              # Domain expert subagents (5 experts)
    |-- command/              # Automated actions (create-pr, review-pr)
    |-- data/                # Supporting data for commands
    |-- skills/              # Implementation skills (7 skills)
    |-- package.json         # Plugin dependencies (@opencode-ai/plugin)
    +-- node_modules/        # Auto-installed (gitignored)
```

## Claude Code Compatibility

AReaL also provides a full configuration set for
[Claude Code](https://docs.anthropic.com/en/docs/claude-code) users:

```bash
cd /path/to/AReaL
claude
```

Claude Code loads project context from `CLAUDE.md` and uses its own agent system.

| Concept           | OpenCode                                    | Claude Code                  |
| ----------------- | ------------------------------------------- | ---------------------------- |
| Project context   | `AGENTS.md`                                 | `CLAUDE.md`                  |
| Model selection   | Task categories (`deep`, `quick`, ...)      | Explicit (Opus, Sonnet, ...) |
| Agent dispatch    | `task(subagent_type="...", ...)` delegation | Automatic routing            |
| General agents    | Built-in orchestration                      | `planner`, `code-verifier`   |
| Expert names      | `fsdp-expert`, `archon-expert`, ...         | `fsdp-engine-expert`, ...    |
| Commit convention | `commit-conventions` skill (auto-triggers)  | `/gen-commit-msg` command    |

Claude Code has additional general-purpose agents not present in OpenCode:

| Agent                  | Purpose                                                        |
| ---------------------- | -------------------------------------------------------------- |
| `planner`              | Creates implementation plans before complex multi-file changes |
| `code-verifier`        | Runs pre-commit hooks and tests after code changes             |
| `simple-code-reviewer` | Performs quick code quality checks before commits              |

Claude Code configuration lives in:

```
AReaL/
|-- CLAUDE.md              # Project context and constraints
+-- .claude/
    |-- agents/            # Specialized AI assistants (8 agents)
    |-- skills/            # Guided workflows (shared with OpenCode)
    |-- commands/          # Automated actions (create-pr, gen-commit-msg, review-pr)
    |-- hooks/             # Pre/post action hooks
    +-- rules/             # Code quality standards
```

## Contributing

We welcome contributions to both the codebase and AI development configurations:

- **Code contributions**: New features, bug fixes, documentation improvements
- **AI config contributions**: New skills, agents, commands, or improvements to existing
  ones
- **OpenCode configs**: Edit files in `.opencode/`
- **Claude Code configs**: Edit files in `.claude/`

See [CONTRIBUTING.md](https://github.com/inclusionAI/AReaL/blob/main/CONTRIBUTING.md)
for guidelines.
