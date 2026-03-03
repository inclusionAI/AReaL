# AI 辅助开发

AReaL  开箱即用提供了 [OpenCode](https://opencode.ai/) 配置，包含领域专家代理、引导式技能、自动化命令和插件系统 —— 所有这些都是专门为 AReaL 代码库定制的。[Claude Code](https://docs.anthropic.com/en/docs/claude-code) 也通过 `.claude/` 中的兼容配置集得到支持。

## 入门

### 安装 OpenCode 和 oh-my-opencode

```bash
curl -fsSL https://opencode.ai/install | bash
```

其他安装方式：`brew install anomalyco/tap/opencode`、
`npm install -g opencode-ai`，或者从 [发布页面](https://github.com/anomalyco/opencode/releases) 获取二进制文件。详见[官方安装指南](https://opencode.ai/docs/)。

我们推荐安装 [oh-my-opencode](https://github.com/code-yeongyu/oh-my-opencode) —— 这是一个功能完备的 OpenCode 插件，提供多代理编排、并行后台代理、LSP/AST 感知工具以及精选的 MCP 集成。

```bash
bunx oh-my-opencode install   # 交互式安装程序（推荐）
npx oh-my-opencode install    # 替代方案
```

核心功能：

- **代理编排** —— Sisyphus（编排器）、Oracle（只读顾问）、Librarian（外部参考搜索）、Explore（代码库搜索）等
- **后台代理** —— 并行启动多个代理用于研究和探索
- **精心设计的工具** —— LSP 诊断、AST-grep 搜索/替换、会话管理
- **Claude Code 兼容性** —— 完整兼容层，使 `.claude/` 配置可以在 OpenCode 中无缝使用

### 启动会话

```bash
cd /path/to/AReaL
opencode
```

OpenCode 从 `AGENTS.md` 加载项目上下文，并从 `.opencode/` 目录发现所有代理、命令、技能和插件。它还会读取 `.claude/skills/` 获取共享的技能定义。

## 开发工作流

典型的开发会话遵循以下流程：

1. **描述你的任务** —— OpenCode 读取 `AGENTS.md` 并理解 AReaL 的架构、约定和约束
2. **使用技能** —— 加载引导式工作流，如 `add-dataset` 或 `add-reward`，用于创建新组件
3. **咨询专家** —— 当任务涉及引擎、算法或基础设施时，自动启动领域专家子代理
4. **提交** —— `commit-conventions` 技能自动触发，确保提交信息一致
5. **提交 PR** —— 运行 `/create-pr` 来变基、压缩提交并创建带有自动生成描述的 PR

示例会话：

```
> 我想为代码执行添加一个新的奖励函数

OpenCode: [加载 add-reward 技能，指导实现]

> 你能审查我的更改吗？

OpenCode: [为相关领域启动专家子代理]

> /create-pr

OpenCode: [变基、压缩提交、创建 PR]
```

## 领域专家代理

专家子代理在特定领域提供深度、只读的知识。它们在 `.opencode/agents/` 中定义，并通过 `task()` 委托调用：

| 专家              | 专业领域                              | 调用方式                                    |
| ---------------- | ------------------------------------ | ------------------------------------------ |
| `fsdp-expert`    | FSDP 配置、内存优化                    | `task(subagent_type="fsdp-expert", ...)`   |
| `archon-expert`  | MoE 训练、专家并行                      | `task(subagent_type="archon-expert", ...)` |
| `megatron-expert`| 流水线并行、大模型                      | `task(subagent_type="megatron-expert", ...)`|
| `algorithm-expert`| GRPO、PPO、DAPO 算法                  | `task(subagent_type="algorithm-expert", ...)`|
| `launcher-expert` | Slurm、Ray、Kubernetes 配置           | `task(subagent_type="launcher-expert", ...)`|

当你问诸如"如何配置专家并行？"这样的问题时，OpenCode 会自动路由到相应的专家。

## 命令

命令是以 `/` 前缀调用的自动化工作流：

| 命令        | 用途                                       |
| ---------- | ------------------------------------------ |
| `/create-pr`| 变基、压缩提交并创建 PR                      |
| `/review-pr`| 智能代码审查，带风险分析                      |

**`/review-pr`** 特别强大 —— 它使用动态模板分析 PR 更改、检测风险级别（CRITICAL/HIGH/MEDIUM/LOW），并生成最小化的定向子代理进行审查。

## 技能

技能是按需加载的引导式工作流，代理可以通过内置的 `skill` 工具加载。它们从多个位置发现：

- **项目**: `.opencode/skills/<name>/SKILL.md`
- **共享**: `.claude/skills/<name>/SKILL.md`（OpenCode 也会读取）
- **全局**: `~/.config/opencode/skills/<name>/SKILL.md`

### 实现技能

这些技能位于 `.opencode/skills/` 中，引导你完成创建新 AReaL 组件的过程：

| 技能                | 使用场景                                                          |
| ------------------ | ---------------------------------------------------------------- |
| `add-dataset`      | 向 `areal/dataset/` 添加新的数据集加载器                          |
| `add-workflow`     | 创建新的 RolloutWorkflow 实现                                    |
| `add-reward`       | 实现新的奖励函数                                                 |
| `add-archon-model` | 向 Archon 引擎添加新的模型架构                                   |
| `add-unit-tests`   | 为新功能或现有功能添加测试                                       |
| `debug-distributed`| 排查分布式训练问题（卡死、OOM、NCCL 错误）                         |
| `commit-conventions`| 提交信息约定（在每次提交时自动触发）                               |

每个技能都会引导你完成完整的过程：文件创建、注册、测试和要避免的常见陷阱。相同的技能也可通过 `.claude/skills/` 供 Claude Code 用户使用。

### 外部技能（skills-lock.json）

OpenCode 支持从外部来源（如 GitHub 仓库）安装技能。已安装的外部技能在项目根目录的 `skills-lock.json` 中跟踪：

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

安装或同步外部技能到本地：

```bash
npx skills experimental_install
```

将 `skills-lock.json` 提交到版本控制，这样每个人都能获得相同版本的技能。

## 插件

插件通过钩子事件（文件编辑、工具执行、会话生命周期等）扩展 OpenCode 的行为。AReaL 在 `.opencode/package.json` 中声明插件依赖，OpenCode 在启动时通过 Bun 安装它们。更多关于编写自定义插件的信息，请参阅 OpenCode [插件文档](https://opencode.ai/docs/plugins/)。

## 配置文件

```
AReaL/
|-- AGENTS.md                # 项目上下文（自动加载）
|-- skills-lock.json         # 外部技能锁定文件
+-- .opencode/
    |-- agents/              # 领域专家子代理（5 个专家）
    |-- command/              # 自动化操作（create-pr, review-pr）
    |-- data/                # 命令支持数据
    |-- skills/              # 实现技能（7 个技能）
    |-- package.json         # 插件依赖（@opencode-ai/plugin）
    +-- node_modules/        # 自动安装（gitignored）
```

## Claude Code 兼容性

AReaL 还为 [Claude Code](https://docs.anthropic.com/en/docs/claude-code) 用户提供完整的配置集：

```bash
cd /path/to/AReaL
claude
```

Claude Code 从 `CLAUDE.md` 加载项目上下文，并使用自己的代理系统。

| 概念            | OpenCode                                  | Claude Code                  |
| -------------- | ---------------------------------------- | ---------------------------- |
| 项目上下文      | `AGENTS.md`                               | `CLAUDE.md`                 |
| 模型选择        | 任务类别（`deep`、`quick`、...）           | 显式选择（Opus、Sonnet、...）|
| 代理调度        | `task(subagent_type="...", ...)` 委托      | 自动路由                     |
| 通用代理        | 内置编排                                   | `planner`、`code-verifier`   |
| 专家名称        | `fsdp-expert`、`archon-expert`、...       | `fsdp-engine-expert`、...   |
| 提交约定        | `commit-conventions` 技能（自动触发）       | `/gen-commit-msg` 命令      |

Claude Code 还有 OpenCode 中没有的额外通用代理：

| 代理                  | 用途                                                        |
| -------------------- | ------------------------------------------------------------|
| `planner`            | 在复杂多文件更改之前创建实现计划                              |
| `code-verifier`      | 在代码更改后运行 pre-commit 钩子和测试                        |
| `simple-code-reviewer`| 在提交前执行快速代码质量检查                                 |

Claude Code 配置位于：

```
AReaL/
|-- CLAUDE.md              # 项目上下文和约束
+-- .claude/
    |-- agents/            # 专业 AI 助手（8 个代理）
    |-- skills/           # 引导式工作流（与 OpenCode 共享）
    |-- commands/         # 自动化操作（create-pr、gen-commit-msg、review-pr）
    |-- hooks/            | 预/后操作钩子
    +-- rules/            # 代码质量标准
```

## 贡献

我们欢迎对代码库和 AI 开发配置的贡献：

- **代码贡献**：新功能、bug 修复、文档改进
- **AI 配置贡献**：新技能、代理、命令或对现有配置的改进
- **OpenCode 配置**：编辑 `.opencode/` 中的文件
- **Claude Code 配置**：编辑 `.claude/` 中的文件

参见 [CONTRIBUTING.md](https://github.com/inclusionAI/AReaL/blob/main/CONTRIBUTING.md) 获取指南。
