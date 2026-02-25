---
description: Intelligent PR code review with dynamic agent allocation based on change types
---

# PR Code Review (Dynamic Agent Allocation)

Intelligent code review for the current branch's Pull Request. Dynamically generates
targeted review tasks based on PR changes, using OpenCode's `task()` delegation system
for parallel multi-agent review.

**STRICTLY READ-ONLY.** You MUST NOT:

- Modify, create, or delete any workspace files
- Switch branches, checkout, or run any git write operations
  (commit/push/reset/rebase/stash/branch -D)
- Merge, close, or edit the PR on GitHub
- Run build, test, install, or any commands that modify the environment
- Use Edit, Write, or any file-mutation tools -- only Read, Grep, Glob, Bash (read-only
  commands), and task()

## Current PR Context

PR info:
!`PR_NUM=$(echo "$ARGUMENTS" | grep -oE '[0-9]+' | head -1); OUT=$(gh pr view ${PR_NUM:+$PR_NUM} --json number,title,state,isDraft,body,files 2>&1); EC=$?; if [ $EC -eq 0 ]; then echo "$OUT"; else if echo "$OUT" | grep -qiE 'no pull requests found'; then echo 'No PR found for current branch'; else echo "ERROR: Failed to fetch PR${PR_NUM:+ #$PR_NUM}: $OUT"; fi; fi`

Current branch: !`git branch --show-current`

## Data Files

The following data files contain detection tables and task templates (auto-included):

@.opencode/data/review-pr-change-types.md @.opencode/data/review-pr-templates.md

## Arguments

`$ARGUMENTS`

- No arguments: Review PR for current branch
- PR number: Review specific PR (e.g., `/review-pr 123`)
- `--quick`: Quick mode, only run Phase 1 analysis (no task delegation). Can combine:
  `/review-pr 123 --quick`

## Quick Start

1. Check PR info from the context above
1. If PR doesn't exist or is closed, stop and explain
1. Reference the data files included above
1. Execute Phases 1-4 in order

## Workflow Overview

```
Phase 1: Deep PR Analysis
    |- 1.0 PR Status Check [quick]
    |- 1.1 Get PR Summary [quick]
    +- 1.2-1.4 Change Type Detection [direct analysis]
    |
Phase 2: Dynamic Agent Planning [direct analysis]
    |
Phase 3: Execute Review Tasks [parallel task() delegation]
    |
Phase 4: Confidence Scoring & Summary [direct analysis]
```

## Delegation Strategy

OpenCode uses `task()` with categories for delegating review work.

**If `--quick` is set**: stop after Phase 1 and output `CHANGE_ANALYSIS_REPORT` only (do
NOT delegate review tasks).

Otherwise, map risk levels to categories:

| Risk Level   | Category                                                                                       |
| ------------ | ---------------------------------------------------------------------------------------------- |
| **CRITICAL** | `deep` + `ultrabrain` (dual-fire: deep for context tracing, ultrabrain for logic verification) |
| **HIGH**     | `deep`                                                                                         |
| **MEDIUM**   | `unspecified-high`                                                                             |
| **LOW**      | `quick`                                                                                        |

**Delegation pattern:**

```
// For CRITICAL risk: fire BOTH categories in parallel for the SAME checklist
task(category="deep", load_skills=[...], run_in_background=true,
  description="PR Review: <task> (deep)", prompt="...")
task(category="ultrabrain", load_skills=[...], run_in_background=true,
  description="PR Review: <task> (ultrabrain)", prompt="...")

// For HIGH/MEDIUM/LOW: single task
task(category="<category from table>", load_skills=[...], run_in_background=true,
  description="PR Review: <task>", prompt="...")
```

When merging dual-fire results, keep the **union** of all findings. If both agents flag
the same issue, use the **higher** confidence score.

**Expert subagent consultation** -- for framework-specific reviews, also fire the
relevant expert in parallel:

| Review Domain    | Expert Subagent    |
| ---------------- | ------------------ |
| Archon changes   | `archon-expert`    |
| FSDP changes     | `fsdp-expert`      |
| Megatron changes | `megatron-expert`  |
| RL algorithm     | `algorithm-expert` |
| Infra/launcher   | `launcher-expert`  |

______________________________________________________________________

## Phase 1: Deep PR Analysis

### 1.0 PR Status Check

Check if PR should be reviewed:

- Is it closed? -> Stop
- Is it a draft? -> Note but continue
- Is it bot-generated? -> Skip

### 1.1 Get PR Summary

Get basic PR info: title, description, modified files, change summary.

### 1.2 Change Type Detection

Analyze each file change, detecting change types by risk level.

**Reference**: See `.opencode/data/review-pr-change-types.md` for complete detection
tables:

- CRITICAL level types (Archon, FSDP, Megatron, DCP)
- HIGH level types (distributed comm, DTensor, MoE, TP/EP/CP)
- MEDIUM level types (tensor ops, workflow, API, compile)
- LOW level types (tests, docs, config)

### 1.3 Framework-Specific Risk Identification

Based on detected types, identify corresponding risks.

**Reference**: See `.opencode/data/review-pr-change-types.md` for risk lists per
framework.

### 1.4 Output Change Analysis Report

```
CHANGE_ANALYSIS_REPORT:
- detected_types: [ARCHON_PARALLEL, EP_ETP, FSDP_CORE, ...]
- risk_level: CRITICAL | HIGH | MEDIUM | LOW
- affected_files: [file1.py, file2.py, ...]
- identified_risks: [risk1, risk2, ...]
- related_frameworks: [archon, fsdp, megatron, ...]
```

______________________________________________________________________

## Phase 2: Dynamic Agent Planning

### 2.1 Planning Principles

1. **Generate tasks by risk area**: Each high-risk area gets a dedicated task
1. **Merge related changes**: Interdependent changes can be merged
1. **Category selection**: CRITICAL/HIGH -> `deep`, MEDIUM -> `unspecified-high`, LOW ->
   `quick`
1. **Minimum coverage**: Even simple changes get at least 1 basic review task
1. **Skill loading**: Include relevant skills for framework-specific reviews (e.g.,
   `add-archon-model` for Archon changes, `debug-distributed` for distributed code)
1. **Expert consultation**: Fire relevant expert subagents for CRITICAL framework
   changes

### 2.2 Task Template Selection

Based on detected change types, select appropriate review task templates.

**Reference**: See `.opencode/data/review-pr-templates.md` for complete task templates:

- Framework-specific tasks (Archon, FSDP, Megatron, DCP, Trainer)
- General tasks (Logic, Concurrency, Tensor, Numerical, TP, etc.)

### 2.3 Output Review Task List

```
GENERATED_REVIEW_TASKS:
1. [deep] Task Name
   - Reason: XXX change type detected
   - Skills: [skill1, skill2]  // or [] if none
   - Expert: archon-expert     // or none
   - Checklist: [...]
   - Focus files: [...]

2. [unspecified-high] Task Name
   - Reason: ...
   - Skills: []
   ...
```

______________________________________________________________________

## Phase 3: Execute Review Tasks \[Parallel\]

### 3.1 Execution Rules

- Use `task()` with the category specified in Phase 2 for each task
- Execute all tasks **in parallel** with `run_in_background=true`
- Fire expert subagents in parallel for framework-specific reviews
- Each agent reviews independently
- Collect all results with `background_output()` before proceeding to Phase 4

### 3.2 Delegation Template

For each review task from Phase 2, delegate as:

```
task(
  category="<deep|unspecified-high|quick>",
  load_skills=["<relevant-skills>"],
  run_in_background=true,
  description="PR Review: <Task Name>",
  prompt="""
    1. TASK: Review PR changes for <specific concern>
    2. EXPECTED OUTCOME: List of findings with severity, file, line, description, and suggestion
    3. REQUIRED TOOLS: Read, Grep, Glob, Bash (for gh/git commands)
    4. MUST DO:
       - Follow the checklist: <checklist from template>
       - Focus on these files: <file list>
       - Provide code snippets for each finding
       - Rate each finding: CRITICAL | HIGH | MEDIUM | LOW
    5. MUST NOT DO:
       - Do not modify any files
       - Do not check build signals or try to build/type-check
       - Do not post comments to PR
       - Do not report pre-existing issues not introduced by this PR
    6. CONTEXT:
       - PR diff: <relevant diff sections>
       - Related code: <caller/callee context as needed>

    OUTPUT FORMAT:
    REVIEW_RESULT:
    task_name: "<Task Name>"
    category: <deep|unspecified-high|quick>
    findings:
      - issue: "Issue description"
        severity: CRITICAL | HIGH | MEDIUM | LOW
        file: "path/to/file.py"
        line: 123
        code_snippet: |
          Relevant code snippet
        reason: "Why this is an issue"
        suggestion: "Fix suggestion"
  """
)
```

### 3.3 Review Depth by Category

| Category             | Requirements                                                               |
| -------------------- | -------------------------------------------------------------------------- |
| **deep**             | Complete context, cross-file traces, verify parallel strategy interactions |
| **unspecified-high** | Changed code + direct callers/callees, type signature consistency          |
| **quick**            | Format and basic correctness only                                          |

______________________________________________________________________

## Phase 4: Confidence Scoring & Summary

### 4.1 Confidence Scoring (0-100)

| Score   | Meaning                               |
| ------- | ------------------------------------- |
| **0**   | False positive or pre-existing issue  |
| **25**  | May be real, cannot verify            |
| **50**  | Real but minor or rare                |
| **75**  | Very likely real, important           |
| **100** | Confirmed real, will frequently occur |

### 4.2 Summary Report Format

```markdown
# PR Review Summary

## PR Overview
- **Title**: PR title
- **Detected Change Types**: [...]
- **Risk Level**: CRITICAL | HIGH | MEDIUM | LOW
- **Generated Review Tasks**: N

## Executed Review Tasks
| # | Category | Task Name | Reason |
|---|----------|-----------|--------|

## Findings

### CRITICAL Severity (Confidence >= 75)
#### Issue 1: [Title]
- **File**: `path/to/file.py:123`
- **Confidence**: 85
- **Description**: ...
- **Fix Suggestion**: ...

### HIGH Severity (Confidence >= 50)
...

## Review Statistics
- Total issues: X (CRITICAL: X, HIGH: X, MEDIUM: X, LOW: X)
- Filtered false positives: X
```

### 4.3 Output Integrity (CRITICAL)

The Phase 4 summary report is the **FINAL DELIVERABLE** of this command.

- Output the COMPLETE report exactly as specified in Section 4.2 -- every section, every
  finding, every field.
- Do NOT abbreviate, summarize, or compress any part of the report.
- Do NOT omit findings, code snippets, fix suggestions, or statistics.
- If the report is long, that is expected and correct -- **completeness > brevity**.
- The orchestrating agent receiving this output MUST present it **VERBATIM** to the
  user. No re-summarization, no condensing, no "brief version".

______________________________________________________________________

## Dynamic Generation Examples

| PR Type        | Detected Types                        | Generated Tasks             |
| -------------- | ------------------------------------- | --------------------------- |
| Docs only      | \[DOCS\]                              | 1 quick                     |
| Config only    | \[CONFIG_ONLY\]                       | 1-2 quick                   |
| Single bug fix | \[TENSOR_OPS\]                        | 2-4 unspecified-high        |
| Archon core    | \[ARCHON\_\*, EP_ETP, DTENSOR\]       | 4-8 deep + expert subagents |
| Cross-domain   | \[WORKFLOW_ENGINE, FSDP_CORE, TESTS\] | 5-10 mixed categories       |

______________________________________________________________________

## False Positive Guide (Rate Confidence 0)

- Pre-existing issues (not introduced by this PR)
- Intentionally designed code that looks like a bug
- Issues linter/compiler would catch
- Issues on lines user didn't modify
- Explicitly disabled issues (lint ignore comments)

______________________________________________________________________

## Important Notes

- **Do NOT** check build signals or try to build/type-check
- Use `gh` to interact with GitHub, not web fetch
- **Do NOT** automatically post comments to PR
- Must provide file path and line number when referencing issues

______________________________________________________________________

<!--
================================================================================
                            MAINTAINER GUIDE
================================================================================

Location: .opencode/command/review-pr.md
Invocation: /review-pr

Related files:
  - .opencode/data/review-pr-change-types.md: Change type detection tables
  - .opencode/data/review-pr-templates.md: Review task templates

## Differences from Claude Code version

1. Model names (Opus/Sonnet/Haiku) -> task() categories (deep/unspecified-high/quick)
2. @import syntax removed -> uses @ file references for auto-inclusion
3. allowed-tools frontmatter removed
4. Added subtask: true to run as subtask (not pollute main context)
5. Added shell output injection (!`command`) for automatic PR context
6. Added explicit task() delegation patterns
7. Added expert subagent consultation for framework-specific reviews
8. Added skill loading guidance

## How to Update

### Adding New Change Types
Edit .opencode/data/review-pr-change-types.md:
1. Add to appropriate level table (CRITICAL/HIGH/MEDIUM/LOW)
2. Add framework risks if applicable
3. Also update .claude/data/review-pr-change-types.md to keep in sync

### Adding New Task Templates
Edit .opencode/data/review-pr-templates.md:
1. Add to framework-specific or general section
2. Include checklist and category assignment
3. Also update .claude/data/review-pr-templates.md to keep in sync

### Adjusting Category Selection
Modify "Delegation Strategy" table in this file.

================================================================================
-->
