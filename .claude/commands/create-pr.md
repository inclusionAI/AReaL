---
name: create-pr
description: Rebase from the latest `origin/main`, squash the commits from it, and then create a PR on github with intelligent commit messages based on staged changes. Invoke with /create-pr.
---

# Create Pull Request

Rebase from the latest `origin/main`, squash commits, and create a PR on GitHub with an
intelligent title and description.

## Usage

```
/create-pr [--draft] [--base <branch>]
```

**Arguments:**

- `--draft`: Create as draft PR
- `--base <branch>`: Target branch (default: `main`)

## Workflow

### Step 1: Verify Prerequisites

```bash
# Check current branch
git branch --show-current

# Check if on main/master (should NOT be)
if [[ $(git branch --show-current) == "main" || $(git branch --show-current) == "master" ]]; then
  echo "ERROR: Cannot create PR from main/master branch"
  exit 1
fi

# Check for uncommitted changes
git status --short

# Ensure gh CLI is available
gh --version
```

**Action:** If there are uncommitted changes, stop, and then ask user to commit or stash
them first.

### Step 2: Check for Existing PR

```bash
# Check if PR already exists for current branch
gh pr view --json number,title,url 2>/dev/null || echo "No existing PR"
```

**Handle Existing PR:**

- If PR exists, inform user and ask permission to force-update it
- Warn that this will rewrite the commit history and PR description
- If user declines, abort the process

### Step 3: Fetch and Rebase

```bash
# Fetch latest from origin
git fetch origin main

# Check divergence
git log --oneline HEAD ^origin/main

# Non-interactive rebase onto origin/main
git rebase origin/main
```

**Handle Conflicts:** If rebase fails due to conflicts, abort and let user handle rebase
manually:

```bash
# On rebase failure, abort automatically
git rebase --abort

# Inform user to resolve conflicts manually
echo "Rebase failed due to conflicts. Please resolve manually and retry /create-pr"
exit 1
```

### Step 4: Squash Commits into Single Commit

After successful rebase, squash all commits since `origin/main` into a single commit:

```bash
# Count commits to squash
git rev-list --count origin/main..HEAD

# Soft reset to origin/main (keeps changes staged)
git reset --soft origin/main

# Generate commit message using /gen-commit-msg logic
# See .claude/commands/gen-commit-msg.md for message generation rules
```

**Generate Commit Message** (following `/gen-commit-msg` format):

1. Analyze staged changes:

   ```bash
   git diff --cached --name-only
   git diff --cached
   ```

1. Categorize changes (feat/fix/docs/refactor/test/chore/perf)

1. Determine scope from changed files (workflow/engine/reward/dataset/api/docs/etc.)

1. Generate message in format:

   ```
   <type>(<scope>): <subject>

   <body>

   [Optional sections:]
   Key changes:
   - change 1
   - change 2

   Refs: #123, #456
   ```

1. Commit with generated message:

   ```bash
   git commit -m "$(cat <<'EOF'
   <generated commit message>
   EOF
   )"
   ```

### Step 5: Analyze Combined Changes

After squashing into a single commit:

```bash
# Get all changes since origin/main
git diff origin/main...HEAD --name-only

# Get full diff content
git diff origin/main...HEAD

# Check commit history
git log --oneline origin/main..HEAD
```

**Categorize Changes:**

Follow same categorization as `/gen-commit-msg`:

| Type       | When to Use                     |
| ---------- | ------------------------------- |
| `feat`     | New feature or capability       |
| `fix`      | Bug fix                         |
| `docs`     | Documentation only              |
| `refactor` | Code change without feature/fix |
| `test`     | Adding or fixing tests          |
| `chore`    | Build, deps, config changes     |
| `perf`     | Performance improvement         |

**Determine Scope:**

Infer from changed files:

- `areal/workflow/` → `workflow`
- `areal/engine/` → `engine`
- `areal/reward/` → `reward`
- `areal/dataset/` → `dataset`
- `areal/api/` → `api`
- `areal/utils/` → `utils`
- `areal/infra/` → `infra`
- `docs/` → `docs`
- `examples/` → `examples`
- Multiple areas → omit scope or use broader term

### Step 6: Generate PR Title and Description

**PR Title Format:**

```
<type>(<scope>): <brief description>
```

**Rules:**

- Keep under 70 characters
- Use imperative mood
- No period at end
- Mirror commit message style

**PR Description Format:**

MUST strictly follow the [GitHub PR template](../../.github/PULL_REQUEST_TEMPLATE.md):

```markdown
## Description

<!-- Clear and concise description of what this PR does -->

## Related Issue

<!-- Link to the issue this PR addresses -->
Fixes #(issue)

## Type of Change

<!-- Mark the relevant option with an 'x' -->

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Code refactoring (no functional changes)
- [ ] Performance improvement
- [ ] Test coverage improvement

## Checklist

<!-- Mark with 'x' what you've done -->

- [ ] I have read the [Contributing Guide](../CONTRIBUTING.md)
- [ ] I have run formatting tools (pre-commit or manual)
- [ ] I have run relevant unit tests and they pass
- [ ] I have added tests for new functionality
- [ ] I have updated documentation if needed
- [ ] My branch is up to date with main
- [ ] This PR introduces breaking changes (if yes, fill out details below)
- [ ] If this PR changes documentation, I have built and previewed it locally with `jb build docs`
- [ ] No critical issues raised by AI reviewers (`/gemini review`)

**Breaking Change Details (if applicable):**

<!-- Describe what breaks and how users should migrate -->

## Additional Context

<!-- Add any other context, screenshots, logs, or explanations here -->
```

**How to Fill the Template:**

1. **Description**: 2-4 sentences explaining what this PR does and why
1. **Related Issue**: Link to issue (search for related issues if exists)
1. **Type of Change**: Mark ONE primary type with `[x]`
1. **Checklist**: Mark completed items with `[x]`, leave uncompleted as `[ ]`
1. **Breaking Change Details**: Only if breaking changes checkbox is marked
1. **Additional Context**: Any extra info, related PRs, performance numbers, etc.

### Step 7: Push and Create/Update PR

Show preview to user:

```
─────────────────────────────────────────────────
Branch: feat/vision-rlvr → main

PR Title:
feat(workflow): add vision support to RLVR

PR Description:
## Description

Add VisionRLVRWorkflow for vision-language RL training. Supports image inputs
alongside text prompts and integrates with existing RLVR pipeline.

## Related Issue

Fixes #789

## Type of Change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [x] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Code refactoring (no functional changes)
- [ ] Performance improvement
- [ ] Test coverage improvement

## Checklist

- [x] I have read the [Contributing Guide](../CONTRIBUTING.md)
- [x] I have run formatting tools (pre-commit or manual)
- [ ] I have run relevant unit tests and they pass
- [x] I have added tests for new functionality
- [x] I have updated documentation if needed
- [x] My branch is up to date with main
- [ ] This PR introduces breaking changes (if yes, fill out details below)
- [x] If this PR changes documentation, I have built and previewed it locally with `jb build docs`
- [ ] No critical issues raised by AI reviewers (`/gemini review`)

**Breaking Change Details (if applicable):**

N/A

## Additional Context

Requires Pillow>=10.0.0 for image processing.

Files changed:
- `areal/workflow/vision_rlvr.py`: New VisionRLVRWorkflow class
- `areal/api/workflow_api.py:45`: Add vision config fields
- `examples/vision_rlvr.py`: Example training script
- `docs/workflows/vision.md`: Documentation

─────────────────────────────────────────────────

Commands to execute:
1. git push -u origin feat/vision-rlvr
2. gh pr create --title "..." --body "..." [--draft]
─────────────────────────────────────────────────
```

**Confirm with user**, then execute:

```bash
# Force push branch to remote (required after squash)
git push -f -u origin $(git branch --show-current)

# Create or edit PR using gh CLI with GitHub template format
# If PR exists, use 'gh pr edit' instead of 'gh pr create'
if gh pr view &>/dev/null; then
  # Update existing PR
  gh pr edit \
    --title "feat(workflow): add vision support to RLVR" \
    --body "$(cat <<'EOF'
[PR description here]
EOF
)"
else
  # Create new PR
  gh pr create \
    --base main \
    --title "feat(workflow): add vision support to RLVR" \
    --body "$(cat <<'EOF'
## Description

Add VisionRLVRWorkflow for vision-language RL training. Supports image inputs
alongside text prompts and integrates with existing RLVR pipeline.

## Related Issue

Fixes #789

## Type of Change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [x] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Code refactoring (no functional changes)
- [ ] Performance improvement
- [ ] Test coverage improvement

## Checklist

- [x] I have read the [Contributing Guide](../CONTRIBUTING.md)
- [x] I have run formatting tools (pre-commit or manual)
- [ ] I have run relevant unit tests and they pass
- [x] I have added tests for new functionality
- [x] I have updated documentation if needed
- [x] My branch is up to date with main
- [ ] This PR introduces breaking changes (if yes, fill out details below)
- [x] If this PR changes documentation, I have built and previewed it locally with `jb build docs`
- [ ] No critical issues raised by AI reviewers (`/gemini review`)

**Breaking Change Details (if applicable):**

N/A

## Additional Context

Requires Pillow>=10.0.0 for image processing.

Files changed:
- `areal/workflow/vision_rlvr.py`: New VisionRLVRWorkflow class
- `areal/api/workflow_api.py:45`: Add vision config fields
- `examples/vision_rlvr.py`: Example training script
- `docs/workflows/vision.md`: Documentation
EOF
)"
fi
```

Add `--draft` flag if requested.

**Capture PR URL** and display to user:

```
✓ PR created/updated successfully!
https://github.com/inclusionAI/AReaL/pull/123
```

## Examples

### Example 1: Feature PR

**Changes:** New dataset loader for MATH dataset

**PR Title:**

```
feat(dataset): add MATH dataset loader
```

**PR Description:**

```markdown
## Description

Add MATHDataset loader for mathematics problem solving with LaTeX rendering and
symbolic math parsing. Includes reward function for automatic answer verification
and full test coverage.

## Related Issue

Fixes #456

## Type of Change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [x] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Code refactoring (no functional changes)
- [ ] Performance improvement
- [ ] Test coverage improvement

## Checklist

- [x] I have read the [Contributing Guide](../CONTRIBUTING.md)
- [x] I have run formatting tools (pre-commit or manual)
- [x] I have run relevant unit tests and they pass
- [x] I have added tests for new functionality
- [x] I have updated documentation if needed
- [x] My branch is up to date with main
- [ ] This PR introduces breaking changes (if yes, fill out details below)
- [x] If this PR changes documentation, I have built and previewed it locally with `jb build docs`
- [ ] No critical issues raised by AI reviewers (`/gemini review`)

**Breaking Change Details (if applicable):**

N/A

## Additional Context

Dataset requires ~500MB download on first use. Added comprehensive test suite
covering all 12,500 problems with >95% reward function accuracy.

Files changed:
- `areal/dataset/math.py`: New MATHDataset class
- `areal/reward/math_reward.py`: Symbolic math reward function
- `examples/math_training.py`: Training script
- `docs/datasets/math.md`: Dataset documentation
- `tests/test_math_dataset.py`: Unit tests
```

### Example 2: Bug Fix PR

**Changes:** Fix memory leak in ArchonEngine

**PR Title:**

```
fix(engine): resolve memory leak in ArchonEngine rollout
```

**PR Description:**

```markdown
## Description

Fix memory leak during ArchonEngine rollout phase by clearing cached activations
after each batch and moving tensors to CPU before deletion. Reduces memory usage
by ~2GB per rollout iteration.

## Related Issue

Fixes #872

## Type of Change

- [x] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Code refactoring (no functional changes)
- [ ] Performance improvement
- [ ] Test coverage improvement

## Checklist

- [x] I have read the [Contributing Guide](../CONTRIBUTING.md)
- [x] I have run formatting tools (pre-commit or manual)
- [x] I have run relevant unit tests and they pass
- [x] I have added tests for new functionality
- [ ] I have updated documentation if needed
- [x] My branch is up to date with main
- [ ] This PR introduces breaking changes (if yes, fill out details below)
- [ ] If this PR changes documentation, I have built and previewed it locally with `jb build docs`
- [ ] No critical issues raised by AI reviewers (`/gemini review`)

**Breaking Change Details (if applicable):**

N/A

## Additional Context

Tested with 100 rollout iterations without OOM. Memory usage stable at 8GB
(previously would grow to 10GB+). Output correctness validated unchanged.

Backported to v0.5.x branch.

Files changed:
- `areal/engine/archon.py:234`: Add explicit cache clearing
- `areal/engine/archon.py:456`: Move tensor to CPU before deletion
- `tests/test_archon_memory.py`: Add memory leak regression test
```

### Example 3: Breaking Change PR

**Changes:** Refactor reward API for better extensibility

**PR Title:**

```
refactor(reward): simplify reward function interface
```

**PR Description:**

```markdown
## Description

Simplify reward function API from 4 methods to 2 by consolidating compute and
compute_batch into a single batched interface. Improves type hints and
documentation. All existing reward functions updated.

## Related Issue

Fixes #901

## Type of Change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [x] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Code refactoring (no functional changes)
- [ ] Performance improvement
- [ ] Test coverage improvement

## Checklist

- [x] I have read the [Contributing Guide](../CONTRIBUTING.md)
- [x] I have run formatting tools (pre-commit or manual)
- [x] I have run relevant unit tests and they pass
- [ ] I have added tests for new functionality
- [x] I have updated documentation if needed
- [x] My branch is up to date with main
- [x] This PR introduces breaking changes (if yes, fill out details below)
- [x] If this PR changes documentation, I have built and previewed it locally with `jb build docs`
- [ ] No critical issues raised by AI reviewers (`/gemini review`)

**Breaking Change Details (if applicable):**

Old `compute_batch` method is deprecated and will be removed in v0.7.0.

See migration guide in `docs/migration/reward_api.md` for details.

## Additional Context

All existing tests pass. Performance unchanged at 10k rewards/sec. Backward
compatibility warnings added for deprecated methods.

Files changed:
- `areal/api/reward_api.py:12`: Consolidate compute/compute_batch
- `areal/reward/gsm8k.py`: Update to new API
- `areal/reward/code_reward.py`: Update to new API
- `areal/reward/geometry3k.py`: Update to new API
- `docs/customization/reward.md`: Update documentation
- `docs/migration/reward_api.md`: Migration guide
- `examples/custom_reward.py`: Update example
```

## Error Handling

### Rebase Conflicts

If rebase fails:

1. Show conflict files
1. Provide resolution instructions
1. Wait for user to resolve
1. After resolution, continue with squashing step
1. Offer to abort rebase if needed: `git rebase --abort`

### Squash Failures

If squash/commit fails:

1. Check if there are changes to commit: `git status`
1. Verify no conflicts remain: `git diff --cached`
1. If needed, abort and return to pre-rebase state

### Push Failures

If force push fails:

1. Verify remote branch exists
1. Check GitHub authentication: `gh auth status`
1. Confirm branch protection rules allow force push
1. Provide manual push instructions if needed

### PR Creation/Update Failures

If `gh pr create` or `gh pr edit` fails:

1. Check if PR already exists: `gh pr view`
1. Verify GitHub authentication: `gh auth status`
1. Check for branch protection rules
1. Provide manual PR creation/update link

## Safety Checks

**Before Starting:**

- Confirm no uncommitted changes
- Confirm not on main/master branch
- Check for existing PR and get user permission to overwrite if exists
- Backup branch: `git branch backup/$(git branch --show-current)-$(date +%s)`

**Before Rebase:**

- Fetch latest from origin
- Show divergence summary

**Before Squash:**

- Show commits that will be squashed
- Confirm user wants to proceed

**Before Force Push:**

- **CRITICAL**: Warn user that force push will rewrite history
- Show current commit that will replace remote history
- Confirm branch name
- If PR exists, emphasize that PR history will be rewritten

**Before PR Creation/Update:**

- Show full preview of title/description
- Confirm target branch
- If updating existing PR, show what will change

______________________________________________________________________

<!--
================================================================================
                            MAINTAINER GUIDE
================================================================================

Location: .claude/commands/create-pr.md
Invocation: /create-pr

## Design Philosophy

- Automates full PR creation workflow: fetch, rebase, **squash to single commit**, push, create/update PR
- **Always squashes all commits** since `origin/main` into a single commit with message generated via `/gen-commit-msg` logic
- **Handles existing PRs** by detecting them and force-updating after user permission
- Follows repository's Conventional Commits format
- Requires user confirmation at critical steps (existing PR detection, rebase, squash, force-push, PR creation/update)
- Generates intelligent commit messages, PR titles, and descriptions based on change analysis
- Uses force push (`-f`) by design, as squashing requires rewriting history

## How to Update

### Adding New Scopes
Update "Determine Scope" section with new file path mappings.

### Changing PR Template
Update "PR Description Format" section with new template structure.

### Modifying Workflow Steps
Update relevant "Step N" sections with new git commands or logic.

================================================================================
-->
