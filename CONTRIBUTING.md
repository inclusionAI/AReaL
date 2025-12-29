# Contributing to AReaL

Thank you for your interest in contributing to AReaL! We welcome contributions from
everyone, whether you're fixing bugs, improving documentation, adding new features, or
helping with code reviews. This guide will help you get started.

## Table of Contents

- [Quick Start](#quick-start)
- [Ways to Contribute](#ways-to-contribute)
- [Tips for Using AI-Assisted Coding](#tips-for-using-ai-assisted-coding)
- [CI/CD](#cicd)

## Quick Start

1. **Fork and Clone:**

   ```bash
   # Fork the repository on GitHub, then:
   git clone https://github.com/YOUR-USERNAME/AReaL
   cd AReaL
   ```

1. **Install Development Dependencies:**

   Check our
   [installation guide](https://inclusionai.github.io/AReaL/tutorial/installation.html)
   for detailed setup instructions.

   ```bash
   # If you are using a local pip environment:
   uv pip install -e ".[all]"
   # Or use the Docker image illustrated in the installation guide
   # In both environments, run the following command:
   pip install -e . --no-deps
   ```

1. **Set Up Code Formatting:**

   ```bash
   pip install pre-commit
   pre-commit install
   # Subsequent commits will automatically format your files:
   git commit -a -m 'my change'
   ```

1. **Find an Issue:**

   - Browse
     [good first issues](https://github.com/inclusionAI/AReaL/labels/good%20first%20issue)
   - Check [help wanted](https://github.com/inclusionAI/AReaL/labels/help%20wanted)
     issues
   - Or create a new issue using our
     [issue templates](https://github.com/inclusionAI/AReaL/issues/new/choose)

1. **Make Your Changes:**

   - Create a branch: `git checkout -b your-feature-name`
   - Make your changes with proper formatting
   - Test your changes following the next step

1. **Test Your Changes:**

   ```bash
   # --sw: step-wise debugging
   # --lf: run the last failed test first
   pytest -sv --sw --lf areal/tests/
   ```

   Our test suite includes:

   - Running all examples to ensure they can execute one RL step
   - Checking individual engine functionalities, including rollout, forward-backward,
     and weight updates
   - Verifying numerical consistency of our packed data format with HuggingFace padded
     input, with and without Ulysses
   - Testing staleness management functionality
   - Ensuring GSM8K SFT loss decreases and RL rewards increase
   - Running other unit tests for individual components

   Some unit tests require multiple GPUs. The entry point scripts are located under
   `areal/tests/torchrun`. In the corresponding test files (e.g.,
   `test_data_redistribution.py`), we use subprocesses to launch distributed experiments
   with `torchrun` and wait for results.

   If you have modified documentation, build it locally and preview it before opening a
   PR:

   ```bash
   # Build docs locally:
   pip install jupyter-book
   jb build docs
   ```

1. **Submit a Pull Request**

## Ways to Contribute

### 🐛 Bug Reports

Found a bug? Please create a
[bug report](https://github.com/inclusionAI/AReaL/issues/new?template=bug.md) with:

- A clear description of the issue
- Steps to reproduce
- Expected vs. actual behavior
- Environment details (commit ID, hardware, software)
- Full logs when possible

### ✨ Feature Requests

Have an idea? Submit a
[feature request](https://github.com/inclusionAI/AReaL/issues/new?template=feature.md)
with:

- Background and use case
- Proposed solution or implementation approach
- Expected benefits to the community

### 📚 Documentation

Documentation improvements are always welcome:

- Fix typos or clarify existing docs
- Add examples or tutorials
- Improve API documentation
- Write blog posts or guides

### 💻 Code Contributions

We accept various types of code contributions:

- Bug fixes
- New features
- Performance improvements
- Algorithm implementations
- Test coverage improvements
- Code refactoring

**IMPORTANT**: For new features and code refactoring, please submit a corresponding
issue or open a draft PR to discuss with the core developers before making any code
changes. Directly opening a PR that conflicts with our future [roadmap](ROADMAP.md) may
waste your effort.

When opening a PR:

- Use the [PR template](.github/PULL_REQUEST_TEMPLATE.md) and complete the checklist
- Link to the related issue using `Fixes #123` or `Closes #456`
- Describe what changed and why (you can use GitHub Copilot summarization)
- Prefix "wip:" in the PR title or mark it as a draft if it's still work-in-progress
- List the testing you performed
- Let AI review first before requesting human reviewers

### 📝 Logging Convention

AReaL uses colored logging to help distinguish log messages from different components.
When adding new loggers, please follow these conventions:

**Logger Naming:**

- Use **PascalCase** with no spaces or hyphens (e.g., `TrainController`, not
  `train_controller` or `Train Controller`)
- Use descriptive names that indicate the component type
- Use category suffixes: `Scheduler`, `Launcher`, `Workflow`, `Controller`, `Engine`,
  etc.
- For per-rank loggers, use the format `[{Component} Rank {N}]` (e.g.,
  `[FSDPEngine Rank 0]`)

**Example:**

```python
import logging

# Good - PascalCase, descriptive
logger = logging.getLogger("TrainController")
logger = logging.getLogger("RLVRWorkflow")
logger = logging.getLogger("[FSDPEngine Rank 0]")

# Bad - avoid these patterns
logger = logging.getLogger(__name__)  # Not descriptive in logs
logger = logging.getLogger("train controller")  # Has spaces
logger = logging.getLogger("train_controller")  # Snake case
```

**Color Scheme by Category:**

| Color           | Category                                    | Examples                                 |
| --------------- | ------------------------------------------- | ---------------------------------------- |
| white           | Infrastructure (Schedulers, Launchers, RPC) | `LocalScheduler`, `SGLangWrapper`        |
| light_purple    | RL-specific (Workflows, Rewards, OpenAI)    | `RLVRWorkflow`, `GSM8KReward`            |
| blue            | Orchestration (Controllers)                 | `TrainController`, `RolloutController`   |
| light_green     | Data/Metrics (Stats, Dataset, Trainers)     | `StatsLogger`, `Dataset`, `RLTrainer`    |
| light_cyan/cyan | Compute backends (Engines, Platforms)       | `FSDPEngine`, `CUDAPlatform`, `PPOActor` |
| yellow/red      | Warning/Error levels (always override)      | Any logger at WARNING+ level             |

When adding a new logger, register it in `areal/utils/logging.py` under
`LOGGER_COLORS_EXACT` or `LOGGER_PATTERNS` to ensure consistent coloring.

Test your logger colors with: `python -m areal.utils.logging`

## Tips for Using AI-Assisted Coding

- [AGENTS.md](AGENTS.md) is a reference guide for AI coding agents working on AReaL.
  Before letting AI make any changes, ensure it understands the codebase using
  `AGENTS.md`.

- You can use the plan mode of coding agents to generate a plan for refactoring or new
  features. Submit it as a draft PR before making any actual code changes and discuss
  with the core developers.

## CI/CD

### Format Check

The format check runs automatically whenever a PR is opened. Your PR will pass the
format check as long as you have properly run the formatting tools using `pre-commit`.

**Important Note on Formatting Tools:**

We are gradually transitioning our Python formatting tool from `black` to `ruff`.
Currently, the CI format check still uses `black` for Python file formatting, while
`pre-commit` uses `ruff`. Please note that `ruff check` will fail on files in `areal/`
and `examples/` because these directories have not been fully re-formatted yet.

`black` and `ruff` have known conflicts when handling long assertions. To pass the CI
format check, you should manually convert long assertions to `if`-`raise` statements.
See [this issue](https://github.com/inclusionAI/AReaL/issues/503) for detailed
information.

### Tests

Tests for PRs are triggered when the PR is manually tagged with `safe-to-test`. The test
suite runs on ephemeral GCP compute engines with 2 A100 GPUs (40GB memory).

> **IMPORTANT:** To re-run tests, **DO NOT** click the "Re-run workflow" button on
> GitHub. Instead, remove the `safe-to-test` tag and then add it back.

**Writing Tests for New Features:**

If you have implemented a new feature, we highly recommend writing tests and adding them
to our pytest workflow. Place your test files under `areal/tests/test_*.py` and mark
them with our pre-defined pytest markers:

- `slow`: Tests that take more than 30 seconds to run. These will not run in the CI/CD
  workflow unless also marked with `ci`.
- `ci`: Tests that should run in the CI/CD workflow (only needed for `slow` tests).
- `gpu`: Tests that use a single GPU.
- `multi_gpu`: Tests that use more than one GPU.

Our CI/CD runs tests selected by `pytest -m "not slow or ci"`. Since our CI machines
only have two GPUs, please skip tests that require more than 2 GPUs to prevent CI
failures. For example:

```python
import pytest
from areal.platforms import current_platform

# ordinary tests are supposed to run fast, and will run in CI
def test_fast_operation():
    ...

# slow operations that will NOT run in CI
@pytest.mark.slow
def test_slow_operation():
    ...

# slow operations BUT must be tested in CI
@pytest.mark.slow
@pytest.mark.ci
def test_slow_operation():
    ...

# skip tests for more than 2 GPUs
@pytest.mark.skipif(current_platform.device_count() < 4, reason="This test requires 4 GPUs")
def test_some_multi_gpu_functionality():
    ...
```

### Image Building

> **NOTE:** The image building CI workflow is experimental and subject to change.

The image building CI runs on the `build-docker-image` branch. Only project members with
write permissions can push to this branch and open a PR.

**Triggering the Workflow:**

The workflow is triggered when:

1. A PR from `build-docker-image` to `main` is opened **AND**
1. The PR is tagged with `new-image`

The workflow will wake up a pinned CPU GCP compute engine instance with 64 vCPUs and 512
GB memory, run the build job with the code and Dockerfile from the current commit, and
push the image as `ghcr.io/inclusionai/areal-runtime:dev`. Building the image from
scratch takes approximately 1-2 hours.

**Testing with the New Image:**

After successfully building the image:

1. Remove the `new-image` tag
1. Add the `safe-to-test` tag to trigger CI tests using the same procedure described
   above

Note that our test suite detects the branch name that triggers the workflow. When the
branch name is `build-docker-image`, it will pull the dev image instead of the stable
image for testing.

**Important:** If you add the `safe-to-test` tag without removing `new-image` first,
both image building and testing workflows will run simultaneously, which is usually
undesired.

______________________________________________________________________

Thank you for contributing to AReaL! 🙏
