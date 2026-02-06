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
   # Build docs locally (jupyter-book is included in dev dependencies):
   jb build docs
   ```

1. **Submit a Pull Request**

We suggest applying our provided claude command `/create-pr` whenever possible.

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

## Tips for Using AI-Assisted Coding

See the full
[AI-Assisted Development Guide](https://inclusionai.github.io/AReaL/reference/ai_assisted_dev.html)
for detailed documentation.

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
from areal.infra.platforms import current_platform

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

The image building workflow can be triggered manually from any branch by users with
write permissions to the repository.

**Triggering the Workflow:**

You can trigger the workflow from any branch using either method:

1. **Via GitHub UI:**

   - Go to **Actions** → **"Build and Test Docker Image"**
   - Click **"Run workflow"** dropdown
   - Select the branch you want to build from
   - Click **"Run workflow"**

1. **Via GitHub CLI:**

   ```bash
   # Build from main
   gh workflow run build-docker-image.yml --ref main

   # Build from a feature branch
   gh workflow run build-docker-image.yml --ref feature/my-changes

   # Build from current branch
   gh workflow run build-docker-image.yml --ref $(git branch --show-current)
   ```

**Pipeline Stages:**

The workflow executes the following stages sequentially:

1. **Build**: Builds the Docker image and pushes it with `:test` tag
1. **Test**: Automatically runs the full test suite using the `:test` image
1. **Promote**: If tests pass, promotes the image by retagging `:test` → `:dev`
1. **Cleanup**: Always deletes the `:test` image from the registry (success or failure)

Building the image from scratch takes approximately 1-2 hours, plus additional time for
running the test suite.

**Normal PR Testing:**

The PR-based test workflow (triggered by the `safe-to-test` label) remains unchanged and
uses the `:dev` image. This allows testing PRs against the last known-good image.

______________________________________________________________________

Thank you for contributing to AReaL! 🙏
