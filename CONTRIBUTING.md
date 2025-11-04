# Contributing to AReaL

Thank you for your interest in contributing to AReaL! We welcome contributions from
everyone, whether you're fixing bugs, improving documentation, adding new features, or
helping with code reviews. This guide will help you get started.

## Table of Contents

- [Contributing to AReaL](#contributing-to-areal)
  - [Table of Contents](#table-of-contents)
  - [Quick Start](#quick-start)
  - [Ways to Contribute](#ways-to-contribute)
    - [🐛 Bug Reports](#-bug-reports)
    - [✨ Feature Requests](#-feature-requests)
    - [📚 Documentation](#-documentation)
    - [💻 Code Contributions](#-code-contributions)
  - [Tips for Using AI-Assisted Coding](#tips-for-using-ai-assisted-coding)

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
   bash examples/env/setup-pip-deps.sh
   # Or use the Docker image illustrated in the installation guide
   # In both environments, run the following command:
   pip install -e ".[dev]"
   ```

   **Note on package structure:** We separate packages like `flash-attn`, `sglang`, and
   `vllm` from other pure Python dependencies in `requirements.txt` and
   `pyproject.toml`. This is primarily due to historical reasons: we previously used the
   NVIDIA PyTorch Docker image, which provides a customized PyTorch version. Installing
   packages that require PyTorch header files would overwrite the existing PyTorch
   version due to pip's dependency resolver. As a result, we isolate these
   compilation-based packages and build them separately, either in the Dockerfile or in
   `examples/env/setup-pip-deps.sh`. Both `requirements.txt` and `pyproject.toml` work
   in local environments and our provided Docker container without overwriting the
   existing PyTorch installation.

   Starting from v0.3.4, AReaL uses the SGLang Docker image as the base image, which
   provides an official PyTorch version. Almost all packages can now be installed
   without compilation. The package dependency structure may be improved in future
   releases.

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
   # Step-wise debugging: run the last failed test first
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
   pip install --upgrade jupyter-book==1.0.4.post1 # higher version might fail to build docs
   jb build docs
   ```

   **NOTE on CI/CD**: We have updated our CI/CD workflow so that tests for PRs could run
   on ephemeral GCP compute engines with 2 A100 GPUs (40GB memory). If you have
   submitted a PR with breaking changes, please contact the core developers to trigger
   the CI/CD workflow. If you have implemented a new feature, you are highly recommended
   to write your own tests and add them to our pytest workflow. Please put your tests in
   files under `areal/tests/test_*.py` and mark the tests with our pre-defined pytest
   markers:

   - `slow`: Tests that are expected to cost more than 30 seconds. They will not run in
     the CI/CD workflow unless marked with `ci`.
   - `ci`: Tests that should run in CI/CD workflow. (Only needed to be marked on `slow`
     tests)
   - `gpu`: Tests that use a single GPU.
   - `multi_gpu`: Tests that use more than one GPU.

   The tests that run in our CI/CD are selected by `pytest -m "not slow or ci"`.
   Moreover, since our CI machine only has two GPUs, please mark skip to tests that uses
   more than 2 GPUs otherwise it will fail in our CI/CD workflow. For example:

   ```python
   import pytest

   @pytest.mark.skip(reason="This test requires 4 GPUs")
   def test_some_multi_gpu_functionality():
       ...
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

## Tips for Using AI-Assisted Coding

- [AGENTS.md](AGENTS.md) is a reference guide for AI coding agents working on AReaL.
  Before letting AI make any changes, ensure it understands the codebase using
  `AGENTS.md`.

- You can use the plan mode of coding agents to generate a plan for refactoring or new
  features. Submit it as a draft PR before making any actual code changes and discuss
  with the core developers.

______________________________________________________________________

Thank you for contributing to AReaL! 🙏
