# Contributing to AReaL

Thank you for your interest in contributing to AReaL! We welcome contributions from
everyone, whether you're fixing bugs, improving documentation, adding new features, or
helping with code reviews. This guide will help you get started.

## Table of Contents

- [Quick Start](#quick-start)
- [Ways to Contribute](#ways-to-contribute)
- [Skill-Level Pathways](#skill-level-pathways)
- [Development Setup](#development-setup)
- [Contribution Workflow](#contribution-workflow)
- [Code Quality Standards](#code-quality-standards)
- [Testing Requirements](#testing-requirements)
- [Pull Request Process](#pull-request-process)
- [Getting Help](#getting-help)
- [Recognition](#recognition)

## Quick Start

1. **Fork and Clone:**

   ```bash
   # Fork the repository on GitHub, then:
   git clone https://github.com/YOUR-USERNAME/AReaL
   cd AReaL
   ```

1. **Install Development Dependencies:**

   Check our [installation guide](docs/tutorial/installation.md).

   ```bash
   # If you are using a local pip environment
   bash examples/env/setup-pip-deps.sh
   # Or you can use the docker image illustrated in the installation guide
   pip install -e .
   pip install -e ".[dev]"  # Install dev dependencies
   ```

   Note that we separate packages like `flash-attn`, `sglang`, and `vllm` from other
   pure python depedencies in `requirements.txt` and `pyproject.toml`. This is primarily
   due to a historical reason: we used the nvidia pytorch docker image, which delivers a
   customized pytorch version. Installing any packages that require compilation will
   overwrite the existing pytorch version thanks to pip dependency resolver. As a
   result, we decide to isolate these compilation-based packages and build them
   separately, either in the Dockerfile or in `examples/env/setup-pip-deps.sh`.
   `requirements.txt` and `pyproject.toml` should work both in a local environment and
   in our provided docker container.

   Starting from v0.3.4, AReaL uses the SGLang docker image as the base image. It
   delevers an official pytorch version, and almost all packages should be able to be
   installed without compilation. The package dependency structure may be improved in
   the future.

1. **Set Up Code Formatting:**

   ```bash
   pip install pre-commit
   pre-commit install
   # Run over all files if you have previous commits
   pre-commit run --all-files
   # The subsequent commits will automatically format your file
   git commit -a -m 'my change'
   ```

1. **Testing:**

   ```bash
   pytest -sv areal/tests/
   ```

   In the test suite, we:

   - Run over all examples to make sure that they can run one RL step
   - Check the individual functionalities of engines, including rollout,
     forward-backward, and weight update
   - Check the numerical consistency of our packed data format with HuggingFace padded
     input, with and without ulysses
   - Test the functionality of staleness management
   - Ensure that the GSM8k SFT loss is decreasing, RL rewards is increasing
   - Other unit-tests of individual components

   As such, some unit-tests may require multiple GPUs to run. The entrypoint scripts are
   placed under `areal/tests/torchrun`. In the corresponding test file, e.g.,
   `test_examples.py`, we use subprocess to launch distributed experiments with
   `torchrun`, and wait for the results in a file.

   Currently, the CI/CD pipeline usually fails because of network issues. We are using
   demestic machines for temporary usage. We are transitting to use machines in
   international cloud providers. Consequently, the PR contributor has to mantually run
   selective tests on a GPU machine and report the results in PR description.

1. **Find an Issue:**

   - Browse
     [good first issues](https://github.com/inclusionAI/AReaL/labels/good%20first%20issue)
   - Check [help wanted](https://github.com/inclusionAI/AReaL/labels/help%20wanted)
     issues
   - Or create a new issue using our
     [issue templates](https://github.com/inclusionAI/AReaL/issues/new/choose)

1. **Make Changes & Submit PR:**

   - Pick an issue to resolve or open a draft PR to illustrate the feature to implement
   - Create a branch: `git checkout -b your-feature-name`
   - Make your changes with proper formatting
   - Test your changes
   - Submit a pull request

## Ways to Contribute

### 🐛 Bug Reports

Found a bug? Please create a
[bug report](https://github.com/inclusionAI/AReaL/issues/new?template=bug.md) with:

- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details (commit ID, hardware, software)
- Full logs if possible

### ✨ Feature Requests

Have an idea? Submit a
[feature request](https://github.com/inclusionAI/AReaL/issues/new?template=feature.md)
with:

- Background and use case
- Proposed solution or implementation
- Benefits to the community

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

## Skill-Level Pathways

### 🌱 Beginner Contributors

**Good for:** First-time contributors or those new to RL/distributed systems

**Recommended Starting Points:**

- **Documentation tasks:** Fix typos, improve clarity, add examples
- **Good first issues:** Look for the
  [`good first issue`](https://github.com/inclusionAI/AReaL/labels/good%20first%20issue)
  label
- **Testing:** Add unit tests for existing functionality
- **Examples:** Improve example scripts or add README files

**Example Tasks:**

- Add code comments to underdocumented functions
- Fix markdown formatting in docs
- Add error messages with helpful context
- Improve logging output readability
- Add type hints to functions

**Learning Resources:**

- Start with [docs/tutorial/quickstart.md](docs/tutorial/quickstart.md)
- Review [examples/math/](examples/math/) for simple examples
- Read [areal/README.md](areal/README.md) for architecture overview

### 🚀 Intermediate Contributors

**Good for:** Contributors familiar with Python, ML, or distributed computing

**Recommended Areas:**

- **Dataset integration:** Add new dataset loaders in `areal/dataset/`
- **Reward functions:** Implement custom rewards in `areal/reward/`
- **Workflow customization:** Create new rollout workflows in `areal/workflow/`
- **Bug fixes:** Tackle bugs in core components
- **Testing infrastructure:** Improve test coverage and CI/CD

**Example Tasks:**

- Add a new dataset loader for a math/coding benchmark
- Implement a reward function for a new task type
- Create a multi-turn workflow variant
- Add integration tests for existing features
- Improve error handling in distributed components

**Learning Resources:**

- Study [docs/customization/](docs/customization/) guides
- Review existing implementations in `areal/workflow/` and `areal/reward/`
- Read the [AReaL-lite design doc](areal/README.md)

### 🔥 Advanced Contributors

**Good for:** Experienced ML/RL researchers or systems engineers

**Recommended Areas:**

- **Algorithm implementations:** Add new RL algorithms (PPO variants, etc.)
- **Training backends:** Improve FSDP/Megatron integration
- **Inference optimization:** Enhance SGLang/vLLM adapters
- **System features:** Distributed training improvements, scheduling
- **Performance optimization:** Profiling and optimization

**Example Tasks:**

- Implement a new RL algorithm (e.g., DAPO, LitePPO)
- Add expert parallelism support
- Optimize weight update synchronization
- Improve async rollout performance
- Add context parallelism features

**Learning Resources:**

- Deep dive into [AGENTS.md](AGENTS.md) for architecture details
- Review algorithm docs in [docs/algorithms/](docs/algorithms/)
- Study [areal/engine/](areal/engine/) implementations
- Read the [research paper](https://arxiv.org/pdf/2505.24298)

## Contribution Workflow

### 1. Create or Find an Issue

- Search [existing issues](https://github.com/inclusionAI/AReaL/issues)
- Create a new issue using appropriate [templates](.github/ISSUE_TEMPLATE/)
- Get feedback on your proposal before starting large changes

### 2. Fork and Create a Branch

```bash
# Add upstream remote
git remote add upstream https://github.com/inclusionAI/AReaL.git

# Create a feature branch
git checkout -b feature/your-feature-name
# or for bugfixes
git checkout -b fix/issue-description
```

**Branch Naming Convention:**

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation changes
- `refactor/description` - Code refactoring
- `test/description` - Test improvements

### 3. Make Your Changes

- Write clean, readable code
- Follow existing code style and patterns
- Add comments for complex logic
- Update documentation as needed
- Add tests for new functionality

### 4. Format Your Code

**Automatic (Recommended):**

```bash
git add .
git commit -m "your message"  # Pre-commit hooks auto-format
```

**Manual:**

```bash
autoflake --in-place --recursive --remove-all-unused-imports --remove-unused-variables areal/ examples/
isort --profile=black .
black .
# For C++ code:
find csrc/ -type f \( -name '*.cpp' -o -name '*.h' -o -name '*.cu' \) -exec clang-format -i {} +
```

### 5. Test Your Changes

```bash
# Run relevant tests
pytest -s -v areal/tests/test_<relevant>.py

# Or run all tests (takes several hours)
pytest -s -v areal/tests/
```

See [Testing Requirements](#testing-requirements) for details.

### 6. Commit Your Changes

Follow [Conventional Commits](https://www.conventionalcommits.org/) style:

```bash
git commit -m "feat: add support for custom reward functions"
git commit -m "fix: resolve race condition in weight updates"
git commit -m "docs: improve quickstart guide"
git commit -m "test: add unit tests for RLVR workflow"
```

**Commit Message Format:**

- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `refactor:` - Code refactoring
- `test:` - Test additions/changes
- `chore:` - Maintenance tasks
- `perf:` - Performance improvements

### 7. Push and Create Pull Request

```bash
# Push to your fork
git push origin feature/your-feature-name
```

Then create a PR on GitHub:

- Use the [PR template](.github/PULL_REQUEST_TEMPLATE.md)
- Link to the related issue
- Describe what changed and why
- List testing you performed
- Mark as draft if work-in-progress

## Code Quality Standards

### Code Formatting

We use automated formatting tools:

- **Python:** Black (line length: 88), isort, autoflake
- **C++/CUDA:** clang-format (Google style)
- **Markdown:** mdformat
- **Notebooks:** nbstripout, nbqa-black, nbqa-isort

**All formatting is enforced by CI and must pass.**

### Code Style Guidelines

1. **Type Hints:** Add type annotations to function signatures

   ```python
   def compute_reward(prompt: str, completion: str) -> float:
       ...
   ```

1. **Docstrings:** Use Google-style docstrings for classes and public functions

   ```python
   def rollout_batch(data: List[Dict[str, Any]]) -> Dict[str, Any]:
       """Execute rollout for a batch of prompts.

       Args:
           data: List of prompt dictionaries containing input_ids and metadata.

       Returns:
           Dictionary with padded tensors of trajectories and rewards.
       """
   ```

1. **Imports:** Organize imports (isort handles this)

   - Standard library
   - Third-party packages
   - Local/application imports

1. **Logging:** Use structured logging, not print statements

   ```python
   from areal.utils.logging import getLogger
   logger = getLogger(__name__)
   logger.info("Starting rollout", extra={"batch_size": len(data)})
   ```

1. **Async Code:** Keep workflows non-blocking, use `await` for I/O

   ```python
   async def arun_episode(self, engine: InferenceEngine, data: Dict) -> Dict:
       resp = await engine.agenerate(req)  # Non-blocking
   ```

1. **Configuration:** Use dataclasses in `areal/api/cli_args.py`, avoid hardcoded paths

1. **Error Handling:** Provide helpful error messages with context

See [AGENTS.md](AGENTS.md) for detailed code patterns and conventions.

## Testing Requirements

### Unit Tests

Add tests for new functionality:

```python
# areal/tests/test_my_feature.py
import pytest

def test_my_new_function():
    result = my_new_function(input_data)
    assert result == expected_output
```

### Running Tests

```bash
# Run all tests
pytest -s -v areal/tests/

# Run specific test file
pytest -s -v areal/tests/test_utils.py

# Run tests matching a pattern
pytest -s -v -k "test_allocation"
```

### GPU Requirements

⚠️ Many tests require GPUs. If you don't have access:

- Mark GPU tests:
  `@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")`
- State testing limitations in PR description
- Maintainers will run GPU tests during review

### Documentation Tests

```bash
# Build docs locally
pip install jupyter-book
jb build docs

# Check for broken links and formatting
mdformat --check docs/
```

## Pull Request Process

### Before Submitting

- [ ] Code is formatted (pre-commit hooks pass)
- [ ] Tests pass locally
- [ ] Documentation updated if needed
- [ ] Commit messages follow conventions
- [ ] Branch is up to date with main

### PR Guidelines

1. **Link to Issue:** PRs must reference a GitHub issue

   - Use `Fixes #123` or `Closes #456` in PR description

1. **Use the Template:** Fill out
   [.github/PULL_REQUEST_TEMPLATE.md](.github/PULL_REQUEST_TEMPLATE.md)

1. **Clear Description:** Explain what, why, and how

1. **List Testing:** Describe how you tested changes

1. **Mark Breaking Changes:** Highlight any breaking changes

1. **Request Review:** Tag maintainers or wait for automatic assignment

### Review Process

- Maintainers will review within a few days
- Address feedback by pushing new commits
- Don't force-push after review starts (unless requested)
- Resolve conversations when addressed
- Squash commits before merge (or maintainer will do it)

### After Merge

- Delete your feature branch
- Celebrate! 🎉
- Your contribution will appear in the next release notes

## Getting Help

### Questions?

- **GitHub Discussions:** [Ask in Q&A](https://github.com/inclusionAI/AReaL/discussions)
- **WeChat Group:** [Join our community](./assets/wechat_qrcode.png)
- **Documentation:** [Read the docs](https://inclusionai.github.io/AReaL/)
- **Issue Comments:** Ask on the related issue

### Stuck on Something?

- Check [docs/best_practices/debugging.md](docs/best_practices/debugging.md)
- Review [examples/](examples/) for similar implementations
- Look at [AGENTS.md](AGENTS.md) for architecture guidance
- Ask in GitHub Discussions - we're here to help!

### Found a Security Issue?

Please see [SECURITY.md](SECURITY.md) for responsible disclosure process.

## Recognition

We value all contributions! Contributors will be:

- Listed in release notes
- Acknowledged in the project
- Part of a growing community building state-of-the-art AI agents

Thank you for contributing to AReaL! 🙏

______________________________________________________________________

**Detailed Guides:**

- [Beginner's Guide](docs/contributing/beginner-guide.md) - Step-by-step for
  first-timers
- [Development Setup](docs/contributing/development-setup.md) - Detailed environment
  setup
- [Testing Guide](docs/contributing/testing-guide.md) - Comprehensive testing
  instructions
- [Code Style Guide](docs/contributing/code-style.md) - Best practices and patterns

**Quick Links:**

- [Good First Issues](https://github.com/inclusionAI/AReaL/labels/good%20first%20issue)
- [Help Wanted](https://github.com/inclusionAI/AReaL/labels/help%20wanted)
- [Project Roadmap](ROADMAP.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
