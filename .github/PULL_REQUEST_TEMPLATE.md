## Description

<!-- Provide a clear and concise description of what this PR does -->

## Related Issue

<!-- Link to the issue this PR addresses. PRs should be related to a well-templated issue. -->

Fixes #(issue)

## Type of Change

<!-- Mark the relevant option with an 'x' -->

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not
  work as expected)
- [ ] Documentation update
- [ ] Code refactoring (no functional changes)
- [ ] Performance improvement
- [ ] Test coverage improvement

## Code Formatting ✨

**This project uses automated code formatting.** Before submitting your PR, please
ensure your code is properly formatted:

### Option 1: Use Pre-commit Hooks (Recommended)

```bash
# Install pre-commit (one-time setup)
pip install pre-commit
pre-commit install

# Now formatting runs automatically on commit
git add .
git commit -m "your commit message"  # Auto-formats before committing
```

### Option 2: Manual Formatting

If pre-commit hooks aren't working, run these commands manually:

```bash
# Install formatting tools
pip install black==25.1.0 isort==6.0.1 autoflake==2.3.1 clang-format==19.1.7

# Format Python code
autoflake --in-place --recursive --remove-all-unused-imports --remove-unused-variables areal/ examples/
isort --profile=black --multi-line=3 --line-length=88 .
black --line-length=88 .

# Format C++/CUDA code (if you modified csrc/)
find csrc/ -type f \( -name '*.c' -o -name '*.h' -o -name '*.cpp' -o -name '*.hpp' -o -name '*.cu' -o -name '*.cuh' \) -exec clang-format -i {} +
```

### Fixing CI Formatting Failures

If the "Check Formatting" CI check fails:

1. Look at the CI logs to see which files failed
1. Run the manual formatting commands above
1. Commit the formatting changes: `git commit -am "fix: apply code formatting"`
1. Push: `git push`

**Note:** Formatting is enforced by CI and must pass before your PR can be merged.

## Testing 🧪

### For All Changes

```bash
# Install test dependencies
pip install pytest

# Run all unit tests (may take several hours)
pytest -s -v areal/tests/

# Or run specific test files related to your changes
pytest -s -v areal/tests/test_<relevant_file>.py
```

### Testing Based on Change Type

**Modified `areal/workflow/` or `areal/engine/`:**

```bash
# Run workflow/engine specific tests
pytest -s -v areal/tests/test_allocation_mode.py
pytest -s -v areal/tests/grpo/
```

**Modified `areal/dataset/` or `areal/reward/`:**

```bash
# Test data processing
pytest -s -v areal/tests/test_utils.py
```

**Modified examples:**

```bash
# Try running the example you modified
python3 -m areal.launcher.local examples/math/gsm8k_grpo.py --config examples/math/gsm8k_grpo.yaml

# For quick validation, you can use a smaller model or fewer steps
```

**Added new features:**

- Add corresponding unit tests in `areal/tests/`
- Ensure new tests pass: `pytest -s -v areal/tests/test_<your_new_test>.py`

### GPU Test Requirements

⚠️ **Many tests require GPUs and multi-node setups.** If you don't have access:

- Mark GPU-dependent tests with `@pytest.mark.skipif` and explain in PR description
- State clearly: "Tests validated locally without GPU" or "Validated through static
  analysis"
- Maintainers will run GPU tests during review

### Documentation Changes

If you modified docs:

```bash
# Install doc dependencies
pip install jupyter-book

# Build docs locally
jb build docs

# Preview changes by opening docs/_build/html/index.html in your browser
```

## Testing Checklist

<!-- Mark with 'x' what you've done -->

- [ ] I have run the formatting tools (pre-commit or manual)
- [ ] I have run relevant unit tests and they pass
- [ ] I have added tests for new functionality
- [ ] I have tested my changes work as expected
- [ ] I have updated documentation if needed
- [ ] I have noted any GPU/hardware requirements for testing

## Breaking Changes

<!-- If this PR introduces breaking changes, list them here -->

- [ ] This PR introduces breaking changes
- [ ] I have updated relevant documentation to reflect breaking changes
- [ ] I have added migration notes for users

**Breaking changes details:**

<!-- Describe what breaks and how users should migrate -->

## Documentation

<!-- List any documentation you've added or updated -->

- [ ] Code comments and docstrings
- [ ] README.md or other markdown files
- [ ] Jupyter Book documentation in `docs/`
- [ ] Example scripts or configurations
- [ ] CHANGELOG or version history

## Additional Context

<!-- Add any other context, screenshots, logs, or explanations here -->

### How Has This Been Tested?

<!-- Describe how you verified your changes work -->

**Test Configuration:**

- Hardware: (e.g., Single GPU, 4x A100, CPU only)
- Software: (e.g., CUDA 12.2, PyTorch 2.1.0)
- Commands run:

```bash
# Paste the commands you used to test
```

### Screenshots (if applicable)

<!-- Add screenshots to demonstrate UI changes, training curves, etc. -->

## Checklist Before Requesting Review

- [ ] My code follows the project's code style (Black, isort, autoflake)
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published
- [ ] I have linked this PR to the related issue
- [ ] I have checked my code and corrected any misspellings

______________________________________________________________________

**Need help?** Check the [Contributing Guide](../CONTRIBUTING.md) or ask in
[GitHub Discussions](https://github.com/inclusionAI/AReaL/discussions)!
