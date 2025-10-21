# Testing Guide

This guide explains how to write and run tests for AReaL contributions.

## Overview

AReaL uses `pytest` for testing. Tests are located in `areal/tests/` and cover:

- Unit tests for individual components
- Integration tests for workflows and engines
- Performance tests (require GPUs)

## Running Tests

### Install Test Dependencies

```bash
pip install pytest
```

### Run All Tests

```bash
# Run all tests (may take several hours)
pytest -s -v areal/tests/
```

### Run Specific Test Files

```bash
# Run a single test file
pytest -s -v areal/tests/test_utils.py

# Run tests in a directory
pytest -s -v areal/tests/grpo/
```

### Run Specific Tests

```bash
# Run tests matching a pattern
pytest -s -v -k "test_allocation"

# Run a specific test function
pytest -s -v areal/tests/test_utils.py::test_concat_padded_tensors
```

### Pytest Flags Explained

- `-s` - Show print statements (don't capture stdout)
- `-v` - Verbose output (show test names)
- `-k` - Only run tests matching pattern
- `-x` - Stop on first failure
- `--lf` - Run last failed tests
- `--tb=short` - Shorter traceback format

## Test Organization

```
areal/tests/
├── test_utils.py              # Utility function tests
├── test_allocation_mode.py    # Allocation mode parsing tests
├── test_data.py              # Data processing tests
├── grpo/                     # GRPO algorithm tests
│   ├── test_grpo_*.py
├── test_fsdp_*.py           # FSDP engine tests (require GPU)
└── test_sglang_engine.py    # SGLang tests (require GPU/server)
```

## Writing Tests

### Basic Test Structure

```python
# areal/tests/test_my_feature.py
import pytest
import torch
from areal.utils.my_module import my_function


def test_my_function_basic():
    """Test basic functionality of my_function."""
    result = my_function([1, 2, 3])
    expected = [2, 4, 6]
    assert result == expected


def test_my_function_edge_case():
    """Test edge case: empty input."""
    result = my_function([])
    assert result == []


def test_my_function_raises_error():
    """Test that invalid input raises ValueError."""
    with pytest.raises(ValueError):
        my_function(None)
```

### Test Naming Conventions

- File names: `test_<module_name>.py`
- Function names: `test_<what_it_tests>()`
- Use descriptive names that explain what's being tested

### Fixtures

Use pytest fixtures for reusable test setup:

```python
import pytest


@pytest.fixture
def sample_data():
    """Provide sample data for tests."""
    return {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "labels": torch.tensor([[2, 3, 4], [5, 6, 7]]),
    }


def test_process_data(sample_data):
    """Test data processing with sample data."""
    result = process_data(sample_data)
    assert result.shape == (2, 3)
```

### Parametrized Tests

Test multiple inputs efficiently:

```python
import pytest


@pytest.mark.parametrize("input,expected", [
    ([1, 2, 3], 6),
    ([0], 0),
    ([-1, -2], -3),
    ([], 0),
])
def test_sum_list(input, expected):
    """Test sum_list with various inputs."""
    assert sum_list(input) == expected
```

## GPU Tests

### Marking GPU Tests

Use `pytest.mark.skipif` for GPU-dependent tests:

```python
import pytest
import torch


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Requires GPU"
)
def test_fsdp_training():
    """Test FSDP training (requires GPU)."""
    # GPU-specific test code
    pass
```

### Running GPU Tests

```bash
# Run only GPU tests
pytest -s -v -m "not skipif" areal/tests/

# Skip GPU tests
pytest -s -v -m "skipif" areal/tests/
```

### Multi-GPU Tests

Some tests require multiple GPUs:

```python
import pytest
import torch


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Requires 2+ GPUs"
)
def test_tensor_parallel():
    """Test tensor parallelism (requires 2+ GPUs)."""
    # Multi-GPU test code
    pass
```

## Testing Best Practices

### 1. Test One Thing at a Time

```python
# Good: Tests one specific behavior
def test_compute_reward_correct_answer():
    reward = compute_reward("What is 2+2?", "4")
    assert reward == 1.0


# Avoid: Tests multiple things
def test_compute_reward():
    # Tests correct answer
    assert compute_reward("What is 2+2?", "4") == 1.0
    # Tests wrong answer
    assert compute_reward("What is 2+2?", "5") == 0.0
    # Tests edge case
    assert compute_reward("", "") == 0.0
```

Better: Split into three separate tests.

### 2. Use Meaningful Assertions

```python
# Good: Clear assertion with message
def test_batch_size():
    batch = create_batch(data)
    assert len(batch) == 32, f"Expected batch size 32, got {len(batch)}"


# Avoid: Vague assertion
def test_batch_size():
    batch = create_batch(data)
    assert batch
```

### 3. Test Edge Cases

```python
def test_tokenize():
    # Normal case
    assert tokenize("hello") == [101, 102]

    # Edge cases
    assert tokenize("") == []
    assert tokenize(" ") == [101]
    assert tokenize("a" * 1000)  # Very long input
```

### 4. Clean Up Resources

```python
import tempfile
import os


def test_save_checkpoint():
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False) as f:
        filepath = f.name

    try:
        # Test code
        save_checkpoint(model, filepath)
        assert os.path.exists(filepath)
    finally:
        # Clean up
        if os.path.exists(filepath):
            os.remove(filepath)
```

Or use fixtures:

```python
@pytest.fixture
def temp_file():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        filepath = f.name
    yield filepath
    # Cleanup happens automatically after test
    if os.path.exists(filepath):
        os.remove(filepath)


def test_save_checkpoint(temp_file):
    save_checkpoint(model, temp_file)
    assert os.path.exists(temp_file)
```

## Testing Different Components

### Testing Workflows

```python
import pytest
from areal.workflow.rlvr import RLVRWorkflow


@pytest.mark.asyncio
async def test_rlvr_workflow():
    """Test RLVR workflow generates correct format."""
    workflow = RLVRWorkflow(config)

    data = {"messages": [{"role": "user", "content": "test"}]}
    result = await workflow.arun_episode(mock_engine, data)

    # Check output format
    assert "input_ids" in result
    assert "rewards" in result
    assert result["input_ids"].shape[0] == config.n_samples
```

### Testing Reward Functions

```python
from areal.reward.math import math_reward


def test_math_reward_correct():
    """Test reward for correct answer."""
    reward = math_reward(
        prompt="What is 2+2?",
        completion="4",
        prompt_ids=None,
        completion_ids=None,
        answer="4"
    )
    assert reward == 1.0


def test_math_reward_incorrect():
    """Test reward for incorrect answer."""
    reward = math_reward(
        prompt="What is 2+2?",
        completion="5",
        prompt_ids=None,
        completion_ids=None,
        answer="4"
    )
    assert reward == 0.0
```

### Testing Data Processing

```python
from areal.utils.data import concat_padded_tensors


def test_concat_padded_tensors():
    """Test concatenating tensors with different lengths."""
    tensors = [
        torch.tensor([[1, 2]]),
        torch.tensor([[3, 4, 5]]),
    ]

    result = concat_padded_tensors(tensors, pad_value=0)

    expected = torch.tensor([[1, 2, 0], [3, 4, 5]])
    assert torch.equal(result, expected)
```

## Continuous Integration

### CI Test Workflow

Tests run automatically on every PR via GitHub Actions:

- `.github/workflows/test-areal-unit.yml` - Unit tests
- `.github/workflows/test-areal-grpo.yml` - GRPO integration tests
- `.github/workflows/test-areal-sft.yml` - SFT integration tests

### Viewing CI Results

1. Go to your PR on GitHub
1. Scroll to "Checks" section
1. Click on failed checks to see error logs
1. Fix issues and push - tests will re-run automatically

### Running CI Tests Locally

```bash
# Install CI dependencies
pip install -e ".[dev]"

# Run the same tests as CI
pytest -s -v areal/tests/

# Check formatting (like CI does)
pre-commit run --all-files
```

## Troubleshooting

### Tests Pass Locally But Fail in CI

- **Different Python version:** CI uses Python 3.10
- **Missing dependencies:** Check `pyproject.toml` dependencies
- **Environment variables:** CI may have different env vars
- **File paths:** Use `os.path.join()` for cross-platform compatibility

### Slow Tests

```bash
# Run tests in parallel (requires pytest-xdist)
pip install pytest-xdist
pytest -s -v -n auto areal/tests/
```

### Memory Issues

```bash
# Run fewer tests at once
pytest -s -v areal/tests/test_utils.py

# Clear CUDA cache in tests
import torch
torch.cuda.empty_cache()
```

### Import Errors

```bash
# Install package in editable mode
pip install -e .

# Or set PYTHONPATH
export PYTHONPATH=/path/to/AReaL:$PYTHONPATH
```

## Coverage Reports

Check test coverage:

```bash
# Install coverage
pip install pytest-cov

# Run tests with coverage
pytest --cov=areal --cov-report=html areal/tests/

# Open htmlcov/index.html to see coverage report
```

## When to Write Tests

### Always Write Tests For:

- New features
- Bug fixes (add a test that fails before fix, passes after)
- Public APIs
- Complex logic

### Tests Are Optional For:

- Trivial changes (typos in docs)
- Experimental code (mark as experimental)
- Temporary debugging code

## Need Help?

- **Example tests:** Look at existing tests in `areal/tests/`
- **Pytest docs:** [https://docs.pytest.org](https://docs.pytest.org)
- **Ask questions:**
  [GitHub Discussions](https://github.com/inclusionAI/AReaL/discussions)

______________________________________________________________________

**Remember:** Good tests make code easier to maintain and give confidence that changes
don't break existing functionality!
