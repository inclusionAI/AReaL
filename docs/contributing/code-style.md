# Code Style Guide

This guide describes the coding standards and best practices for AReaL beyond automatic
formatting.

## Table of Contents

- [Python Style](#python-style)
- [Type Hints](#type-hints)
- [Docstrings](#docstrings)
- [Naming Conventions](#naming-conventions)
- [Code Organization](#code-organization)
- [Common Patterns](#common-patterns)
- [Anti-Patterns](#anti-patterns)

## Python Style

### Formatting (Automatic)

The following are handled automatically by formatters:

- **Black:** Line length (88 chars), indentation, spacing
- **isort:** Import organization
- **autoflake:** Remove unused imports and variables

Just run `pre-commit install` and formatting happens on commit.

### Line Length

- **Limit:** 88 characters (Black's default)
- **Exception:** Long URLs, file paths in comments

```python
# Good: Within 88 chars
result = some_function(
    parameter1=value1,
    parameter2=value2,
    parameter3=value3,
)

# Acceptable: Long URL in comment
# See documentation: https://very-long-url.com/path/to/documentation/page.html
```

### Imports

Organized by isort into three groups:

1. Standard library
1. Third-party packages
1. Local/application imports

```python
# Standard library
import os
import sys
from typing import Dict, List, Optional

# Third-party
import torch
import numpy as np
from transformers import AutoTokenizer

# Local
from areal.api import InferenceEngine
from areal.utils.logging import getLogger
```

## Type Hints

### Always Use Type Hints

Add type annotations to all function signatures:

```python
# Good
def compute_reward(
    prompt: str,
    completion: str,
    answer: str,
) -> float:
    ...

# Avoid
def compute_reward(prompt, completion, answer):
    ...
```

### Common Types

```python
from typing import Dict, List, Optional, Union, Any, Tuple, Callable

# Basic types
def process_text(text: str) -> str: ...
def count_tokens(tokens: List[int]) -> int: ...
def get_config() -> Dict[str, Any]: ...

# Optional (can be None)
def find_user(id: int) -> Optional[User]: ...

# Union (one of several types)
def parse_input(data: Union[str, List[str]]) -> List[str]: ...

# Callable (functions)
def apply_fn(fn: Callable[[int], int], value: int) -> int: ...

# Tuples
def split_data(data: List[int]) -> Tuple[List[int], List[int]]: ...
```

### Tensor Types

```python
import torch
from torch import Tensor

def forward(
    input_ids: Tensor,  # Shape: [batch, seq_len]
    attention_mask: Optional[Tensor] = None,  # Shape: [batch, seq_len]
) -> Tensor:  # Shape: [batch, seq_len, vocab_size]
    """Include shape information in docstring."""
    ...
```

### Generic Types

```python
from typing import TypeVar, Generic

T = TypeVar('T')

def first_element(items: List[T]) -> T:
    return items[0]
```

## Docstrings

### Use Google Style

```python
def rollout_batch(
    data: List[Dict[str, Any]],
    workflow: RolloutWorkflow,
    should_accept: Optional[Callable] = None,
) -> Dict[str, Tensor]:
    """Execute rollout for a batch of prompts.

    Submits each prompt to the inference engine and waits for all
    responses to complete. Optionally filters responses using a
    should_accept predicate.

    Args:
        data: List of prompt dictionaries, each containing 'input_ids'
            and optional metadata fields.
        workflow: Rollout workflow defining the generation strategy.
        should_accept: Optional function that returns True if a response
            should be included in the batch. Signature: (response) -> bool.

    Returns:
        Dictionary containing:
            - input_ids: Padded tensor of shape [batch, max_seq_len]
            - rewards: Tensor of shape [batch]
            - Additional workflow-specific fields

    Raises:
        ValueError: If data is empty or contains invalid prompts.
        TimeoutError: If rollout exceeds configured timeout.

    Example:
        >>> data = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5]}]
        >>> batch = rollout_batch(data, rlvr_workflow)
        >>> print(batch["rewards"])
        tensor([0.85, 0.92])
    """
    ...
```

### Module/Class Docstrings

```python
"""Dataset loading utilities for reinforcement learning.

This module provides loaders for common RL datasets including math
problems, coding tasks, and conversational data. All loaders return
StatefulDataLoader instances for checkpoint-safe iteration.

Typical usage:
    loader = create_gsm8k_loader(split='train', batch_size=32)
    for batch in loader:
        process(batch)
"""

class RLVRWorkflow:
    """Reinforcement Learning with Verifier Rewards workflow.

    Generates multiple completions per prompt and scores them using
    a reward function. Designed for single-turn tasks with verifiable
    rewards (e.g., math problems, code generation).

    Attributes:
        config: Workflow configuration including n_samples and temperature.
        tokenizer: Tokenizer for encoding/decoding text.
        reward_fn: Function computing rewards from completions.
    """
```

## Naming Conventions

### Variables and Functions

```python
# Use snake_case
user_name = "Alice"
total_count = 100

def compute_loss(logits, labels):
    ...

def get_device_count():
    ...
```

### Classes

```python
# Use PascalCase
class RolloutWorkflow:
    ...

class InferenceEngine:
    ...

class PPOActor:
    ...
```

### Constants

```python
# Use UPPER_SNAKE_CASE
MAX_SEQUENCE_LENGTH = 2048
DEFAULT_TEMPERATURE = 0.7
BATCH_SIZE = 32
```

### Private/Internal

```python
# Prefix with underscore
class MyClass:
    def __init__(self):
        self._internal_state = {}
        self.__private_method()  # Name mangling

    def _helper_function(self):
        """Internal helper, not part of public API."""
        ...

    def __private_method(self):
        """Truly private, name mangled."""
        ...
```

### Abbreviations

Use clear, standard abbreviations:

```python
# Good
config = load_config()
max_len = 512
num_samples = 10

# Avoid unclear abbreviations
cfg = load_config()  # Use 'config' instead
ml = 512  # What does 'ml' mean?
n = 10  # Too vague
```

## Code Organization

### File Structure

```python
"""Module docstring explaining purpose."""

# Imports
import os
from typing import Dict

import torch

from areal.api import InferenceEngine

# Constants
DEFAULT_BATCH_SIZE = 32

# Module-level variables (if needed)
logger = getLogger(__name__)

# Helper functions
def _internal_helper():
    ...

# Main classes/functions
class MainClass:
    ...

# Entry point (if script)
if __name__ == "__main__":
    main()
```

### Class Structure

```python
class MyClass:
    """Class docstring."""

    # Class variables
    class_var = "value"

    def __init__(self):
        """Initialize instance."""
        # Instance variables
        self.public_var = 1
        self._internal_var = 2

    # Public methods
    def public_method(self):
        """Public API method."""
        ...

    # Properties
    @property
    def computed_value(self):
        """Computed property."""
        return self._internal_var * 2

    # Private/internal methods
    def _internal_method(self):
        """Internal helper method."""
        ...

    # Magic methods last
    def __str__(self):
        return f"MyClass(public_var={self.public_var})"
```

## Common Patterns

### Logging

Use structured logging, not print:

```python
from areal.utils.logging import getLogger

logger = getLogger(__name__)

# Good
logger.info("Starting training", extra={"epoch": epoch, "batch_size": 32})
logger.warning("High loss detected", extra={"loss": loss_value})

# Avoid
print(f"Starting training epoch {epoch}")
```

### Error Handling

```python
# Provide helpful error messages
def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            f"Please check that the path is correct and the file exists.\n"
            f"Expected location: {path}"
        )

# Catch specific exceptions
try:
    result = risky_operation()
except ValueError as e:
    logger.error(f"Invalid value: {e}")
    raise
except FileNotFoundError:
    logger.warning("File not found, using default")
    result = default_value()
```

### Configuration

Use dataclasses for configuration:

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Configuration for model training."""

    model_name: str
    batch_size: int = 32
    learning_rate: float = 1e-4
    max_epochs: int = 10
    checkpoint_dir: Optional[str] = None

    def __post_init__(self):
        """Validate configuration."""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
```

### Async Code

Keep workflows non-blocking:

```python
import asyncio
import aiofiles

# Good: Use async I/O
async def save_results(results: Dict, path: str):
    async with aiofiles.open(path, 'w') as f:
        await f.write(json.dumps(results))

# Avoid: Blocking I/O in async function
async def save_results_bad(results: Dict, path: str):
    with open(path, 'w') as f:  # Blocks event loop!
        f.write(json.dumps(results))
```

## Anti-Patterns

### Don't Use Bare `except`

```python
# Bad
try:
    risky_operation()
except:  # Catches everything, including KeyboardInterrupt!
    pass

# Good
try:
    risky_operation()
except (ValueError, TypeError) as e:
    logger.error(f"Operation failed: {e}")
```

### Don't Mutate Default Arguments

```python
# Bad
def add_item(item, items=[]):
    items.append(item)  # Mutates shared default!
    return items

# Good
def add_item(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items
```

### Don't Use Global State

```python
# Bad
global_counter = 0

def increment():
    global global_counter
    global_counter += 1

# Good
class Counter:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1
```

### Avoid Deep Nesting

```python
# Bad: Too deep
def process(data):
    if data:
        if data.get('valid'):
            if data['value'] > 0:
                if data['type'] == 'expected':
                    return data['value']
    return None

# Good: Early returns
def process(data):
    if not data:
        return None
    if not data.get('valid'):
        return None
    if data['value'] <= 0:
        return None
    if data['type'] != 'expected':
        return None
    return data['value']
```

### Don't Repeat Yourself (DRY)

```python
# Bad: Repetitive code
def process_user(user):
    name = user['name'].strip().lower()
    email = user['email'].strip().lower()
    address = user['address'].strip().lower()
    return {'name': name, 'email': email, 'address': address}

# Good: Extract common logic
def normalize_text(text: str) -> str:
    return text.strip().lower()

def process_user(user):
    return {
        key: normalize_text(value)
        for key, value in user.items()
    }
```

## Project-Specific Conventions

### Workflow Pattern

```python
class MyWorkflow(RolloutWorkflow):
    """Custom workflow implementation."""

    async def arun_episode(
        self,
        engine: InferenceEngine,
        data: Dict[str, Any]
    ) -> Dict[str, Tensor]:
        """Generate trajectories for a single episode.

        This method must be async and use await for engine calls.
        """
        # Create request
        req = ModelRequest(...)

        # Non-blocking generation
        resp = await engine.agenerate(req)

        # Compute reward
        reward = self.reward_fn(...)

        return result
```

### Engine Pattern

```python
class MyEngine(TrainEngine):
    """Custom training engine."""

    def initialize(self, addr: Optional[str], ft_spec: Optional[FinetuneSpec]):
        """Initialize distributed environment and load model."""
        ...

    def train_batch(self, input_: Dict, loss_fn: Callable, loss_weight_fn: Callable):
        """Execute training step on batch."""
        ...
```

## Getting Help

- **Examples:** Look at existing code in `areal/` for patterns
- **AGENTS.md:** See [AGENTS.md](../../AGENTS.md) for architecture patterns
- **Questions:** Ask in
  [GitHub Discussions](https://github.com/inclusionAI/AReaL/discussions)

______________________________________________________________________

**Remember:** Consistent style makes code easier to read, understand, and maintain for
everyone!
