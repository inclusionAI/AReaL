# Contribution Guide

Thank you for your interest in contributing to AReaL! We welcome contributions from everyone, whether you're fixing bugs, improving documentation, or adding new system and algorithmic features.

## Setting Up Your Development Environment

New contributors do not have write permissions to the official repository. Please fork the repository and clone your fork locally. AReaL is fully Python-based, making installation straightforward.

```bash
git clone https://github.com/${your-username}/AReaL
cd AReaL
pip3 install -r requirements.txt
pip3 install -e .
```

## Issue Guidelines

### Issue Templates

Please follow the [issue template on GitHub](https://github.com/inclusionAI/AReaL/tree/main/.github/ISSUE_TEMPLATE). Issues can be:
- Bug reports
- Feature requests  
- Refactor requests

The required fields in the template help reduce communication overhead when resolving issues. **Issues with arbitrary formatting may be ignored.**

## Pull Request Guidelines

There are no specific PR templates, but **pull requests should be related to a well-templated issue**. Your PR should:
- Explain how the issue is resolved
- Describe the benefits this change will provide
- Reference the related issue number

## Code Quality

### Code Formatting

Please format your code before opening a PR:

```bash
isort . && black .
```

### Running Tests

AReaL's unit tests are based on the `pytest` framework:

```bash
# Run all tests (excluding GPU tests)
pytest -m "not gpu"

# Run a specific test case
pytest tests/test_something.py
```

**Note**: Running all tests may take several hours to complete.

## Documentation

Writing documentation is an excellent starting point for new contributors. The documentation is located in the `docs` folder and built using [Jupyter Book](https://jupyterbook.org/en/stable/intro.html).

### Adding New Documentation

1. Create your documentation files in the `docs` folder
2. Add the file path to `docs/_toc.yaml`
3. Build the documentation:

```bash
jb build docs
```

4. Preview your changes by opening the HTML files in `docs/_build/html`

This process allows you to see how your documentation will appear before submitting your contribution.