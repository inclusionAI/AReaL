# AReaL Documentation

This directory contains the documentation for AReaL built with
[Jupyter Book](https://jupyterbook.org/).

## Building the Documentation

### Prerequisites

Install the required dependencies:

```bash
pip install jupyter-book
```

### Steps

1. Generate CLI documentation:

   ```bash
   python generate_cli_docs.py
   ```

1. Build the documentation:

   ```bash
   jupyter-book build . --all
   ```

## Documentation Structure

- **CLI Reference**: Automatically generated from dataclass definitions in
  `areal/api/cli_args.py`
- **Tutorials**: Step-by-step guides for getting started
- **Best Practices**: Guidelines for efficient usage
- **Developer Guides**: In-depth technical documentation
- **Algorithm Descriptions**: Details on implemented algorithms

## Updating CLI Documentation

The CLI reference documentation is automatically generated from the help strings and
metadata in the dataclass fields of `areal/api/cli_args.py`.

When you modify CLI arguments:

1. Update the help strings in the dataclass field metadata
1. Run `python generate_cli_docs.py` to regenerate the documentation
1. Commit the updated `cli_reference.md` file

## Configuration

- `_config.yml`: Jupyter Book configuration
- `_toc.yml`: Table of contents structure
- `generate_cli_docs.py`: Script to automatically generate CLI documentation
