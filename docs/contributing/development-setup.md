# Development Setup Guide

This guide provides detailed instructions for setting up a development environment for
AReaL.

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation Methods](#installation-methods)
- [Development Tools](#development-tools)
- [IDE Configuration](#ide-configuration)
- [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements

- **OS:** Linux (Ubuntu 20.04+ recommended)
- **Python:** 3.10 or higher
- **RAM:** 16 GB
- **Disk Space:** 50 GB (for models and datasets)

### Recommended for Training

- **GPU:** NVIDIA GPU with 24GB+ VRAM (A100, H100, etc.)
- **CUDA:** 12.2 or higher
- **RAM:** 64 GB+
- **Disk Space:** 500 GB+ (shared storage for multi-node)

### Supported Platforms

- ✅ Linux (Ubuntu, CentOS, Debian)
- ⚠️ macOS (CPU-only, limited testing support)
- ❌ Windows (not supported, use WSL2)

## Installation Methods

### Method 1: Standard Installation (Recommended for Development)

```bash
# Clone repository
git clone https://github.com/YOUR-USERNAME/AReaL
cd AReaL

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows WSL: source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks
pip install pre-commit
pre-commit install
```

### Method 2: Docker (Recommended for Consistency)

```bash
# Build Docker image
docker build -t areal:dev -f Dockerfile .

# Run container with GPU support
docker run --gpus all -it \
  -v $(pwd):/workspace \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  areal:dev bash

# Inside container: install in editable mode
pip install -e ".[dev]"
```

### Method 3: Conda Environment

```bash
# Create conda environment
conda create -n areal python=3.10
conda activate areal

# Install PyTorch with CUDA
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia

# Install AReaL
pip install -e ".[dev]"
```

## Development Tools

### Required Tools

#### 1. Pre-commit Hooks

Automatically format code before commits:

```bash
pip install pre-commit
pre-commit install

# Test hooks manually
pre-commit run --all-files
```

#### 2. Code Formatters

```bash
# Install formatters (included in dev dependencies)
pip install black==25.1.0 isort==6.0.1 autoflake==2.3.1 clang-format==19.1.7

# Format Python code
black .
isort .
autoflake --in-place --recursive --remove-all-unused-imports --remove-unused-variables areal/ examples/

# Format C++/CUDA code
find csrc/ -name "*.cpp" -o -name "*.h" -o -name "*.cu" | xargs clang-format -i
```

#### 3. Testing Framework

```bash
# Install pytest
pip install pytest pytest-asyncio pytest-cov

# Run tests
pytest -s -v areal/tests/
```

#### 4. Documentation Builder

```bash
# Install Jupyter Book
pip install jupyter-book

# Build documentation
jb build docs

# View docs
# Open docs/_build/html/index.html in browser
```

### Optional Tools

#### 1. GPU Monitoring

```bash
# Install gpustat
pip install gpustat

# Monitor GPUs
watch -n 1 gpustat

# Or use nvidia-smi
watch -n 1 nvidia-smi
```

#### 2. Code Quality Tools

```bash
# Install linters
pip install flake8 mypy pylint

# Run linters
flake8 areal/
mypy areal/
```

#### 3. Profiling Tools

```bash
# Install profilers
pip install line_profiler memory_profiler py-spy

# Profile code
python -m line_profiler script.py
python -m memory_profiler script.py
```

## IDE Configuration

### Visual Studio Code

#### Recommended Extensions

- **Python** (ms-python.python)
- **Pylance** (ms-python.vscode-pylance)
- **Black Formatter** (ms-python.black-formatter)
- **isort** (ms-python.isort)
- **GitLens** (eamodio.gitlens)
- **Jupyter** (ms-toolsai.jupyter)

#### Settings (.vscode/settings.json)

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length=88"],
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  },
  "[python]": {
    "editor.rulers": [88],
    "editor.tabSize": 4
  },
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    "**/.pytest_cache": true,
    "**/.mypy_cache": true
  }
}
```

### PyCharm

#### Setup

1. **Open Project:** File → Open → Select AReaL directory
1. **Configure Interpreter:**
   - File → Settings → Project → Python Interpreter
   - Add → Virtualenv Environment → Existing → Select `venv/bin/python`
1. **Enable Black Formatting:**
   - File → Settings → Tools → External Tools → Add
   - Name: Black, Program: `black`, Arguments: `$FilePath$`
1. **Configure pytest:**
   - File → Settings → Tools → Python Integrated Tools
   - Testing → Default test runner: pytest

#### Recommended Plugins

- **Key Promoter X** - Learn keyboard shortcuts
- **Rainbow Brackets** - Easier to read nested code
- **GitToolBox** - Enhanced Git integration

### Vim/Neovim

#### Basic Setup

```vim
" ~/.vimrc or ~/.config/nvim/init.vim

" Python syntax
syntax on
filetype plugin indent on
set number
set expandtab
set tabstop=4
set shiftwidth=4

" Auto-format on save
autocmd BufWritePre *.py execute ':!black %'
autocmd BufWritePre *.py execute ':!isort %'
```

#### With CoC (Conquer of Completion)

```bash
# Install coc.nvim
# Then install Python extension
:CocInstall coc-python
```

## Environment Variables

### Hugging Face Cache

```bash
# Set Hugging Face cache directory (recommended for shared storage)
export HF_HOME=/path/to/shared/cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers

# For downloading models
export HF_TOKEN=your_huggingface_token  # If accessing gated models
```

### CUDA and GPU

```bash
# Specify visible GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3

# CUDA cache directory
export CUDA_CACHE_PATH=/tmp/cuda_cache

# Enable NCCL debug logging (for distributed training issues)
export NCCL_DEBUG=INFO
```

### Python Path

```bash
# Add AReaL to Python path (if not using pip install -e .)
export PYTHONPATH=/path/to/AReaL:$PYTHONPATH
```

### Wandb (for experiment tracking)

```bash
# Login to Weights & Biases
wandb login

# Set wandb directory
export WANDB_DIR=/path/to/wandb_logs
```

## Troubleshooting

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'areal'`

**Solution:**

```bash
# Install in editable mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH
```

### CUDA Out of Memory

**Problem:** `RuntimeError: CUDA out of memory`

**Solutions:**

```bash
# 1. Reduce batch size in config
# 2. Enable gradient checkpointing
# 3. Use smaller model
# 4. Clear CUDA cache
import torch
torch.cuda.empty_cache()
```

### Pre-commit Hooks Slow

**Problem:** Pre-commit hooks take too long

**Solution:**

```bash
# Run hooks on staged files only (automatic on commit)
git commit -m "message"

# Skip hooks temporarily (not recommended)
git commit --no-verify -m "message"

# Update hook revisions
pre-commit autoupdate
```

### Docker Permission Issues

**Problem:** Permission denied accessing files in Docker

**Solution:**

```bash
# Run as current user
docker run --gpus all -it \
  --user $(id -u):$(id -g) \
  -v $(pwd):/workspace \
  areal:dev bash
```

### Slow Downloads

**Problem:** Model/dataset downloads are slow

**Solution:**

```bash
# Use mirror (for users in China)
export HF_ENDPOINT=https://hf-mirror.com

# Or download manually and point to local path in config
```

### Port Already in Use

**Problem:** `Address already in use` when running services

**Solution:**

```bash
# Find process using port
lsof -i :8000

# Kill process
kill -9 <PID>

# Or use different port in config
```

## Development Workflow Tips

### Keep Fork Updated

```bash
# Fetch upstream changes
git fetch upstream

# Update main branch
git checkout main
git merge upstream/main
git push origin main

# Update feature branch
git checkout feature/my-feature
git merge main  # Or rebase: git rebase main
```

### Quick Test Loop

```bash
# Watch mode: Re-run tests on file changes
pip install pytest-watch
ptw -- -s -v areal/tests/test_utils.py
```

### Debug Mode

```bash
# Run with debug logging
export AREAL_LOG_LEVEL=DEBUG

# Python debugger
python -m pdb script.py

# Or add breakpoint in code
breakpoint()  # Python 3.7+
```

### Clean Build

```bash
# Remove build artifacts
rm -rf build/ dist/ *.egg-info
find . -type d -name __pycache__ -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

# Reinstall
pip install -e ".[dev]"
```

## Next Steps

- [Beginner's Guide](beginner-guide.md) - Make your first contribution
- [Testing Guide](testing-guide.md) - Write and run tests
- [Code Style Guide](code-style.md) - Follow coding standards
- [Contributing Guide](../../CONTRIBUTING.md) - Full contribution workflow

## Getting Help

- **Installation issues:**
  [GitHub Discussions](https://github.com/inclusionAI/AReaL/discussions)
- **GPU/CUDA issues:** Check
  [docs/best_practices/handling_oom.md](../best_practices/handling_oom.md)
- **General questions:** [WeChat Group](../../assets/wechat_qrcode.png)

______________________________________________________________________

**Having trouble?** Don't hesitate to ask for help in
[GitHub Discussions](https://github.com/inclusionAI/AReaL/discussions) or the WeChat
group!
