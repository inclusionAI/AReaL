# Docker Image Explanation: Original vs Forked Code

## Quick Answer

**You can use the original Docker image** (`ghcr.io/inclusionai/areal-runtime:v0.3.4`) **even with your forked code!**

The Docker image only provides the **runtime environment** (CUDA, PyTorch, dependencies). Your **actual AReaL code gets cloned from GitHub** inside the container, so you can use your forked branch without building a new image.

## Understanding Docker Images vs Code

### What's in the Docker Image?

The Docker image `ghcr.io/inclusionai/areal-runtime:v0.3.4` contains:

- ✅ **Base OS**: Ubuntu with CUDA support
- ✅ **Python environment**: Python 3.x with pip
- ✅ **Deep learning libraries**: PyTorch, CUDA toolkit, etc.
- ✅ **AReaL dependencies**: SGLang, vLLM, transformers, etc.
- ✅ **System libraries**: NCCL, Gloo, etc.

**What it does NOT contain:**
- ❌ Your actual AReaL code (this gets cloned from GitHub)
- ❌ Your training scripts (these are in your forked repo)

### How It Works

```
┌─────────────────────────────────────┐
│  Docker Image (Runtime Environment) │
│  - CUDA, PyTorch, Dependencies      │
│  - ghcr.io/inclusionai/areal-       │
│    runtime:v0.3.4                    │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  Container Starts                    │
│  - Mounts volumes                    │
│  - Sets environment variables        │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  Clone Your Forked Code              │
│  git clone -b DL4Math                │
│    https://github.com/nexthybrid/    │
│    AReaL.git                         │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  Install AReaL                       │
│  pip install -e .                    │
│  (Uses your forked code)             │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  Run Training                        │
│  bash examples/cloud_gsm8k/         │
│    run_training_cloud.sh             │
│  (Uses your custom scripts)          │
└─────────────────────────────────────┘
```

## Why This Works

Since you **only modified the `examples/` folder** and didn't change:
- Core AReaL library structure (`areal/` folder)
- Dependencies (`requirements.txt`, `setup.py`)
- Build system
- Dockerfile or container configuration

The original Docker image has all the dependencies your code needs. When you run `pip install -e .` inside the container, it installs AReaL using your forked code, but uses the same dependencies from the image.

## When You WOULD Need a Custom Docker Image

You would need to build a custom Docker image if you:

1. **Changed dependencies**: Modified `requirements.txt` or `setup.py` with new packages
2. **Changed system libraries**: Need different CUDA version, different PyTorch version
3. **Changed Dockerfile**: Modified the container setup itself
4. **Changed core structure**: Modified `areal/` folder in ways that require different build steps

Since you only modified `examples/`, you don't need a custom image!

## Using Your Forked Branch

### In RunPod Template

```bash
# Docker Command in template:
bash -c "cd /workspace && git clone -b DL4Math https://github.com/nexthybrid/AReaL.git && cd AReaL && pip install -e . && export WANDB_API_KEY=\$WANDB_API_KEY && bash examples/cloud_gsm8k/run_training_cloud.sh 1hour"
```

### Manual Setup

```bash
# Inside container
cd /workspace
git clone -b DL4Math https://github.com/nexthybrid/AReaL.git
cd AReaL
pip install -e .
```

## Verification

To verify you're using your forked code:

```bash
# Inside container, after cloning
cd /workspace/AReaL

# Check git remote
git remote -v
# Should show: origin  https://github.com/nexthybrid/AReaL.git

# Check branch
git branch
# Should show: * DL4Math

# Check your custom files exist
ls examples/cloud_gsm8k/
# Should show your custom training scripts
```

## Summary

| Component | Source | Notes |
|-----------|--------|-------|
| **Docker Image** | `ghcr.io/inclusionai/areal-runtime:v0.3.4` | Original image is fine - provides runtime environment |
| **AReaL Code** | `nexthybrid/AReaL` branch `DL4Math` | Your forked code with custom examples |
| **Training Scripts** | `examples/cloud_gsm8k/` in your fork | Your custom training configurations |

**Bottom line**: Use the original Docker image, clone your forked branch, and you're good to go! 🚀

## If You Need a Custom Image (Future)

If you later need to modify dependencies or system libraries, you can build a custom image:

```dockerfile
# Dockerfile
FROM ghcr.io/inclusionai/areal-runtime:v0.3.4

# Add your custom dependencies
RUN pip install your-custom-package

# Or modify system libraries
RUN apt-get update && apt-get install -y your-package
```

Then build and push:
```bash
docker build -t your-username/areal-runtime:custom .
docker push your-username/areal-runtime:custom
```

But for now, **the original image works perfectly** with your forked code!

