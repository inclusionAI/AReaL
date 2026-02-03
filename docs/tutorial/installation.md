# Installation

## Prerequisites

### Hardware Requirements

The following hardware configuration has been extensively tested:

- **GPU**: 8x H800 per node
- **CPU**: 64 cores per node
- **Memory**: 1TB per node
- **Network**: NVSwitch + RoCE 3.2 Tbps
- **Storage**:
  - 1TB local storage for single-node experiments
  - 10TB shared storage (NAS) for distributed experiments

### Software Requirements

| Component                |                                                                                                Version                                                                                                 |
| ------------------------ | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Operating System         |                                                                  CentOS 7 / Ubuntu 22.04 or any system meeting the requirements below                                                                  |
| NVIDIA Driver            |                                                                                               550.127.08                                                                                               |
| CUDA                     |                                                                                                  12.8                                                                                                  |
| Git LFS                  | Required for downloading models, datasets, and AReaL code. See [installation guide](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage) |
| Docker                   |                                                                                                 27.5.1                                                                                                 |
| NVIDIA Container Toolkit |                                         See [installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)                                          |
| AReaL Image              |                                                     `ghcr.io/inclusionai/areal-runtime:v0.5.3` (includes runtime dependencies and Ray components)                                                      |

**Note**: This tutorial does not cover the installation of NVIDIA Drivers, CUDA, or
shared storage mounting, as these depend on your specific node configuration and system
version. Please complete these installations independently.

## Runtime Environment

**For multi-node training**: Ensure a shared storage path is mounted on every node (and
mounted to the container if you are using Docker). This path will be used to save
checkpoints and logs.

### Option 1: Docker (Recommended)

We recommend using Docker with our provided image. The Dockerfile is available in the
top-level directory of the AReaL repository.

```bash
docker pull ghcr.io/inclusionai/areal-runtime:v0.5.3
docker run -it --name areal-node1 \
   --privileged --gpus all --network host \
   --shm-size 700g -v /path/to/mount:/path/to/mount \
   ghcr.io/inclusionai/areal-runtime:v0.5.3 \
   /bin/bash
git clone https://github.com/inclusionAI/AReaL /path/to/mount/AReaL
cd /path/to/mount/AReaL
uv pip install -e . --no-deps --no-build-isolation
```

### Option 2: Custom Environment Installation

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).

1. Clone the repo:

```bash
git clone https://github.com/inclusionAI/AReaL
cd AReaL
```

3. (Optional) Specify a custom index URL.

Add this section to your `pyproject.toml`:

```
[[tool.uv.index]]
url = "https://mirrors.aliyun.com/pypi/simple/"  # Change to your preferred mirror site
default = true
```

If your network connection is stable, you can skip this step.

4. Install dependencies using uv sync:

```bash
# Use `--extra cuda` on Linux with CUDA for full functionality
uv sync --extra cuda
# Or without CUDA support
# uv sync
# Or with additional packages for development and testing
# uv sync --group dev
```

This installs all CUDA-dependent packages including SGLang, vLLM, Megatron, Flash
Attention, etc. These packages require Linux x86_64 with CUDA 12.x and compatible NVIDIA
drivers.

The same command also works on macOS and Linux without CUDA support. CUDA packages are
automatically skipped via platform markers. However, training and inference features
requiring CUDA will not be available. This configuration is suitable only for
development, testing, and non-GPU workflows.

You can also install individual extras instead of the full `cuda` bundle:

- `sglang`: SGLang inference engine
- `vllm`: vLLM inference engine
- `megatron`: Megatron training backend
- `tms`: Torch Memory Saver
- `flash-attn`: Flash Attention v2
- `cuda`: All of the above (convenience extra)

**Note**: You can install these extras individually:

```bash
# If you do not need SGLang and Megatron
uv sync --extra vllm --extra flash-attn
# If you encounter connection issues when installing flash-attn
uv sync --extra vllm --extra sglang --extra megatron --extra tms
```

### Additional CUDA Packages (Optional, Manual Installation)

The Docker image includes additional compiled packages that are NOT in `pyproject.toml`.
These packages require CUDA and must be compiled from source. If you are using a custom
environment (not Docker) and need optimizations from these packages (e.g., FP8 training,
fused Adam kernel), install them manually after running `uv sync --extra cuda`:

| Package           | Purpose                                  | Installation Command                                                                                                                                              |
| ----------------- | ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| grouped_gemm      | MoE model support in Megatron            | `uv pip install --no-build-isolation git+https://github.com/fanshiqing/grouped_gemm@v1.1.4`                                                                       |
| NVIDIA apex       | Fused Adam, etc. in Megatron             | `NVCC_APPEND_FLAGS="--threads 4" APEX_PARALLEL_BUILD=8 APEX_CPP_EXT=1 APEX_CUDA_EXT=1 uv pip install --no-build-isolation git+https://github.com/NVIDIA/apex.git` |
| TransformerEngine | FP8 training, optimized GEMM in Megatron | `uv pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable`                                                                  |
| flash-attn-3      | Flash Attention v3 (Hopper)              | Built from source, see [Dockerfile](https://github.com/inclusionAI/AReaL/blob/main/Dockerfile)                                                                    |

**Important**: These packages require `--no-build-isolation` because they need access to
the already-installed PyTorch for CUDA compilation. Install PyTorch first via
`uv sync --extra cuda` before attempting to install these packages.

### DeepSeek-V3 Optimization Packages (Optional)

For running DeepSeek-V3 style MoE models with optimal performance, the Docker image also
includes the following packages. These packages have complex build requirements and GPU
architecture constraints. **Refer to the
[Dockerfile](https://github.com/inclusionAI/AReaL/blob/main/Dockerfile) for the exact
installation commands and environment variables.**

| Package                | Purpose                                     | GPU Requirement |
| ---------------------- | ------------------------------------------- | --------------- |
| FlashMLA               | Multi-head Latent Attention kernels         | SM90+ (Hopper)  |
| DeepGEMM               | FP8 GEMM library for MoE                    | SM90+ (Hopper)  |
| DeepEP                 | Expert Parallelism communication library    | SM80+ (Ampere)  |
| flash-linear-attention | Linear attention with Triton kernels        | Any GPU         |
| NVSHMEM                | Required for DeepEP internode communication | Any GPU         |

**Notes**:

- FlashMLA and DeepGEMM require Hopper (H100/H800) or newer GPUs.
- DeepEP requires NVSHMEM for internode and low-latency features. The Docker image
  installs `nvidia-nvshmem-cu12` via pip, which DeepEP auto-detects.
- flash-linear-attention uses pure Triton kernels and works on any GPU.
- These packages are cloned from GitHub and compiled during Docker build. For manual
  installation, ensure git submodules are initialized
  (`git submodule update --init --recursive`) for FlashMLA and DeepGEMM as they depend
  on CUTLASS.

5. Validate your AReaL installation:

We provide a script to validate your AReaL installation. Simply run:

```bash
uv run python3 areal/tools/validate_installation.py
```

After the validation passes, you are ready to go!

(install-skypilot)=

## (Optional) Install SkyPilot

SkyPilot helps you run AReaL easily on 17+ different clouds or your own Kubernetes
infrastructure. For more details about SkyPilot, check the
[SkyPilot Documentation](https://docs.skypilot.co/en/latest/overview.html). Below are
the minimal steps to set up SkyPilot on GCP or Kubernetes.

### Install SkyPilot with pip

```bash
# In your conda environment
# NOTE: SkyPilot requires 3.7 <= python <= 3.13
pip install -U "skypilot[gcp,kubernetes]"
```

### GCP Setup

```bash
# Install Google Cloud SDK
conda install -y -c conda-forge google-cloud-sdk

# Initialize gcloud and select your account/project
gcloud init

# (Optional) Choose a project explicitly
gcloud config set project <PROJECT_ID>

# Create Application Default Credentials
gcloud auth application-default login
```

### Kubernetes Setup

See the
[SkyPilot Kubernetes setup guide](https://docs.skypilot.co/en/latest/reference/kubernetes/kubernetes-setup.html)
for a comprehensive guide on how to set up a Kubernetes cluster for SkyPilot.

### Verify

```bash
sky check
```

If `GCP: enabled` or `Kubernetes: enabled` are shown, you're ready to use SkyPilot with
AReaL. See the
[SkyPilot example](https://github.com/inclusionAI/AReaL/blob/main/examples/skypilot/README.md)
for a detailed guide on running AReaL with SkyPilot. For more options and details, see
the official
[SkyPilot installation guide](https://docs.skypilot.co/en/latest/getting-started/installation.html).

## (Optional) Launch Ray Cluster for Distributed Training

On the first node, start the Ray Head:

```bash
ray start --head
```

On all other nodes, start Ray Workers:

```bash
# Replace with the actual IP address of the first node
RAY_HEAD_IP=xxx.xxx.xxx.xxx
ray start --address $RAY_HEAD_IP
```

You should see the Ray resource status displayed when running `ray status`.

Set the `n_nodes` argument appropriately in AReaL's training command, and AReaL will
automatically detect the resources and allocate workers to the cluster.

## Next Steps

Check the [quickstart section](quickstart.md) to launch your first AReaL job.
