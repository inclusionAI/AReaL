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

| Component | Version |
|---|:---:|
| Operating System | CentOS 7 / Ubuntu 22.04 or any system meeting the requirements below |
| NVIDIA Driver | 550.127.08 |
| CUDA | 12.8 |
| Git LFS | Required for downloading models, datasets, and AReaL code. See [installation guide](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage) |
| Docker | 27.5.1 |
| NVIDIA Container Toolkit | See [installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |
| AReaL Image | `ghcr.io/inclusionai/areal-runtime:v0.3.0.post1` (includes runtime dependencies and Ray components) |

**Note**: This tutorial does not cover the installation of NVIDIA Drivers, CUDA, or shared storage mounting, as these depend on your specific node configuration and system version. Please complete these installations independently.

## Runtime Environment

**For multi-node training**: Ensure a shared storage path is mounted on every node (and mounted to the container if you are using Docker). This path will be used to save checkpoints and logs.

### Option 1: Docker (Recommended)

We recommend using Docker with our provided image. The Dockerfile is available in the top-level directory of the AReaL repository.

```bash
docker pull ghcr.io/inclusionai/areal-runtime:v0.3.0.post1
docker run -it --name areal-node1 \
   --privileged --gpus all --network host \
   --shm-size 700g -v /path/to/mount:/path/to/mount \
   ghcr.io/inclusionai/areal-runtime:v0.3.0.post1 \
   /bin/bash
git clone https://github.com/inclusionAI/AReaL
cd AReaL
bash examples/env/scripts/setup-container-deps.sh
```

### Option 2: Custom Environment Installation

1. Install [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) or [Anaconda](https://www.anaconda.com/docs/getting-started/anaconda/install).

2. Create a conda virtual environment:

```bash
conda create -n areal python=3.12
conda activate areal
```

3. Install pip dependencies:

```bash
git clone https://github.com/inclusionAI/AReaL
cd AReaL
bash examples/env/scripts/setup-pip-deps.sh
```

::::{note}
The SGLang patch is applied via `examples/env/scripts/setup-container-deps.sh` or `examples/env/scripts/setup-pip-deps.sh`. To confirm whether it has been applied, run `git status` in the `/sglang` directory (for Docker) or `AReaL/sglang` (for custom setups).
::::

## (Optional) Launch Ray Cluster for Distributed Training

On the first node, start the Ray Head:

```bash
ray start --head
```

On all other nodes, start the Ray Worker:

```bash
# Replace with the actual IP address of the first node
RAY_HEAD_IP=xxx.xxx.xxx.xxx
ray start --address $RAY_HEAD_IP
```

You should see the Ray resource status displayed when running `ray status`.

Properly set the `n_nodes` argument in AReaL's training command, then AReaL's training script will automatically detect the resources and allocate workers to the cluster.

## Next Steps

Check the [quickstart section](quickstart.md) to launch your first AReaL job.