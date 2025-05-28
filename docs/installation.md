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
| AReaL Image | `ghcr.io/inclusionai/areal-runtime:v0.3.0` (includes runtime dependencies and Ray components) |

**Note**: This tutorial does not cover the installation of NVIDIA Drivers, CUDA, or shared storage mounting, as these depend on your specific node configuration and system version. Please complete these installations independently.

## Runtime Environment

We recommend using Docker with our provided image. The Dockerfile is available in the top-level directory of the AReaL repository.

Pull the Docker image:

```bash
docker pull ghcr.io/inclusionai/areal-runtime:v0.3.0
```

This image includes all training requirements for AReaL.

**For multi-node training**: Ensure shared storage is mounted to the `/storage` directory on every node. All downloads and resources will be stored in this directory, and the AReaL container will mount this directory to `/storage` within the container.

## Code Setup

Clone the AReaL project code to `/storage/codes`:

```bash
mkdir -p /storage/codes
cd /storage/codes/
git clone https://github.com/inclusionAI/AReaL
pip install -r AReaL/requirements.txt
```

## Dataset

Download the provided training dataset and place it in `/storage/datasets/`:

```bash
mkdir -p /storage/datasets/
cd /storage/datasets/
wget https://huggingface.co/datasets/inclusionAI/AReaL-RL-Data/resolve/main/data/boba_106k_0319.jsonl?download=true
```

## Model

We train using open-source models available on Hugging Face Hub. Here's an example using Qwen3 (ensure Git LFS is installed):

```bash
mkdir -p /storage/models
cd /storage/models
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/Qwen/Qwen3-1.7B
cd Qwen3-1.7B
git lfs pull
```

**Alternative**: You can also use the Hugging Face CLI to download models after installing the `huggingface_hub` package. Refer to the [official documentation](https://huggingface.co/docs/huggingface_hub/guides/cli) for details.