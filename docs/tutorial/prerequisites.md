# Prerequisites

## Hardware Requirements

The following hardware configuration has been extensively tested:

- **GPU**: 8x H800 per node
- **CPU**: 64 cores per node
- **Memory**: 1TB per node
- **Network**: NVSwitch + RoCE 3.2 Tbps
- **Storage**: 
  - 1TB local storage for single-node experiments
  - 10TB shared storage (NAS) for distributed experiments

## Software Requirements

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

**For multi-node training**: Ensure a shared storage path is mounted on every node (and mounted to the container if your are using docker). This path will be used to save checkpoints and logs.
