# Installation (Ascend NPU)

## Prerequisites

### Hardware Requirements

The following hardware configuration has been extensively tested:

- **NPU**: 16x NPU per node
- **CPU**: 64 cores per node
- **Memory**: 1TB per node
- **Network**: RoCE 3.2 Tbps
- **Storage**:
  - 1TB local storage for single-node experiments
  - 10TB shared storage (NAS) for distributed experiments

### Software Requirements

| Component        |                                                                                                Version                                                                                                 |
| ---------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Operating System |                                                                      Ubuntu, EulerOS or any system meeting the requirements below                                                                      |
| Ascend HDK       |                                                                                                 25.2.1                                                                                                 |
| CANN             |                                                                                                8.2.RC1                                                                                                 |
| Git LFS          | Required for downloading models, datasets, and AReaL code. See [installation guide](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage) |
| Docker           |                                                                                                 27.5.1                                                                                                 |
| AReaL Image      |                                                            `swr.cn-north-9.myhuaweicloud.com/areal/areal_npu:test_v0.1` (see details below)                                                            |

**Note**: This tutorial does not cover the installation of CANN, or shared storage
mounting, as these depend on your specific node configuration and system version. Please
complete these installations independently. You can check out more details from the vLLM
Ascend community at
[this page](https://docs.vllm.ai/projects/ascend/en/latest/installation.html).

## Runtime Environment

We recommend using Docker with our provided image for NPU.

### Create Container

```bash
work_dir=<your_workspace>
container_work_dir=<your_container_workspace>

image=swr.cn-north-9.myhuaweicloud.com/areal/areal_npu:test_v0.1
container_name=areal_npu

cd ${work_dir}

docker pull ${image}

docker run -itd --cap-add=SYS_PTRACE --net=host \
--device=/dev/davinci0 \
--device=/dev/davinci1 \
--device=/dev/davinci2 \
--device=/dev/davinci3 \
--device=/dev/davinci4 \
--device=/dev/davinci5 \
--device=/dev/davinci6 \
--device=/dev/davinci7 \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
--shm-size=1200g \
-v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /sys/fs/cgroup:/sys/fs/cgroup:ro \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /var/log/npu/:/usr/slog \
-v ${work_dir}:${container_work_dir} \
--privileged=true \
--name ${container_name} \
${image}  \
/bin/bash
```

**For multi-node training**: Ensure a shared storage path is mounted on every node (and
mounted to the container if you are using Docker). This path will be used to save
checkpoints and logs.

### Custom Environment Installation

```bash
git clone https://github.com/inclusionAI/AReaL
cd AReaL

# Checkout to npu branch
git checkout npu

# Install AReaL
pip install -e .

# For torch-npu dev version or x86 machine
pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/ https://mirrors.huaweicloud.com/ascend/repos/pypi"

# Install vllm-ascend
pip install vllm-ascend==v0.11.0rc0
```

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

Properly set the `n_nodes` argument in AReaL's training command, then AReaL's training
script will automatically detect the resources and allocate workers to the cluster.

## Next Steps

Check the [quickstart section](quickstart.md) to launch your first AReaL job. The only
things you need to change for NPU: For NPU usage, update the following:

- **Training script:** use `examples/math/gsm8k_grpo_npu.py`
- **Configuration file:** use `examples/math/gsm8k_grpo_npu.yaml`

Follow the instructions there. If you want to run multi-node training with Ray, make
sure your Ray cluster is started as described above before launching the job.
