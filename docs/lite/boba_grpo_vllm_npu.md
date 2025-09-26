# Running GRPO with vLLM on BOBA dataset on NPU

In this instruction, we will introduce how to run GRPO with vLLM on BOBA dataset on NPU.

## Prerequisites

### Hardware

The following hardware configuration has been extensively tested:

- **NPU**: 16x NPU per node
- **CPU**: 64 cores per node
- **Memory**: 1TB per node
- **Network**: RoCE 3.2 Tbps
- **Storage**:
  - 1TB local storage for single-node experiments
  - 10TB shared storage (NAS) for distributed experiments

### Dataset

Download
[AReaL-boba-106k](https://huggingface.co/datasets/inclusionAI/AReaL-boba-Data/blob/main/AReaL-boba-106k.jsonl)

### Model

Download
[DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)

## Runtime Environment

We recommend using Docker with our provided image for NPU containing vllm and
vllm-ascend.

This image is currently in testing phase. The release version and Dockerfile will be
available in AReaL repository soon.

```bash
work_dir=<your_workspace>
container_work_dir=<your_container_workspace>

# Containing vllm 0.10.2 and vllm-ascend 0.10.0
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

git clone https://github.com/inclusionAI/AReaL
cd AReaL
# Package used for calculating math reward
pip install -e evaluation/latex2sympy
# Install AReaL
pip install -e .
```

## Start training

```
python3 -m areal.launcher.ray examples/math/boba_grpo.py \
    --config examples/math/boba_grpo_vllm.yaml
```
