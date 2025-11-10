# Docker Commands for Cloud Platforms

## Quick Copy-Paste Commands

### Basic Docker Run (All Platforms)

```bash
docker run -it --name areal-grpo-cloud \
    --gpus all \
    --ipc=host \
    --shm-size=16g \
    --network host \
    -e WANDB_API_KEY=your-api-key-here \
    -e PYTHONPATH=/workspace/AReaL \
    -v /workspace/AReaL:/workspace/AReaL:rw \
    -v /workspace/outputs:/workspace/outputs:rw \
    -w /workspace/AReaL \
    ghcr.io/inclusionai/areal-runtime:v0.3.4 \
    /bin/bash
```

### With Persistent Storage (Lambda AI / RunPod)

```bash
# Lambda AI: Mount persistent volume at /data
docker run -it --name areal-grpo-cloud \
    --gpus all \
    --ipc=host \
    --shm-size=16g \
    --network host \
    -e WANDB_API_KEY=your-api-key-here \
    -e PYTHONPATH=/workspace/AReaL \
    -v /home/ubuntu/AReaL:/workspace/AReaL:rw \
    -v /data:/workspace/outputs:rw \
    -w /workspace/AReaL \
    ghcr.io/inclusionai/areal-runtime:v0.3.4 \
    /bin/bash

# RunPod: Mount network volume at /workspace/outputs
docker run -it --name areal-grpo-cloud \
    --gpus all \
    --ipc=host \
    --shm-size=16g \
    --network host \
    -e WANDB_API_KEY=your-api-key-here \
    -e PYTHONPATH=/workspace/AReaL \
    -v /workspace/AReaL:/workspace/AReaL:rw \
    -v /workspace/outputs:/workspace/outputs:rw \
    -w /workspace/AReaL \
    ghcr.io/inclusionai/areal-runtime:v0.3.4 \
    /bin/bash
```

### One-Liner: Clone and Run Training

```bash
docker run -it --name areal-grpo-cloud \
    --gpus all \
    --ipc=host \
    --shm-size=16g \
    --network host \
    -e WANDB_API_KEY=your-api-key-here \
    -e PYTHONPATH=/workspace/AReaL \
    -v /workspace/outputs:/workspace/outputs:rw \
    -w /workspace \
    ghcr.io/inclusionai/areal-runtime:v0.3.4 \
    bash -c "pip config set global.index-url https://pypi.org/simple && pip config set global.extra-index-url '' && cd /workspace && if [ -d AReaL/.git ]; then cd AReaL && git fetch && git checkout DL4Math && git pull || (cd .. && rm -rf AReaL && git clone -b DL4Math https://github.com/nexthybrid/AReaL.git); else rm -rf AReaL && git clone -b DL4Math https://github.com/nexthybrid/AReaL.git; fi && cd AReaL && pip install -e . && bash examples/cloud_gsm8k/run_training_cloud.sh 1hour"
```

## Platform-Specific Commands

### Lambda AI

```bash
# SSH to instance first
ssh ubuntu@<instance-ip>

# Then run
export WANDB_API_KEY=your-api-key-here
docker run -it --name areal-grpo-cloud \
    --gpus all \
    --ipc=host \
    --shm-size=16g \
    --network host \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -e PYTHONPATH=/workspace/AReaL \
    -v /home/ubuntu/AReaL:/workspace/AReaL:rw \
    -v /data:/workspace/outputs:rw \
    -w /workspace/AReaL \
    ghcr.io/inclusionai/areal-runtime:v0.3.4 \
    /bin/bash
```

### RunPod

```bash
# In RunPod web terminal or Jupyter
export WANDB_API_KEY=your-api-key-here
docker run -it --name areal-grpo-cloud \
    --gpus all \
    --ipc=host \
    --shm-size=16g \
    --network host \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -e PYTHONPATH=/workspace/AReaL \
    -v /workspace/AReaL:/workspace/AReaL:rw \
    -v /workspace/outputs:/workspace/outputs:rw \
    -w /workspace/AReaL \
    ghcr.io/inclusionai/areal-runtime:v0.3.4 \
    /bin/bash
```

### Vast.ai

```bash
# SSH to instance first
ssh root@<instance-ip> -p <port>

# Then run
export WANDB_API_KEY=your-api-key-here
docker run -it --name areal-grpo-cloud \
    --gpus all \
    --ipc=host \
    --shm-size=16g \
    --network host \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -e PYTHONPATH=/workspace/AReaL \
    -v /root/AReaL:/workspace/AReaL:rw \
    -v /root/outputs:/workspace/outputs:rw \
    -w /workspace/AReaL \
    ghcr.io/inclusionai/areal-runtime:v0.3.4 \
    /bin/bash
```

## Inside Container: Setup and Training

```bash
# 1. Fix pip configuration (Docker image uses internal PyPI that's not accessible)
cd /workspace
pip config set global.index-url https://pypi.org/simple
pip config set global.extra-index-url ""

# 2. Clone repository (DL4Math branch, if not mounted)
git clone -b DL4Math https://github.com/nexthybrid/AReaL.git
cd AReaL

# 3. Install AReaL
pip install -e .

# 3. Verify GPU
nvidia-smi

# 4. Set WandB API key (if not set via environment)
export WANDB_API_KEY=your-api-key-here

# 5. Run training
bash examples/cloud_gsm8k/run_training_cloud.sh 1hour
```

## Training Commands

```bash
# Fast training (20-30 min)
bash examples/cloud_gsm8k/run_training_cloud.sh fast

# 1-hour training
bash examples/cloud_gsm8k/run_training_cloud.sh 1hour

# 3-hour training
bash examples/cloud_gsm8k/run_training_cloud.sh 3hour

# Full training (takes days)
bash examples/cloud_gsm8k/run_training_cloud.sh full
```

## Evaluation Commands

```bash
# Test trained model
python3 -m areal.launcher.local examples/cloud_gsm8k/test_trained_model_cloud.py \
    --config examples/cloud_gsm8k/gsm8k_grpo_1hour.yaml \
    experiment_name=gsm8k-grpo-cloud-eval \
    trial_name=full_test \
    rollout.experiment_name=gsm8k-grpo-cloud-1hour \
    rollout.trial_name=trial0
```

## Environment Variables

Set these before running Docker:

```bash
export WANDB_API_KEY=your-api-key-here
export PROJECT_PATH=/workspace/AReaL  # Optional
export OUTPUTS_PATH=/workspace/outputs  # Optional
export SHARED_MEMORY=16g  # Optional
```

## Troubleshooting Commands

```bash
# Check GPU
nvidia-smi

# Check Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi

# Check container status
docker ps -a | grep areal-grpo-cloud

# View container logs
docker logs areal-grpo-cloud

# Enter running container
docker exec -it areal-grpo-cloud bash

# Remove container
docker rm -f areal-grpo-cloud
```

