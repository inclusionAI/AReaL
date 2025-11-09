# Quick Start: Enter Container and Run Training

## Enter Docker Container from Windows 11

### Option 1: If Container Already Running

```bash
# In PowerShell or WSL2
docker exec -it areal-grpo bash
```

### Option 2: Start New Container (if not running)

```bash
# In WSL2 or PowerShell
cd /mnt/c/Users/tongz/git/GT/AReaL

docker run -it --name areal-grpo \
    --gpus all \
    --ipc=host \
    --shm-size=16g \
    -v /mnt/c/Users/tongz/git/GT/AReaL:/workspace/AReaL:rw \
    -w /workspace/AReaL \
    ghcr.io/inclusionai/areal-runtime:v0.3.4 \
    /bin/bash
```

**Note**: If container name already exists, remove it first:
```bash
docker stop areal-grpo
docker rm areal-grpo
# Then run docker run again
```

## Run Training Script

Once inside the container:

```bash
# 1. Navigate to workspace
cd /workspace/AReaL

# 2. Set WandB API key
export WANDB_API_KEY=$(cat examples/local_gsm8k/wandb/.wandb_api_key 2>/dev/null || echo "e1adc5be02c03fd34828c84b1ece937e0c2feb6e")

# 3. Run training
python3 -m areal.launcher.local examples/docker_gsm8k/gsm8k_grpo.py \
    --config examples/docker_gsm8k/gsm8k_grpo.yaml \
    experiment_name=gsm8k-grpo-docker \
    trial_name=trial0
```

## One-Liner (if container already running)

```bash
docker exec -it areal-grpo bash -c "cd /workspace/AReaL && export WANDB_API_KEY=\$(cat examples/local_gsm8k/wandb/.wandb_api_key 2>/dev/null || echo 'e1adc5be02c03fd34828c84b1ece937e0c2feb6e') && python3 -m areal.launcher.local examples/docker_gsm8k/gsm8k_grpo.py --config examples/docker_gsm8k/gsm8k_grpo.yaml experiment_name=gsm8k-grpo-docker trial_name=trial0"
```

## Check Container Status

```bash
# Check if container is running
docker ps | grep areal-grpo

# Check all containers (including stopped)
docker ps -a | grep areal-grpo
```

## View Training Logs

```bash
# From Windows (outside container)
docker logs -f areal-grpo

# Or from inside container
tail -f /workspace/AReaL/outputs/grpo/logs/root/gsm8k-grpo-docker/trial0/trainer.log
```

## Stop Container

```bash
# Stop container (keeps it for later)
docker stop areal-grpo

# Remove container (deletes it)
docker rm areal-grpo
```

