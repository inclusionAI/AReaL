# Lambda AI Setup Guide

Lambda AI (https://lambdalabs.com) provides cloud GPU instances with Docker support.

## Step 1: Create Lambda AI Account

1. Go to https://lambdalabs.com
2. Sign up for an account
3. Add payment method

## Step 2: Launch GPU Instance

1. **Go to "Instances"** in Lambda AI dashboard
2. **Select GPU**: 
   - For training: A100 (40GB or 80GB)
   - For testing: A10 or RTX 4090
3. **Select OS**: Ubuntu 22.04 (recommended)
4. **Launch instance**

## Step 3: Connect to Instance

Lambda AI provides SSH access. Connect via:

```bash
ssh ubuntu@<instance-ip>
```

## Step 4: Install Docker (if not pre-installed)

```bash
# Check if Docker is installed
docker --version

# If not installed:
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker
```

## Step 5: Install NVIDIA Container Toolkit

```bash
# Add NVIDIA package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Verify
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

## Step 6: Set Up Persistent Storage (Optional)

Lambda AI supports persistent storage volumes:

1. **Create volume** in Lambda AI dashboard
2. **Attach to instance** when launching
3. **Mount point**: Usually `/data` or `/mnt/volume`

## Step 7: Run Docker Container

Use the provided script or run manually:

```bash
# Clone AReaL repository (or mount from persistent storage)
cd /home/ubuntu
git clone -b DL4Math https://github.com/nexthybrid/AReaL.git

# Set WandB API key
export WANDB_API_KEY=your-api-key-here

# Run container
docker run -it --name areal-grpo \
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

## Step 8: Run Training

Inside the container:

```bash
cd /workspace/AReaL

# Install AReaL (if needed)
pip install -e .

# Run training
bash examples/cloud_gsm8k/run_training_cloud.sh
```

## Lambda AI Specific Tips

1. **Instance Types**:
   - `gpu_1x_a100`: 1x A100 40GB - Good for training
   - `gpu_1x_a10`: 1x A10 24GB - Budget option
   - `gpu_1x_rtx_4090`: 1x RTX 4090 - Good for testing

2. **Pricing**: Check current rates on Lambda AI website
   - A100: ~$1.10/hour
   - A10: ~$0.50/hour

3. **Persistent Storage**: 
   - Create volumes in dashboard
   - Attach when launching instance
   - Mount at `/data` or custom path

4. **SSH Keys**: 
   - Add your SSH public key in Lambda AI dashboard
   - Use for secure access

## Monitoring

1. **Lambda AI Dashboard**: View instance status, GPU utilization
2. **WandB**: Training metrics and curves
3. **Container logs**: `docker logs areal-grpo`

## Stopping Instance

**Important**: Stop instance when done to avoid charges!

```bash
# From Lambda AI dashboard: Stop instance
# Or use Lambda AI CLI if installed
```

## Cost Estimation

For 3-hour training:
- A100: ~$3.30
- A10: ~$1.50
- RTX 4090: ~$0.75

## Troubleshooting

### GPU Not Available

```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

### Out of Memory

- Use smaller batch size
- Enable gradient checkpointing
- Use A100 80GB if available

### Network Issues

- Lambda AI instances have public IPs
- Use `--network host` for SGLang server
- Check security groups if issues persist

