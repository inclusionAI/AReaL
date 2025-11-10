# Vast.ai Setup Guide

Vast.ai (https://vast.ai) provides access to consumer GPUs at competitive prices.

## Step 1: Create Vast.ai Account

1. Go to https://vast.ai
2. Sign up for an account
3. Add credits to your account

## Step 2: Find GPU Instance

1. **Go to "Search"** in Vast.ai dashboard
2. **Filter by**:
   - GPU: RTX 4090, A100, etc.
   - CUDA: 12.0+
   - Storage: 50GB+ (for models and checkpoints)
3. **Select instance** with good price/performance
4. **Rent instance**

## Step 3: Connect to Instance

Vast.ai provides SSH access:

```bash
ssh root@<instance-ip> -p <port>
```

Password or SSH key provided in Vast.ai dashboard.

## Step 4: Verify Docker and GPU

```bash
# Check Docker
docker --version

# Check GPU
nvidia-smi

# Test GPU in Docker
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

## Step 5: Set Up Persistent Storage

Vast.ai instances have local storage. For persistence:

1. **Use instance storage** (lost when instance ends)
2. **Mount external storage** (S3, GCS, etc.)
3. **Download checkpoints** before instance ends

## Step 6: Run Docker Container

```bash
# Clone repository
cd /root
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
    -v /root/AReaL:/workspace/AReaL:rw \
    -v /root/outputs:/workspace/outputs:rw \
    -w /workspace/AReaL \
    ghcr.io/inclusionai/areal-runtime:v0.3.4 \
    /bin/bash
```

## Step 7: Run Training

Inside the container:

```bash
cd /workspace/AReaL

# Install AReaL
pip install -e .

# Run training
bash examples/cloud_gsm8k/run_training_cloud.sh
```

## Vast.ai Specific Tips

### Instance Selection

1. **RTX 4090**: Best price/performance for training
2. **A100**: Faster but more expensive
3. **Multiple GPUs**: For distributed training (advanced)

### Pricing

Vast.ai uses dynamic pricing:
- **RTX 4090**: ~$0.20-0.40/hour
- **A100**: ~$1.00-2.00/hour
- Prices vary by demand

### Storage

- **Instance storage**: Usually 50-200GB
- **Download checkpoints** before instance ends
- **Or use cloud storage** (S3, GCS) for persistence

### Instance Lifecycle

1. **Rent instance** → Get SSH access
2. **Run training** → Monitor progress
3. **Download results** → Before instance ends
4. **Release instance** → Stop charges

## Monitoring

1. **Vast.ai Dashboard**: Instance status, costs
2. **SSH access**: Direct terminal access
3. **WandB**: Training metrics

## Downloading Results

Before instance ends, download checkpoints:

```bash
# From host (outside container)
# Use scp or rsync to download outputs
scp -r root@<instance-ip>:/root/outputs ./local_outputs
```

Or use cloud storage:

```bash
# Inside container, upload to S3/GCS
aws s3 sync /workspace/outputs s3://your-bucket/areal-outputs/
```

## Cost Estimation

For 3-hour training:
- RTX 4090: ~$0.60-1.20
- A100: ~$3.00-6.00

## Troubleshooting

### Instance Disconnected

- Vast.ai instances may disconnect
- Use `tmux` or `screen` for persistent sessions
- Check instance status in dashboard

### Slow Performance

- Some Vast.ai instances may be slower
- Check GPU utilization: `nvidia-smi`
- Consider different instance

### Storage Full

- Clean up old checkpoints
- Download and delete
- Use external storage

