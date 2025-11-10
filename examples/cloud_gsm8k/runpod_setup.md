# RunPod Setup Guide (Detailed)

RunPod (https://runpod.io) provides serverless GPU pods with Docker support. **This is the most economical cloud GPU platform!**

**Quick Start**: See `RUNPOD_QUICK_START.md` for fastest setup.

## Step 1: Create RunPod Account

1. Go to https://runpod.io
2. Sign up for an account
3. Add credits to your account

## Step 2: Create GPU Pod

### Option A: Using RunPod Template (Recommended)

1. **Go to "Templates"** in RunPod dashboard
2. **Create new template**:
   - **Name**: `areal-grpo-training`
   - **Container Image**: `ghcr.io/inclusionai/areal-runtime:v0.3.4`
   - **Docker Command**: 
     ```bash
     bash -c "pip config set global.index-url https://pypi.org/simple && pip config set global.extra-index-url '' && git clone -b DL4Math https://github.com/nexthybrid/AReaL.git /workspace/AReaL && cd /workspace/AReaL && pip install -e . && bash examples/cloud_gsm8k/run_training_cloud.sh"
     ```
   - **Environment Variables**:
     - `WANDB_API_KEY`: Your WandB API key
     - `PYTHONPATH`: `/workspace/AReaL`
   - **Volume Mounts**: 
     - `/workspace/outputs` → RunPod network volume (for checkpoints)

3. **Save template**

4. **Deploy pod** using the template

### Option B: Manual Pod Creation

1. **Go to "Pods"** in RunPod dashboard
2. **Click "Deploy"**
3. **Select GPU**: RTX 4090, A100, etc.
4. **Container Image**: `ghcr.io/inclusionai/areal-runtime:v0.3.4`
5. **Docker Command**: 
   ```bash
   /bin/bash
   ```
6. **Environment Variables**:
   - `WANDB_API_KEY`: Your WandB API key
   - `PYTHONPATH`: `/workspace/AReaL`
7. **Volume Mounts**: Create network volume for outputs
8. **Deploy**

## Step 3: Connect to Pod

RunPod provides:
- **Web Terminal**: Access via RunPod dashboard
- **Jupyter**: If enabled
- **SSH**: If configured

## Step 4: Set Up Inside Pod

If using manual pod creation:

```bash
# Inside pod
cd /workspace

# Fix pip configuration (Docker image uses internal PyPI that's not accessible)
pip config set global.index-url https://pypi.org/simple
pip config set global.extra-index-url ""

# Clone repository (DL4Math branch)
git clone -b DL4Math https://github.com/nexthybrid/AReaL.git
cd AReaL

# Install AReaL
pip install -e .

# Verify GPU
nvidia-smi
```

## Step 5: Run Training

```bash
# Set WandB API key (if not set via environment)
export WANDB_API_KEY=your-api-key-here

# Run training
bash examples/cloud_gsm8k/run_training_cloud.sh
```

## RunPod Specific Features

### Network Volumes

RunPod provides persistent network volumes:

1. **Create volume** in "Volumes" section
2. **Mount to pod** at `/workspace/outputs`
3. **Checkpoints saved** automatically persist

### Spot Instances

RunPod supports spot instances (cheaper):

1. **Enable "Spot"** when creating pod
2. **Save 50-70%** on costs
3. **Warning**: Pod may be terminated if demand is high

### Auto-Shutdown

Configure auto-shutdown to save costs:

1. **Set idle timeout** in pod settings
2. **Pod stops** when idle
3. **Resume** when needed

## RunPod Pricing

- **RTX 4090**: ~$0.29/hour
- **A100 40GB**: ~$1.39/hour
- **A100 80GB**: ~$1.89/hour
- **Spot instances**: 50-70% discount

## Monitoring

1. **RunPod Dashboard**: GPU utilization, costs
2. **WandB**: Training metrics
3. **Pod Logs**: View in RunPod dashboard

## Stopping Pod

**Important**: Stop pod when done!

1. **Go to "Pods"** in dashboard
2. **Click "Stop"** on your pod
3. **Or use auto-shutdown** feature

## Cost Estimation

For 3-hour training:
- RTX 4090: ~$0.87
- A100 40GB: ~$4.17
- A100 40GB (Spot): ~$1.25-2.08

## Troubleshooting

### Pod Won't Start

- Check GPU availability
- Verify container image name
- Check environment variables

### Out of Memory

- Use smaller GPU instance
- Reduce batch size
- Enable gradient checkpointing

### Network Volume Issues

- Verify volume is mounted
- Check permissions
- Use `/workspace/outputs` path

