# RunPod Quick Start Guide

RunPod is the most economical option for cloud GPU training. This guide gets you started quickly.

## Why RunPod?

- ✅ **Best pricing**: RTX 4090 at ~$0.29/hour
- ✅ **Spot instances**: 50-70% discount
- ✅ **Network volumes**: Persistent storage
- ✅ **Easy setup**: Template system
- ✅ **Good GPUs**: RTX 4090, A100, H100 available

## Step 1: Create RunPod Account

1. Go to https://runpod.io
2. Sign up for an account
3. Add credits to your account (minimum $10 recommended)

## Step 2: Create Network Volume (Required for Persistence)

**⚠️ IMPORTANT**: You MUST use a network volume to ensure your trained models persist across pod restarts!

1. **Go to "Volumes"** in RunPod dashboard
2. **Click "Create Volume"**
3. **Name**: `areal-outputs`
4. **Size**: 50GB (enough for checkpoints and logs)
5. **Click "Create"**

**Why this is critical**: 
- ✅ Checkpoints are saved to `/workspace/outputs/grpo/checkpoints/`
- ✅ Network volumes persist even when pod stops
- ✅ You can resume training from the last checkpoint
- ✅ Models are safe even if pod is interrupted or stopped

## Step 3: Deploy Pod

### Option A: Using RunPod Template (Easiest)

1. **Go to "Templates"** in RunPod dashboard
2. **Click "New Template"**
3. **Fill in**:
   - **Name**: `areal-grpo-training`
   - **Container Image**: `ghcr.io/inclusionai/areal-runtime:v0.3.4`
   - **Docker Command**: 
     ```bash
     bash -c "set -e && pip config set global.index-url https://pypi.org/simple && pip config set global.extra-index-url '' && cd /workspace && if [ -d AReaL/.git ]; then cd AReaL && git fetch origin && git checkout -B DL4Math origin/DL4Math 2>/dev/null || git checkout -B DL4Math origin/DL4Math 2>/dev/null || (cd .. && rm -rf AReaL && git clone -b DL4Math https://github.com/nexthybrid/AReaL.git); else rm -rf AReaL && git clone -b DL4Math https://github.com/nexthybrid/AReaL.git; fi && cd /workspace/AReaL && (python3 -c 'import areal' 2>/dev/null || pip install -e .) && export WANDB_API_KEY=\$WANDB_API_KEY && bash examples/cloud_gsm8k/run_training_cloud.sh 1hour"
     ```
     
     **⚠️ Important**: 
     - The Docker image uses an internal PyPI mirror that's not accessible from RunPod. The command above overrides it to use the public PyPI.
     - **Smart git handling**: If AReaL exists and is valid, it updates with `git fetch` and `git checkout` (safe during restarts). Only removes/clones if invalid or missing.
     - **Smart installation**: Only installs AReaL if not already installed (`python3 -c 'import areal'` check). This prevents re-installing dependencies on every pod restart.
   - **Environment Variables** (Click "Add Environment Variable" for each):
     - **Key**: `WANDB_API_KEY`
     - **Value**: `your-actual-wandb-api-key-here` (get it from https://wandb.ai → Settings → API Keys)
     - **Key**: `PYTHONPATH`
     - **Value**: `/workspace/AReaL`
   - **Volume Mounts**:
     - `/workspace/outputs` → Your `areal-outputs` volume
   - **GPU**: RTX 4090 (or A100 for faster training)
   - **Spot Instance**: ✅ Enable (saves 50-70%)
4. **Save Template**
5. **Deploy Pod** using the template

### Option B: Manual Pod Creation

1. **Go to "Pods"** → **"Deploy"**
2. **Select GPU**: RTX 4090 (recommended) or A100
3. **Container Image**: `ghcr.io/inclusionai/areal-runtime:v0.3.4`
4. **Docker Command**: `/bin/bash`
5. **Environment Variables** (Click "Add Environment Variable"):
   - **Key**: `WANDB_API_KEY`
   - **Value**: `your-actual-wandb-api-key-here` (get it from https://wandb.ai → Settings → API Keys)
   - **Key**: `PYTHONPATH`
   - **Value**: `/workspace/AReaL`
6. **Volume Mounts**: 
   - `/workspace/outputs` → Your `areal-outputs` volume
7. **Spot Instance**: ✅ Enable (recommended)
8. **Deploy**

## Step 4: Connect to Pod

RunPod provides multiple ways to connect:

### Option A: Web Terminal
1. Click on your pod in RunPod dashboard
2. Click "Connect" → "Terminal"
3. Web terminal opens in browser

### Option B: Jupyter (Optional)
1. Enable Jupyter in pod settings
2. Access via RunPod dashboard

### Option C: SSH (Advanced)
1. Configure SSH in pod settings
2. Use provided SSH command

## Step 5: Set Up Inside Pod

If using manual pod creation (Option B):

```bash
# Inside pod (via web terminal)
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

If using template (Option A), setup is automatic!

## Step 6: Run Training

### If Using Template
Training starts automatically! Monitor in:
- RunPod dashboard (logs tab)
- WandB dashboard

### If Manual Setup

```bash
# Inside pod
cd /workspace/AReaL

# Verify WandB API key is set
echo $WANDB_API_KEY

# Run training (choose one):
# Fast training (20-30 min)
bash examples/cloud_gsm8k/run_training_cloud.sh fast

# 1-hour training
bash examples/cloud_gsm8k/run_training_cloud.sh 1hour

# 3-hour training
bash examples/cloud_gsm8k/run_training_cloud.sh 3hour

# Full training (takes days)
bash examples/cloud_gsm8k/run_training_cloud.sh full
```

## Step 7: Monitor Training

1. **RunPod Dashboard**: 
   - View GPU utilization
   - Check logs
   - Monitor costs

2. **WandB Dashboard**: 
   - Go to https://wandb.ai
   - Project: `gsm8k-grpo-local`
   - View training curves

## Step 8: Download Results

### Option A: Via RunPod Dashboard
1. Go to your volume (`areal-outputs`)
2. Download files directly from dashboard

### Option B: Via Web Terminal
```bash
# Inside pod, check outputs
ls -lh /workspace/outputs/grpo/checkpoints/
```

### Option C: Via Cloud Storage
```bash
# Upload to S3/GCS (if configured)
aws s3 sync /workspace/outputs s3://your-bucket/areal-outputs/
```

## RunPod Cost Optimization

### Use Spot Instances
- **Enable "Spot"** when creating pod
- **Saves 50-70%** on costs
- **Warning**: Pod may be interrupted if demand is high

### Auto-Shutdown
- **Set idle timeout** in pod settings
- **Pod stops** when idle (saves money)
- **Resume** when needed

### Right-Size GPU
- **RTX 4090**: Best for training (~$0.29/hour)
- **A100**: Faster but more expensive (~$1.39/hour)
- **A100 Spot**: Best value (~$0.42-0.70/hour)

## Cost Estimates

For 3-hour training:
- **RTX 4090**: ~$0.87
- **RTX 4090 Spot**: ~$0.26-0.35 (70% savings!)
- **A100**: ~$4.17
- **A100 Spot**: ~$1.26-2.08 (70% savings!)

## RunPod-Specific Tips

1. **Network Volumes**: Use for persistent storage
2. **Spot Instances**: Always enable for training
3. **Templates**: Save time on repeated runs
4. **Auto-Shutdown**: Set to save costs
5. **Monitor Costs**: Check dashboard regularly

## Troubleshooting

### Pod Won't Start
- Check GPU availability
- Verify container image name
- Check environment variables

### Out of Memory (OOM)

**For A40 GPU**: Use the A40-optimized config:
```bash
# The script auto-detects A40 and uses optimized config
bash examples/cloud_gsm8k/run_training_cloud.sh 1hour
```

**Manual fix**: If still OOM, reduce memory usage:
- Enable `gradient_checkpointing: true` in config
- Reduce `sglang.mem_fraction_static: 0.5` (from 0.8)
- Reduce `train_dataset.batch_size: 4` (from 8)
- Reduce `gconfig.max_new_tokens: 256` (from 512)

**For other GPUs**: 
- Use RTX 4090 or A100 (more memory)
- Reduce batch size in config
- Enable gradient checkpointing

### Network Volume Issues
- Verify volume is mounted
- Check permissions
- Use `/workspace/outputs` path

### Spot Instance Interrupted
- This is normal for spot instances
- Checkpoints are saved to volume
- Resume training from checkpoint (see "Resuming Training" section below)

## Ensuring Model Persistence Across Pod Restarts

### ✅ Critical Steps for Persistence

1. **Always use Network Volume**: Mount your `areal-outputs` volume at `/workspace/outputs`
2. **Verify Checkpoints Location**: Checkpoints are saved to `/workspace/outputs/grpo/checkpoints/`
3. **Verify Before Stopping Pod**: Always check that checkpoints exist before stopping pod

### Verifying Checkpoints Are Saved

```bash
# Inside pod, check that checkpoints exist on the volume
ls -lh /workspace/outputs/grpo/checkpoints/

# Should show directories like:
# gsm8k-grpo-cloud-1hour/
#   └── trial0/
#       └── checkpoint_epoch_1_step_63/
```

### Resuming Training from Checkpoint

When you restart a pod (after stopping or interruption):

1. **Deploy new pod** (or restart existing) with same volume mounted
2. **Clone repository** (if not already there):
   ```bash
   cd /workspace
   git clone -b DL4Math https://github.com/nexthybrid/AReaL.git
   cd AReaL
   pip install -e .
   ```
3. **Verify checkpoint exists**:
   ```bash
   ls -lh /workspace/outputs/grpo/checkpoints/gsm8k-grpo-cloud-1hour/trial0/
   ```
4. **Resume training** (training script will automatically resume from last checkpoint):
   ```bash
   cd /workspace/AReaL
   export WANDB_API_KEY=your-api-key-here
   bash examples/cloud_gsm8k/run_training_cloud.sh 1hour
   ```

**Note**: The training script automatically detects and resumes from the last checkpoint if it exists in the output directory.

### Best Practices for Persistence

1. ✅ **Always mount network volume** at `/workspace/outputs`
2. ✅ **Verify checkpoints exist** before stopping pod: `ls /workspace/outputs/grpo/checkpoints/`
3. ✅ **Check checkpoint frequency** in config: `saver.freq_epochs: 1` (saves after each epoch)
4. ✅ **Download important checkpoints** to local machine as backup
5. ✅ **Use same experiment_name and trial_name** when resuming to continue from same checkpoint

## Next Steps

1. ✅ Deploy pod using template
2. ✅ Monitor training in WandB
3. ✅ Verify checkpoints are saved: `ls /workspace/outputs/grpo/checkpoints/`
4. ✅ Stop pod when done (checkpoints persist in volume!)
5. ✅ Resume training later by deploying new pod with same volume

## Key Points for Model Persistence

**✅ DO:**
- Always mount network volume at `/workspace/outputs`
- Verify checkpoints exist before stopping pod
- Use same experiment_name/trial_name when resuming
- Check checkpoint location: `/workspace/outputs/grpo/checkpoints/`

**❌ DON'T:**
- Don't save checkpoints to `/workspace` (not persistent)
- Don't forget to mount the volume when creating pod
- Don't use different experiment_name when resuming (won't find checkpoint)

See `RUNPOD_COMPLETE_GUIDE.md` for more detailed instructions.

