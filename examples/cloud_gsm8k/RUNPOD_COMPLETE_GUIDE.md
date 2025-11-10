# Complete RunPod Guide for AReaL GRPO Training

RunPod is the most economical cloud GPU platform. This complete guide covers everything you need.

## Why RunPod?

- 💰 **Best Pricing**: RTX 4090 at $0.29/hour (vs $0.40+ on other platforms)
- 💰 **Spot Instances**: 50-70% discount (RTX 4090 spot: ~$0.09/hour!)
- 📦 **Network Volumes**: Persistent storage for checkpoints
- 🚀 **Easy Setup**: Template system for quick deployment
- 🎯 **Good GPUs**: RTX 4090, A100 40GB/80GB, H100 available
- ⚡ **Fast Startup**: Pods start in seconds

## Step-by-Step Setup

### Step 1: Create RunPod Account

1. Go to https://runpod.io
2. Click "Sign Up"
3. Verify email
4. Add credits (minimum $10 recommended for testing)

### Step 2: Create Network Volume (CRITICAL for Model Persistence)

**⚠️ CRITICAL**: Network volumes are **REQUIRED** to ensure your trained models persist across pod restarts!

**Why**: 
- ✅ Checkpoints are saved to `/workspace/outputs/grpo/checkpoints/`
- ✅ Network volumes persist even after pod stops or is interrupted
- ✅ You can resume training from the last checkpoint
- ✅ Models are safe even if pod crashes or is terminated

1. **Go to "Volumes"** in RunPod dashboard
2. **Click "Create Volume"**
3. **Settings**:
   - **Name**: `areal-outputs`
   - **Size**: 50GB (enough for multiple training runs)
   - **Description**: "AReaL GRPO training outputs and checkpoints"
4. **Click "Create"**

**Note**: Volume is created and ready to use. It persists across pod restarts, crashes, and interruptions.

### Step 3: Deploy Pod

#### Method A: Using Template (Recommended - Easiest)

1. **Go to "Templates"** in RunPod dashboard
2. **Click "New Template"**
3. **Fill in template**:

   **Basic Info:**
   - **Name**: `areal-grpo-training`
   - **Container Image**: `ghcr.io/inclusionai/areal-runtime:v0.3.4`
   - **Container Disk**: 15GB (enough for code and dependencies, with headroom for training artifacts)

   **Docker Command:**
   ```bash
   bash -c "set -e && pip config set global.index-url https://pypi.org/simple && pip config set global.extra-index-url '' && cd /workspace && if [ -d AReaL/.git ]; then cd AReaL && git fetch origin && git checkout -B DL4Math origin/DL4Math 2>/dev/null || git checkout -B DL4Math origin/DL4Math 2>/dev/null || (cd .. && rm -rf AReaL && git clone -b DL4Math https://github.com/nexthybrid/AReaL.git); else rm -rf AReaL && git clone -b DL4Math https://github.com/nexthybrid/AReaL.git; fi && cd /workspace/AReaL && pip install -e . && export WANDB_API_KEY=\$WANDB_API_KEY && bash examples/cloud_gsm8k/run_training_cloud.sh 1hour"
   ```
   
   **⚠️ Important**: 
   - **Smart git handling**: If AReaL exists and is a valid git repo, it updates with `git pull` (preserves code during container restarts). Only removes/clones if invalid or missing. This prevents deleting code during training.
   - **Pip fix**: The Docker image (`ghcr.io/inclusionai/areal-runtime:v0.3.4`) is configured to use an internal PyPI mirror (`pypi.antfin-inc.com`) that's not accessible from RunPod. The command above overrides it to use the public PyPI (`pypi.org`).

   **Environment Variables:**
   - Click "Add Environment Variable" for each:
   - **Key**: `WANDB_API_KEY`
   - **Value**: Your WandB API key (get it from https://wandb.ai → Settings → API Keys)
   - **Key**: `PYTHONPATH`
   - **Value**: `/workspace/AReaL`
   
   **💡 Tip**: Your WandB API key is found at https://wandb.ai → Click your profile → Settings → API Keys

   **Volume Mounts:**
   - `/workspace/outputs` → Select your `areal-outputs` volume

   **GPU:**
   - Select: **RTX 4090** (recommended) or **A100** (faster)
   - **Spot Instance**: ✅ Enable (saves 50-70%!)

4. **Save Template**
5. **Deploy Pod** using the template
6. **Training starts automatically!**

#### Method B: Manual Pod Creation

1. **Go to "Pods"** → **"Deploy"**
2. **Select GPU**: 
   - **RTX 4090** (recommended, $0.29/hour)
   - **A100 40GB** (faster, $1.39/hour)
   - **A100 80GB** (most memory, $1.89/hour)
3. **Container Image**: `ghcr.io/inclusionai/areal-runtime:v0.3.4`
4. **Docker Command**: `/bin/bash`
5. **Environment Variables** (Click "Add Environment Variable" for each):
   - **Key**: `WANDB_API_KEY`
   - **Value**: Your WandB API key (get it from https://wandb.ai → Settings → API Keys)
   - **Key**: `PYTHONPATH`
   - **Value**: `/workspace/AReaL`
6. **Volume Mounts**:
   - `/workspace/outputs` → Your `areal-outputs` volume
7. **Spot Instance**: ✅ **Enable** (highly recommended!)
8. **Container Disk**: 15GB
9. **Click "Deploy"**

### Step 4: Connect to Pod

RunPod provides multiple connection methods:

#### Option 1: Web Terminal (Easiest)
1. Click on your pod in dashboard
2. Click **"Connect"** → **"Terminal"**
3. Web terminal opens in browser
4. You're now inside the pod!

#### Option 2: Jupyter (Optional)
1. Enable Jupyter in pod settings
2. Access via RunPod dashboard
3. Use Jupyter notebooks for interactive work

#### Option 3: SSH (Advanced)
1. Configure SSH in pod settings
2. Use provided SSH command from dashboard

### Step 5: Set Up (If Using Manual Pod)

If you used Method B (manual pod), run these commands in the web terminal:

```bash
# Inside pod (via web terminal)
cd /workspace

# Fix pip configuration (Docker image uses internal PyPI that's not accessible from RunPod)
pip config set global.index-url https://pypi.org/simple
pip config set global.extra-index-url ""

# Clone AReaL repository (DL4Math branch)
git clone -b DL4Math https://github.com/nexthybrid/AReaL.git
cd AReaL

# Install AReaL
pip install -e .

# Verify GPU
nvidia-smi

# Verify WandB API key
echo $WANDB_API_KEY
```

### Step 6: Run Training

#### If Using Template (Method A)
Training starts automatically! Skip to Step 7.

#### If Manual Setup (Method B)

```bash
# Inside pod
cd /workspace/AReaL

# Verify WandB API key is set
export WANDB_API_KEY=your-api-key-here  # If not set via environment

# Run training (choose one):
# Fast training (20-30 min, 200 samples)
bash examples/cloud_gsm8k/run_training_cloud.sh fast

# 1-hour training (500 samples, 2 epochs)
bash examples/cloud_gsm8k/run_training_cloud.sh 1hour

# 3-hour training (1000 samples, 3 epochs)
bash examples/cloud_gsm8k/run_training_cloud.sh 3hour

# Full training (all samples, 5 epochs) - takes days!
bash examples/cloud_gsm8k/run_training_cloud.sh full
```

## Step 7: Monitor Training

### RunPod Dashboard

1. **Go to your pod** in RunPod dashboard
2. **View**:
   - GPU utilization (should be 80-100%)
   - Memory usage
   - Network I/O
   - **Costs** (real-time)
3. **Logs tab**: View training logs

### WandB Dashboard

1. **Go to**: https://wandb.ai
2. **Login** with your WandB account
3. **Project**: `gsm8k-grpo-local`
4. **View**:
   - Training curves (`grpo_actor/task_reward/avg`)
   - Loss curves
   - GPU utilization
   - All metrics in real-time

## Step 8: Download Results

### Via RunPod Dashboard

1. **Go to "Volumes"** in RunPod dashboard
2. **Click on** `areal-outputs` volume
3. **Browse files**:
   - Checkpoints: `outputs/grpo/checkpoints/`
   - Logs: `outputs/grpo/logs/`
4. **Download** files directly from dashboard

### Via Web Terminal

```bash
# Inside pod
ls -lh /workspace/outputs/grpo/checkpoints/
ls -lh /workspace/outputs/grpo/logs/
```

### Via Cloud Storage (S3/GCS)

```bash
# Inside pod, upload to cloud storage
aws s3 sync /workspace/outputs s3://your-bucket/areal-outputs/
# or
gsutil -m cp -r /workspace/outputs gs://your-bucket/areal-outputs/
```

## Step 9: Stop Pod (Save Costs!)

**Important**: Stop pod when done to avoid charges!

**Before stopping, verify checkpoints are saved:**

```bash
# Inside pod, verify checkpoints exist on the volume
ls -lh /workspace/outputs/grpo/checkpoints/

# Should show your experiment directories with checkpoints
# Example:
# gsm8k-grpo-cloud-1hour/
#   └── trial0/
#       └── checkpoint_epoch_1_step_63/
#           ├── actor.pt
#           ├── optimizer.pt
#           └── ...
```

1. **Go to "Pods"** in RunPod dashboard
2. **Click "Stop"** on your pod
3. **Or enable auto-shutdown** in pod settings

**✅ Your checkpoints are safe in the network volume!** They will persist even after the pod stops.

## Step 10: Resuming Training from Checkpoint

When you want to continue training (after pod restart or interruption):

### Step 10.1: Deploy New Pod (or Restart Existing)

1. **Go to "Pods"** → **"Deploy"** (or restart existing pod)
2. **Use same settings** as before:
   - Same container image: `ghcr.io/inclusionai/areal-runtime:v0.3.4`
   - **Same volume mount**: `/workspace/outputs` → `areal-outputs` (CRITICAL!)
   - Same environment variables (WandB API key, etc.)

### Step 10.2: Verify Checkpoint Exists

```bash
# Inside pod, check that checkpoints exist
ls -lh /workspace/outputs/grpo/checkpoints/

# Find your experiment and trial
# Example:
ls -lh /workspace/outputs/grpo/checkpoints/gsm8k-grpo-cloud-1hour/trial0/
```

### Step 10.3: Set Up Repository (If Needed)

```bash
# Inside pod
cd /workspace

# Fix pip configuration (Docker image uses internal PyPI that's not accessible from RunPod)
pip config set global.index-url https://pypi.org/simple
pip config set global.extra-index-url ""

# Clone repository if not already there
if [ ! -d "AReaL" ]; then
    git clone -b DL4Math https://github.com/nexthybrid/AReaL.git
    cd AReaL
    pip install -e .
else
    cd AReaL
fi
```

### Step 10.4: Resume Training

```bash
# Inside pod
cd /workspace/AReaL

# Verify WandB API key is set
export WANDB_API_KEY=your-api-key-here  # If not set via environment

# Resume training with SAME experiment_name and trial_name
# The training script will automatically detect and load the last checkpoint
bash examples/cloud_gsm8k/run_training_cloud.sh 1hour
```

**Important**: 
- ✅ Use the **same `experiment_name` and `trial_name`** as before
- ✅ The training script automatically detects checkpoints in `outputs/grpo/checkpoints/{experiment_name}/{trial_name}/`
- ✅ Training will resume from the last saved checkpoint automatically
- ✅ Progress continues seamlessly from where it left off

### Verifying Resume Works

After starting training, check the logs:

```bash
# Training should show:
# "Loading checkpoint from: outputs/grpo/checkpoints/..."
# "Resuming from epoch X, step Y"
```

If you see "Starting training from scratch", the checkpoint wasn't found. Verify:
1. Volume is mounted at `/workspace/outputs`
2. Checkpoint path is correct
3. Experiment name and trial name match previous run

## RunPod Cost Optimization

### 1. Use Spot Instances (50-70% Savings!)

**Always enable spot instances for training:**
- RTX 4090: $0.29/hour → **$0.09/hour** (spot)
- A100: $1.39/hour → **$0.42/hour** (spot)

**Trade-off**: Spot instances may be interrupted if demand is high. But:
- ✅ Checkpoints are saved to network volume
- ✅ Can resume training from checkpoint
- ✅ 50-70% cost savings!

### 2. Auto-Shutdown

**Enable in pod settings:**
- **Idle timeout**: 10-30 minutes
- **Pod stops** when idle
- **Resume** when needed
- **Saves money** when not training

### 3. Right-Size GPU

**For AReaL GRPO training:**
- **RTX 4090**: Best value ($0.29/hour, spot: $0.09/hour)
- **A100**: Faster but 5x more expensive
- **Recommendation**: Start with RTX 4090

### 4. Monitor Costs

- **Check RunPod dashboard** regularly
- **Set budget alerts** if available
- **Stop pods** when not in use

## Cost Estimates

### RTX 4090 (Recommended)

| Training | Regular | Spot | Savings |
|----------|---------|------|---------|
| Fast (30 min) | $0.15 | $0.05 | $0.10 |
| 1-hour | $0.58 | $0.18 | $0.40 |
| 3-hour | $0.87 | $0.27 | $0.60 |
| Full (5 days) | $34.80 | $10.80 | $24.00 |

### A100 40GB

| Training | Regular | Spot | Savings |
|----------|---------|------|---------|
| Fast (30 min) | $0.70 | $0.21 | $0.49 |
| 1-hour | $2.78 | $0.84 | $1.94 |
| 3-hour | $4.17 | $1.26 | $2.91 |
| Full (5 days) | $166.80 | $50.40 | $116.40 |

**Recommendation**: Use RTX 4090 spot instances for best value!

## RunPod-Specific Features

### Network Volumes

- **Persistent storage** across pod restarts
- **Shared** between pods (if needed)
- **Fast access** (network-attached)
- **Recommended size**: 50GB for checkpoints

### Spot Instances

- **50-70% discount** on regular pricing
- **May be interrupted** if demand is high
- **Checkpoints saved** to network volume
- **Resume training** from checkpoint

### Templates

- **Save time** on repeated deployments
- **Consistent setup** across runs
- **Share with team** members
- **One-click deployment**

### Auto-Shutdown

- **Saves costs** when pod is idle
- **Configurable timeout** (10-30 min recommended)
- **Resume** when needed
- **No data loss** (checkpoints in volume)

## Troubleshooting

### Pod Won't Start

**Check:**
- GPU availability (may need to wait)
- Container image name (must be exact)
- Environment variables (WandB API key)
- Volume mount path

### Out of Memory

**Solutions:**
- Use A100 instead of RTX 4090
- Reduce `batch_size` in config (8 → 4)
- Reduce `max_new_tokens` (512 → 256)
- Enable `gradient_checkpointing: true`

### Spot Instance Interrupted

**This is normal!** Solutions:
- Checkpoints are saved to network volume
- Resume training from last checkpoint
- Or restart training (if early in training)

### Network Volume Not Mounted

**Check:**
- Volume is created in RunPod dashboard
- Volume is attached to pod
- Mount path is `/workspace/outputs` (must be exact!)
- Permissions are correct

**Verify volume is mounted:**
```bash
# Inside pod
ls /workspace/outputs/
# Should show: grpo/ directory (not empty)
```

### Checkpoints Not Persisting

**Common causes:**
- Volume not mounted at `/workspace/outputs`
- Checkpoints saved to wrong location (not on volume)
- Volume not attached to pod

**Solution:**
1. Verify volume mount in pod settings: `/workspace/outputs` → `areal-outputs`
2. Check checkpoint location: `ls /workspace/outputs/grpo/checkpoints/`
3. If empty, check if checkpoints were saved elsewhere: `find /workspace -name "checkpoint_*" -type d`
4. Ensure config uses: `cluster.fileroot: /workspace/outputs/grpo` (absolute path to mounted volume)

### Pip Installation Fails (Connection Timeout to pypi.antfin-inc.com)

**Problem**: The Docker image is configured to use an internal PyPI mirror (`pypi.antfin-inc.com`) that's not accessible from RunPod, causing connection timeouts.

**Error messages:**
- `Connection to pypi.antfin-inc.com timed out`
- `ERROR: Could not find a version that satisfies the requirement`
- `ERROR: No matching distribution found`

**Solution**: Override pip configuration to use public PyPI before installing:

```bash
# Inside pod, before pip install
pip config set global.index-url https://pypi.org/simple
pip config set global.extra-index-url ""

# Then proceed with installation
pip install -e .
```

**For templates/pod commands**, include this fix in the Docker command:
```bash
bash -c "pip config set global.index-url https://pypi.org/simple && pip config set global.extra-index-url '' && cd /workspace && git clone -b DL4Math https://github.com/nexthybrid/AReaL.git && cd AReaL && pip install -e . && ..."
```

**Why this happens**: The AReaL Docker image (`ghcr.io/inclusionai/areal-runtime:v0.3.4`) is built for internal use at Ant Group and includes pip configuration pointing to their internal PyPI mirror, which is not accessible from external cloud platforms like RunPod.

### Container Restart Loop ⚠️

**Problem**: Container keeps restarting every ~17 seconds, never actually runs training.

**Symptoms:**
- Container status shows "Restarting" repeatedly
- System logs show container starting every ~17 seconds
- Container logs show: `fatal: destination path 'AReaL' already exists and is not an empty directory`
- No training output
- GPU never gets used

**Root Cause**: Docker command tries to `git clone` AReaL, but directory already exists from previous run. `git clone` fails, container exits, RunPod auto-restarts it, creating infinite loop.

**Solution**: Updated Docker command to remove AReaL before cloning:
```bash
bash -c "pip config set global.index-url https://pypi.org/simple && pip config set global.extra-index-url '' && cd /workspace && (rm -rf AReaL || true) && git clone -b DL4Math https://github.com/nexthybrid/AReaL.git && cd AReaL && pip install -e . && export WANDB_API_KEY=\$WANDB_API_KEY && bash examples/cloud_gsm8k/run_training_cloud.sh 1hour"
```

**Key change**: Smart git handling:
- If AReaL exists and is a valid git repo: Updates it with `git pull` (preserves code during restarts)
- If AReaL exists but is invalid: Removes and clones fresh
- If AReaL doesn't exist: Clones fresh

This prevents deleting code during training while still handling restart scenarios.

**Detailed guide**: See `CONTAINER_RESTART_LOOP_FIX.md` for complete analysis.

### Training Crashes Around Step 50-60 (Repeating Pattern)

**Problem**: Training consistently crashes around step 50-60 with terminal closing and GPU stopping. This is **not** a spot interruption but an execution crash.

**Symptoms:**
- Training runs for ~50-60 steps
- Terminal closes suddenly
- GPU utilization drops to zero
- Pattern repeats across multiple runs
- No error messages in terminal

**Root Cause**: Likely a **timeout or crash during disk-based weight updates**. The weight update process has hardcoded timeouts that may be insufficient.

**Solutions** (try in order):

1. **Increase timeouts in config** (already done in A40 config):
   ```yaml
   rollout:
     request_timeout: 7200  # 2 hours (from default 3600)
     request_retries: 5  # More retries (from default 3)
     setup_timeout: 300  # 5 minutes (from default 120)
   ```

2. **Check system logs for crash details**:
   ```bash
   # Inside pod
   dmesg | tail -50
   journalctl -n 50
   dmesg | grep -i "out of memory\|killed"
   ```

3. **Check SGLang server logs**:
   ```bash
   tail -100 /workspace/AReaL/outputs/grpo/logs/*/llm_server.log
   grep -i "error\|crash\|timeout" /workspace/AReaL/outputs/grpo/logs/*/llm_server.log
   ```

4. **Check disk space and I/O**:
   ```bash
   df -h /workspace/outputs
   # Test disk speed
   dd if=/dev/zero of=/workspace/outputs/test bs=1G count=1 oflag=direct
   ```

5. **Check memory usage**:
   ```bash
   # Monitor during training
   watch -n 1 'free -h && nvidia-smi'
   ```

6. **Try regular (non-spot) instance** to rule out spot interruptions

**Detailed guide**: See `CRASH_DIAGNOSIS.md` for complete analysis and solutions.

### Training Too Slow

**Optimizations:**
- Use A100 instead of RTX 4090
- Check GPU utilization (`nvidia-smi`)
- Verify batch size is appropriate
- Check network volume performance

### CUDA Out of Memory (OOM) on A40 GPU

**Problem**: A40 GPU (44GB) runs out of memory when both SGLang inference server and trainer share the same GPU.

**Error message:**
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 48.00 MiB. 
GPU 0 has a total capacity of 44.42 GiB of which 27.50 MiB is free.
Process 2963820 has 36.34 GiB memory in use.  # SGLang server
Process 2965022 has 8.05 GiB memory in use.   # Trainer
```

**Solutions** (apply in order):

1. **Use A40-optimized config** (recommended):
   ```bash
   # Use the A40-optimized config file
   bash examples/cloud_gsm8k/run_training_cloud.sh 1hour
   # But override config to use A40 version:
   python3 -m areal.launcher.local examples/docker_gsm8k/gsm8k_grpo_1hour.py \
       --config examples/cloud_gsm8k/gsm8k_grpo_1hour_a40.yaml \
       experiment_name=gsm8k-grpo-cloud-1hour \
       trial_name=trial0
   ```

2. **Enable gradient checkpointing** (reduces training memory by ~30-40%):
   ```yaml
   actor:
     gradient_checkpointing: true  # Change from false to true
   ```

3. **Reduce SGLang memory fraction** (leaves more memory for trainer):
   ```yaml
   sglang:
     mem_fraction_static: 0.5  # Reduce from 0.8 to 0.5
   ```

4. **Reduce batch size**:
   ```yaml
   train_dataset:
     batch_size: 4  # Reduce from 8 to 4
   ```

5. **Reduce max_new_tokens**:
   ```yaml
   gconfig:
     max_new_tokens: 256  # Reduce from 512 to 256
   ```

6. **Reduce max_tokens_per_mb**:
   ```yaml
   actor:
     mb_spec:
       max_tokens_per_mb: 4096  # Reduce from 5120
   ```

7. **Set PyTorch memory allocator** (reduces fragmentation):
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   ```

8. **Reduce max_concurrent_rollouts**:
   ```yaml
   rollout:
     max_concurrent_rollouts: 16  # Reduce from 32
   ```

**Quick fix for A40**: Use the provided `gsm8k_grpo_1hour_a40.yaml` config file which includes all these optimizations.

**Memory breakdown on A40**:
- SGLang server: ~36GB (with `mem_fraction_static: 0.8`)
- Trainer: ~8GB
- Total: ~44GB (exceeds A40 capacity)

**With A40 optimizations**:
- SGLang server: ~22GB (with `mem_fraction_static: 0.5`)
- Trainer: ~6GB (with gradient checkpointing)
- Total: ~28GB (fits comfortably in A40)

## Best Practices

1. ✅ **Always use spot instances** (50-70% savings)
2. ✅ **ALWAYS use network volumes** for checkpoints (required for persistence!)
3. ✅ **Verify checkpoints exist** before stopping pod: `ls /workspace/outputs/grpo/checkpoints/`
4. ✅ **Enable auto-shutdown** to save costs (checkpoints are safe in volume)
5. ✅ **Monitor in WandB** for training progress
6. ✅ **Stop pod when done** (don't leave running - checkpoints persist in volume)
7. ✅ **Use same experiment_name/trial_name** when resuming training
8. ✅ **Download important checkpoints** to local machine as backup
9. ✅ **Use templates** for repeated runs (ensures consistent volume mounting)

## Quick Reference

### Deploy Pod
```
RunPod Dashboard → Pods → Deploy
- GPU: RTX 4090 (spot enabled)
- Image: ghcr.io/inclusionai/areal-runtime:v0.3.4
- Volume: /workspace/outputs → areal-outputs
- Env: WANDB_API_KEY=your-key
```

### Run Training
```bash
# Inside pod
cd /workspace/AReaL
bash examples/cloud_gsm8k/run_training_cloud.sh 1hour
```

### Monitor
- RunPod dashboard: GPU, costs, logs
- WandB: https://wandb.ai (training curves)

### Download Results
- RunPod dashboard → Volumes → areal-outputs
- Or use cloud storage (S3/GCS)

## Next Steps

1. ✅ Create RunPod account
2. ✅ Create network volume
3. ✅ Deploy pod (use template for easiest setup)
4. ✅ Monitor training in WandB
5. ✅ Download results
6. ✅ Stop pod (save costs!)

Happy training on RunPod! 🚀

