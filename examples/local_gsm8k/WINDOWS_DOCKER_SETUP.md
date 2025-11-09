# Step-by-Step: Running GRPO Training on Windows 11 with Docker

Complete guide to set up Docker + NVIDIA GPU support for AReaL GRPO training on Windows 11 with RTX 4080 SUPER.

## Prerequisites Checklist

- ✅ Windows 11 (you have this)
- ✅ NVIDIA RTX 4080 SUPER (you have this)
- ✅ Latest NVIDIA drivers installed
- ⬜ Docker Desktop (we'll install)
- ⬜ WSL2 (we'll install/verify)
- ⬜ NVIDIA Container Toolkit (we'll install)

---

## Step 1: Verify NVIDIA Drivers

First, ensure your GPU drivers are up to date:

1. **Open NVIDIA Control Panel**: Right-click desktop → NVIDIA Control Panel
2. **Check driver version**: Help → System Information
3. **Update if needed**: Visit https://www.nvidia.com/Download/index.aspx
   - Download latest Game Ready or Studio Driver for RTX 4080 SUPER
   - Install and restart if needed

**Verify GPU is working:**
```powershell
# Open PowerShell
nvidia-smi
```

You should see your RTX 4080 SUPER listed. If `nvidia-smi` command not found:
- Drivers not installed properly
- Restart after driver installation
- Reinstall drivers if needed

---

## Step 2: Install WSL2 (Windows Subsystem for Linux)

WSL2 is required for Docker Desktop on Windows 11.

### Check if WSL2 is already installed:

```powershell
# Open PowerShell as Administrator (Right-click → Run as Administrator)
wsl --status
```

If you see "WSL 2" as the default version, skip to Step 3.

### Install WSL2:

```powershell
# Run PowerShell as Administrator
wsl --install
```

This will:
- Enable WSL feature
- Install Ubuntu (default Linux distribution)
- Set WSL 2 as default

**After installation:**
1. **Restart your computer** (required!)
2. **Launch Ubuntu** from Start menu
3. **Complete Ubuntu setup**:
   - Create username
   - Set password (you'll use this for `sudo` commands)

### Verify WSL2 Installation:

```powershell
# After restart, open PowerShell
wsl --list --verbose
```

You should see:
```
  NAME      STATE           VERSION
* Ubuntu    Running         2
```

If VERSION shows "1", convert it:
```powershell
wsl --set-version Ubuntu 2
```

---

## Step 3: Install Docker Desktop

### Download and Install:

1. **Download Docker Desktop**:
   - Visit: https://www.docker.com/products/docker-desktop/
   - Click "Download for Windows"
   - Save the installer

2. **Run Installer**:
   - Double-click `Docker Desktop Installer.exe`
   - Follow installation wizard
   - **Important**: When prompted, check "Use WSL 2 instead of Hyper-V"
   - Complete installation

3. **Start Docker Desktop**:
   - Launch from Start menu
   - Accept service agreement
   - **Wait for Docker to start** (whale icon in system tray should be steady)

### Configure Docker Desktop:

1. **Open Settings**: Right-click Docker icon in system tray → **Settings** (or **Settings** from Docker Desktop window)

2. **General Settings**:
   - ✅ **Use the WSL 2 based engine** (should be enabled by default)
   - ✅ Start Docker Desktop when you log in (optional)
   - Click **Apply & Restart** if you made changes

3. **Resources Settings**:
   - Go to **Resources** → **Advanced**
   - **CPUs**: Set to maximum available
   - **Memory**: Set to at least 16GB (or 50% of total RAM)
   - **Swap**: 2GB
   - Click **Apply & Restart**

4. **WSL Integration (CRITICAL)**:
   - Go to **Resources** → **WSL Integration**
   - **Find your Ubuntu distro** in the list (should show "Ubuntu" or similar)
   - **Enable the toggle switch** next to your Ubuntu distro
   - ✅ Make sure it shows "Enabled" status
   - Click **Apply & Restart**
   
   **If you don't see WSL Integration tab**:
   - Make sure WSL2 is installed and working
   - Restart Docker Desktop
   - Check that Ubuntu is installed and accessible

### Verify Docker Installation:

```powershell
# In PowerShell
docker --version
docker ps
```

Both commands should work without errors.

---

## Step 4: Install NVIDIA Container Toolkit in WSL2

This allows Docker containers to access your GPU.

### Step 4a: Open WSL2 Ubuntu

```powershell
# In PowerShell
wsl
# OR launch Ubuntu from Start menu
```

You should now be in a Linux terminal (prompt will look like: `username@computername:~$`)

### Step 4b: Install NVIDIA Container Toolkit

```bash
# In WSL2 Ubuntu terminal

# Update package list
sudo apt update

# Install prerequisites
sudo apt install -y ca-certificates curl gnupg lsb-release

# Add NVIDIA GPG key
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# Add repository
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Update and install
sudo apt update
sudo apt install -y nvidia-container-toolkit

# Configure Docker
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker daemon
sudo service docker restart
```

**Note**: If `sudo service docker restart` doesn't work, you can restart Docker Desktop from Windows instead.

### Step 4c: Verify GPU Access in WSL2

```bash
# In WSL2 Ubuntu
nvidia-smi
```

You should see your RTX 4080 SUPER! If not:
- Ensure NVIDIA drivers are installed on Windows
- Restart WSL2: `wsl --shutdown` (in PowerShell), then reopen Ubuntu
- Check Windows drivers are latest version

### Step 4d: Test Docker GPU Access

```bash
# In WSL2 Ubuntu
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

You should see GPU information printed. If this fails:
- Ensure Docker Desktop is running
- Check WSL integration is enabled in Docker Desktop settings
- Try restarting Docker Desktop

---

## Step 5: Prepare Your Project

### Option A: Use Project from Windows (Easier for Cursor)

Your project is at `C:\Users\tongz\git\GT\AReaL`. Docker can access this directly.

### Option B: Copy to WSL2 (Better Performance)

For faster file access:

```bash
# In WSL2 Ubuntu
cd ~
# Clone or copy your project
git clone <your-repo-url> AReaL
# OR copy from Windows
cp -r /mnt/c/Users/tongz/git/GT/AReaL ~/AReaL
```

---

## Step 6: Handle CUDA Version Compatibility

**IMPORTANT**: The official AReaL Docker image requires CUDA 12.9+, but your drivers support CUDA 12.6.

You have two options:

### Option A: Update NVIDIA Drivers (Recommended - ~15 minutes)

1. **Download latest drivers**:
   - Visit: https://www.nvidia.com/Download/index.aspx
   - Select: RTX 4080 SUPER, Windows 64-bit
   - Download latest Game Ready or Studio Driver
   - **Current minimum**: Driver 560.70+ supports CUDA 12.9+

2. **Install drivers**:
   - Run installer
   - Choose "Clean install" if offered
   - Restart computer

3. **Verify new CUDA version**:
   ```powershell
   nvidia-smi
   ```
   Look for "CUDA Version: 12.9" or higher

Then proceed to Step 7.

### Option B: Use Alternative Workaround (Not Recommended)

If you absolutely cannot update drivers right now:
- See detailed instructions in `CUDA_VERSION_FIX.md`
- This requires building a custom Docker image from CUDA 12.6 base
- This is complex and time-consuming (~1+ hours)
- **Strongly recommend Option A instead**

---

## Step 7: Pull AReaL Docker Image (Option A only)

If you updated drivers (Option A), pull the official image:

```bash
# In WSL2 Ubuntu terminal (or PowerShell with wsl prefix)
docker pull ghcr.io/inclusionai/areal-runtime:v0.3.4
```

This downloads ~10-15GB, so it may take 10-30 minutes depending on your internet.

**Verify image:**
```bash
docker images | grep areal
```

---

## Step 8: Run GRPO Training Container

### Start Container with GPU Access

```bash
# In WSL2 Ubuntu (or PowerShell)

# If using Windows path (Cursor editing works best):
docker run -it --name areal-grpo \
    --gpus all \
    --ipc=host \
    --shm-size=16g \
    -v /mnt/c/Users/tongz/git/GT/AReaL:/workspace/AReaL \
    -w /workspace/AReaL \
    ghcr.io/inclusionai/areal-runtime:v0.3.4 \
    /bin/bash

# OR if using WSL2 path (better performance):
docker run -it --name areal-grpo \
    --gpus all \
    --ipc=host \
    --shm-size=16g \
    -v ~/AReaL:/workspace/AReaL \
    -w /workspace/AReaL \
    ghcr.io/inclusionai/areal-runtime:v0.3.4 \
    /bin/bash
```

**Explanation of flags:**
- `--gpus all`: Enable GPU access
- `--ipc=host`: Better performance for multi-process apps
- `--shm-size=16g`: Shared memory (important for data loading)
- `-v ...`: Mount your project directory
- `-w ...`: Working directory inside container

You should now be inside the container (prompt changes to show you're in container).

### Verify Inside Container

```bash
# Inside container
nvidia-smi
python --version
cd /workspace/AReaL
ls
```

All should work!

---

## Step 9: Install AReaL in Container (if needed)

If using the pre-built image, AReaL might already be installed. Check:

```bash
# Inside container
python -c "import areal; print(areal.__version__)"
```

If it works, skip to Step 9.

If not, install:

```bash
# Inside container
cd /workspace/AReaL
pip install -e .
```

---

## Step 10: Run GRPO Training

```bash
# Inside container
cd /workspace/AReaL

# Set WandB API key (optional)
export WANDB_API_KEY=$(cat wandb/.wandb_api_key 2>/dev/null || echo "your-api-key")

# Run training using the Docker-optimized script
python3 -m areal.launcher.local examples/docker_gsm8k/gsm8k_grpo.py \
    --config examples/docker_gsm8k/gsm8k_grpo.yaml \
    experiment_name=gsm8k-grpo-docker \
    trial_name=trial0
```

**Note**: The `examples/docker_gsm8k/` scripts include a fix for single-GPU setups that automatically uses disk-based weight updates instead of NCCL, avoiding the "duplicate GPU" error.

**First run will:**
- Download the model (Qwen 0.5B) from HuggingFace
- Download GSM8K dataset
- Start SGLang inference server
- Begin training

This may take several minutes on first run.

---

## Step 10: Monitor Training

### View Logs

Logs appear in the terminal. You can also check:

```bash
# In another terminal (PowerShell or WSL2)
docker logs -f areal-grpo
```

### Check GPU Usage

```bash
# Inside container or from host
nvidia-smi
```

Should show Python processes using GPU.

### WandB Dashboard

If WandB is enabled, visit https://wandb.ai to see training metrics.

---

## Common Commands Reference

### Container Management

```bash
# Start existing container
docker start areal-grpo
docker exec -it areal-grpo /bin/bash

# Stop container
docker stop areal-grpo

# Remove container (if you need to recreate)
docker rm areal-grpo

# View running containers
docker ps

# View all containers (including stopped)
docker ps -a
```

### Execute Commands Without Entering Container

```bash
# Run Python script
docker exec -it areal-grpo python -m areal.launcher.local \
    examples/local_gsm8k/train_grpo.py \
    --config examples/local_gsm8k/train_grpo.yaml

# Check GPU
docker exec areal-grpo nvidia-smi

# View logs
docker logs areal-grpo
docker logs -f areal-grpo  # Follow logs (live)
```

---

## Troubleshooting

### Issue: "docker: command not found"

**Solution**: Docker Desktop not running or not in PATH
- Start Docker Desktop
- Restart terminal/WSL2

### Issue: "nvidia-smi: command not found" in WSL2

**Solution**: NVIDIA drivers not properly installed on Windows
- Update NVIDIA drivers on Windows
- Restart computer
- Check `nvidia-smi` works in Windows PowerShell first

### Issue: "docker: Error response from daemon: could not select device driver"

**Solution**: NVIDIA Container Toolkit not configured
```bash
# In WSL2
sudo nvidia-ctk runtime configure --runtime=docker
sudo service docker restart
# Or restart Docker Desktop
```

### Issue: "Cannot connect to the Docker daemon"

**Solution**: Docker Desktop not running
- Start Docker Desktop from Windows
- Wait for it to fully start (whale icon steady in system tray)

### Issue: Container runs out of memory

**Solution**: Increase shared memory
```bash
docker run --shm-size=32g ...  # Instead of 16g
```

### Issue: Files not syncing between Windows and Docker

**Solution**: 
- Use forward slashes: `/mnt/c/Users/...`
- Or copy files to WSL2 filesystem first
- Check mount path is correct in `docker run` command

### Issue: Training starts but GPU not used (CPU only)

**Solution**:
1. Verify `--gpus all` is in docker run command
2. Check `nvidia-smi` inside container shows GPU
3. Verify CUDA version compatibility

---

## Next Steps

1. **First successful run**: Let it complete 1 epoch to verify everything works
2. **Tune hyperparameters**: Edit `train_grpo.yaml` in Cursor (changes sync automatically!)
3. **Monitor training**: Use WandB dashboard
4. **Scale up**: Once working, increase epochs, batch size, etc.

---

## Quick Start Command (After Setup)

Once everything is set up, you can use this single command to start training:

```bash
# In WSL2 or PowerShell
docker start areal-grpo
docker exec -it areal-grpo bash -c \
    "cd /workspace/AReaL/examples/local_gsm8k && \
    export WANDB_API_KEY=5cd583e967c0e092a7f7be82e0479c1f71eeeab9 && \
    python -m areal.launcher.local train_grpo.py --config train_grpo.yaml"
```

---

## Need Help?

If you encounter issues:

1. Check this troubleshooting section
2. Verify each step completed successfully
3. Check Docker Desktop logs: Settings → Troubleshoot → View logs
4. Verify GPU: `nvidia-smi` in both Windows and WSL2
5. Check container logs: `docker logs areal-grpo`

Good luck with your GRPO training! 🚀


