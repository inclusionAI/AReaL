# Using Cursor with Docker for AReaL Development

Yes! You can absolutely use Cursor on Windows to edit and debug code while running it in Docker. Here's how to set up the perfect workflow.

## How It Works

When you mount your Windows directory as a Docker volume, you get:
- ✅ **Real-time file sync**: Edit files in Cursor → changes immediately visible in Docker
- ✅ **Shared filesystem**: Both Windows and Docker see the same files
- ✅ **Full IDE features**: Cursor's IntelliSense, debugging, and Git work normally
- ✅ **Docker execution**: Code runs in Linux environment with all dependencies

## Setup: Mount Your Project Directory

### Option 1: Mount Entire Project (Recommended)

```bash
# In WSL2 or PowerShell
docker run -it --name areal-training \
    --gpus all \
    --shm-size=16g \
    -v C:\Users\tongz\git\GT\AReaL:/workspace/AReaL \
    -w /workspace/AReaL \
    ghcr.io/inclusionai/areal-runtime:v0.3.4 \
    /bin/bash
```

**Windows path format**: Use forward slashes or double backslashes:
- `C:\Users\tongz\git\GT\AReaL` → `/c/Users/tongz/git/GT/AReaL` (in WSL2)
- Or `C:/Users/tongz/git/GT/AReaL` (Docker Desktop handles this)

### Option 2: Using WSL2 Path (Better Performance)

WSL2 paths are faster than Windows mounts:

```bash
# First, copy/clone repo to WSL2 filesystem (one-time)
# In WSL2 Ubuntu terminal:
cd ~
git clone <your-repo-url> AReaL
# OR copy from Windows
cp -r /mnt/c/Users/tongz/git/GT/AReaL ~/AReaL

# Then mount from WSL2
docker run -it --name areal-training \
    --gpus all \
    --shm-size=16g \
    -v ~/AReaL:/workspace/AReaL \
    -w /workspace/AReaL \
    ghcr.io/inclusionai/areal-runtime:v0.3.4 \
    /bin/bash
```

**Trade-off**: Faster performance, but need to sync changes between Windows and WSL2.

## Workflow Options

### Workflow 1: Edit in Cursor, Execute in Docker Terminal

1. **Open project in Cursor** (Windows):
   ```powershell
   # Open Cursor in your project
   cd C:\Users\tongz\git\GT\AReaL
   cursor .
   ```

2. **Edit files in Cursor** (Windows):
   - Edit `examples/local_gsm8k/train_grpo.py`
   - Edit `examples/local_gsm8k/train_grpo.yaml`
   - Changes save to Windows filesystem

3. **Execute in Docker** (WSL2/PowerShell terminal):
   ```bash
   # Connect to running container
   docker exec -it areal-training /bin/bash
   
   # Your changes are already there! Run:
   cd /workspace/AReaL/examples/local_gsm8k
   python -m areal.launcher.local train_grpo.py --config train_grpo.yaml
   ```

### Workflow 2: Use Cursor's Integrated Terminal with Docker

1. **Open Cursor**

2. **Open Integrated Terminal** (Ctrl+`)

3. **Run Docker commands in Cursor's terminal**:
   ```powershell
   # Start container (if not running)
   docker start areal-training
   
   # Execute Python in container
   docker exec -it areal-training python -m areal.launcher.local \
       examples/local_gsm8k/train_grpo.py \
       --config examples/local_gsm8k/train_grpo.yaml
   ```

4. **Edit files normally in Cursor** - changes sync automatically!

### Workflow 3: Hybrid - Develop Locally, Test in Docker

Use Cursor's Python extension for syntax checking, then run in Docker:

1. **Setup local Python environment** (for IntelliSense):
   ```powershell
   # In Cursor terminal
   python -m venv .venv-local
   .venv-local\Scripts\Activate.ps1
   pip install torch transformers datasets  # Just for IDE support
   ```

2. **Select this interpreter in Cursor**:
   - Press `Ctrl+Shift+P`
   - "Python: Select Interpreter"
   - Choose `.venv-local`

3. **Edit code with full IntelliSense support**

4. **Test in Docker**:
   ```bash
   docker exec -it areal-training bash -c \
       "cd /workspace/AReaL/examples/local_gsm8k && \
       python -m areal.launcher.local train_grpo.py --config train_grpo.yaml"
   ```

## Cursor Features That Work

### ✅ What Works
- **File editing**: Edit any file, changes sync to Docker
- **Git integration**: Commit, push, pull all work normally
- **Search/Find**: Search across codebase
- **Multi-file editing**: Open multiple files
- **Terminal**: Run Docker commands
- **Extensions**: Most extensions work (Python, Git, etc.)

### ⚠️ Limitations
- **Python Debugger**: Can't directly debug Docker Python processes from Cursor
- **Python IntelliSense**: May not work perfectly if Docker has different packages
- **Breakpoints**: Need to use Python debugger inside Docker

## Debugging Setup

### Option A: Debug Inside Docker Container

1. **Add breakpoints in Cursor** (they're in the code)

2. **Run with debugger in Docker**:
   ```bash
   docker exec -it areal-training bash
   cd /workspace/AReaL/examples/local_gsm8k
   
   # Install debugpy for remote debugging
   pip install debugpy
   
   # Run with debugger
   python -m debugpy --listen 0.0.0.0:5678 --wait-for-client \
       -m areal.launcher.local train_grpo.py --config train_grpo.yaml
   ```

3. **Connect from Cursor** (if it supports remote debugging)

### Option B: Use Print Statements + Docker Logs

```bash
# In Cursor, add print() statements
# Save file

# In Docker, run and watch output
docker logs -f areal-training

# Or run interactively
docker exec -it areal-training bash
python your_script.py
```

### Option C: Use Jupyter/IPython in Docker

```bash
# Install Jupyter in container
docker exec -it areal-training pip install jupyter ipython

# Start Jupyter
docker exec -it areal-training jupyter notebook \
    --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Access from Windows browser: http://localhost:8888
# Edit code in Cursor, test cells in Jupyter
```

## Best Practices

### 1. Use `.dockerignore`

Create `.dockerignore` to exclude unnecessary files:

```dockerignore
.venv/
__pycache__/
*.pyc
.git/
.pytest_cache/
wandb/
outputs/
*.log
```

### 2. Persistent Container

Keep container running between edits:

```bash
# Start container (keeps running)
docker start areal-training

# Execute commands anytime
docker exec -it areal-training <command>

# Stop when done
docker stop areal-training
```

### 3. Hot Reload Development

For iterative development:

```bash
# Run in watch mode (if your script supports it)
docker exec -it areal-training bash -c \
    "cd /workspace/AReaL/examples/local_gsm8k && \
    watch -n 1 'python -m areal.launcher.local train_grpo.py --config train_grpo.yaml'"
```

### 4. Separate Dev/Test Containers

```bash
# Dev container (lightweight, for testing)
docker run -it --name areal-dev \
    --gpus all \
    -v C:\Users\tongz\git\GT\AReaL:/workspace/AReaL \
    ghcr.io/inclusionai/areal-runtime:v0.3.4 \
    /bin/bash

# Training container (full resources)
docker run -it --name areal-training \
    --gpus all \
    --shm-size=16g \
    -v C:\Users\tongz\git\GT\AReaL:/workspace/AReaL \
    ghcr.io/inclusionai/areal-runtime:v0.3.4 \
    /bin/bash
```

## Quick Reference Commands

```powershell
# In Cursor's integrated terminal (PowerShell):

# Start container
docker start areal-training

# Execute Python script
docker exec -it areal-training python -m areal.launcher.local \
    examples/local_gsm8k/train_grpo.py --config examples/local_gsm8k/train_grpo.yaml

# View logs
docker logs -f areal-training

# Interactive shell
docker exec -it areal-training /bin/bash

# Check GPU in container
docker exec areal-training nvidia-smi

# Copy file to container (if needed)
docker cp local_file.py areal-training:/workspace/AReaL/

# Copy file from container
docker cp areal-training:/workspace/AReaL/outputs ./outputs
```

## Example Complete Workflow

1. **Open project in Cursor**:
   ```powershell
   cd C:\Users\tongz\git\GT\AReaL
   cursor .
   ```

2. **Edit `train_grpo.yaml`** in Cursor:
   - Change `total_train_epochs: 5` to `total_train_epochs: 1` (for testing)
   - Save (Ctrl+S)

3. **In Cursor's terminal, run Docker**:
   ```powershell
   docker exec -it areal-training bash -c \
       "cd /workspace/AReaL/examples/local_gsm8k && \
       python -m areal.launcher.local train_grpo.py --config train_grpo.yaml"
   ```

4. **See output in Cursor terminal** - all logs appear!

5. **Edit code again** → save → run again → iterate

## Tips

- **File watching**: Changes sync immediately (no manual sync needed)
- **Performance**: WSL2 mounts (`~/AReaL`) are faster than Windows mounts (`C:\...`)
- **Git**: Works normally - commit from Cursor, push from anywhere
- **Extensions**: Python extension works for syntax checking
- **Terminal**: Use Cursor's integrated terminal for Docker commands

## Troubleshooting

### Changes Not Reflecting

```bash
# Force remount
docker restart areal-training

# Or check mount
docker inspect areal-training | grep Mounts -A 20
```

### Permission Issues

```bash
# Fix permissions in container
docker exec -it areal-training chown -R $(id -u):$(id -g) /workspace/AReaL
```

### Path Issues

Use forward slashes or WSL2 paths:
- ✅ `C:/Users/...` or `/mnt/c/Users/...`
- ❌ `C:\Users\...` (may not work in Docker Desktop)

## Summary

✅ **Yes, you can use Cursor on Windows!**
- Edit files in Cursor → changes sync to Docker automatically
- Run commands in Cursor's terminal → execute in Docker
- Use all Cursor features (Git, search, multi-file editing)
- Debug with print statements and Docker logs
- Full development workflow maintained!

The key is mounting your project directory as a Docker volume - then you get the best of both worlds: Cursor's IDE on Windows + Linux environment in Docker!
