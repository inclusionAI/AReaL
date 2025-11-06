# Fix: Docker Not Found in WSL2

## The Problem

When running `docker` in WSL2, you get:
```
The command 'docker' could not be found in this WSL 2 distro.
We recommend to activate the WSL integration in Docker Desktop settings.
```

This means Docker Desktop's WSL2 integration is not enabled.

## Quick Fix (2 minutes)

### Step 1: Open Docker Desktop Settings

1. **Right-click** the Docker icon in your Windows system tray (bottom-right)
2. Click **Settings** (or open Docker Desktop and click the gear icon)

### Step 2: Enable WSL Integration

1. In Docker Desktop Settings, go to **Resources** → **WSL Integration**
2. You should see a list of WSL distributions
3. **Find "Ubuntu"** in the list
4. **Toggle the switch** next to Ubuntu to **ON** (Enabled)
5. Click **Apply & Restart** at the bottom

**Visual Guide:**
```
Resources → WSL Integration
┌─────────────────────────────────────┐
│ Enable integration with my default │
│ WSL distro                          │ [ON/OFF] ← Toggle this
│                                     │
│ Ubuntu                              │ [Enabled] ← Should show this
│ (or your distro name)               │
└─────────────────────────────────────┘
```

### Step 3: Restart WSL2

After Docker Desktop restarts:

```powershell
# In PowerShell (Windows)
wsl --shutdown
```

Then reopen Ubuntu from Start menu or run:
```powershell
wsl
```

### Step 4: Verify Docker Works

```bash
# In WSL2 Ubuntu
docker --version
docker ps
```

Both should work without errors!

---

## Verify Everything is Working

Run this quick test:

```bash
# In WSL2
docker run hello-world
```

Should output: "Hello from Docker!"

Then test GPU access:

```bash
# In WSL2
docker run --rm --gpus all nvidia/cuda:13.0-base-ubuntu22.04 nvidia-smi
```

Should show your RTX 4080 SUPER!

---

## Troubleshooting

### Issue: "WSL Integration" tab not visible

**Solution:**
1. Make sure Docker Desktop is fully started (whale icon steady in system tray)
2. Close and reopen Settings
3. Make sure you're using WSL2 (not WSL1):
   ```powershell
   wsl --list --verbose
   ```
   Should show `VERSION 2` for Ubuntu

### Issue: Ubuntu not in the list

**Solution:**
1. Make sure Ubuntu is installed and accessible:
   ```powershell
   wsl -l -v
   ```
2. If Ubuntu shows VERSION 1, convert to VERSION 2:
   ```powershell
   wsl --set-version Ubuntu 2
   ```
3. Restart Docker Desktop

### Issue: Toggle won't stay enabled

**Solution:**
1. Restart Docker Desktop completely
2. Make sure Ubuntu is the default distro:
   ```powershell
   wsl --set-default Ubuntu
   ```
3. Try enabling integration again

### Issue: Docker still not found after enabling

**Solution:**
1. **Shutdown WSL2 completely**:
   ```powershell
   wsl --shutdown
   ```
2. **Restart Docker Desktop** (close and reopen)
3. **Reopen Ubuntu** (from Start menu or `wsl` command)
4. Test again:
   ```bash
   docker --version
   ```

---

## After Fixing

Once Docker works in WSL2, you can proceed with the AReaL container:

```bash
# In WSL2
docker run -it --name areal-grpo \
    --gpus all \
    --ipc=host \
    --shm-size=16g \
    -v /mnt/c/Users/tongz/git/GT/AReaL:/workspace/AReaL \
    -w /workspace/AReaL \
    ghcr.io/inclusionai/areal-runtime:v0.3.4 \
    /bin/bash
```

This should work now! 🎉

---

## Quick Checklist

- [ ] Docker Desktop is running (whale icon in system tray)
- [ ] Settings → Resources → WSL Integration → Ubuntu is **Enabled**
- [ ] Clicked "Apply & Restart" in Docker Desktop
- [ ] Ran `wsl --shutdown` in PowerShell
- [ ] Reopened Ubuntu
- [ ] Verified `docker --version` works in WSL2
