# Docker Quick Start Guide

## Windows 11 (CUDA GPU) - 3 Steps

```powershell
# 1. Navigate to directory
cd examples/local_gsm8k

# 2. Run setup script
.\docker-run-windows.ps1

# 3. Inside container, test and run
python examples/local_gsm8k/test_grpo_docker.py
python -m areal.launcher.local examples/local_gsm8k/train_grpo.py --config examples/local_gsm8k/train_grpo.yaml experiment_name=test trial_name=t1
```

## macOS (CPU-only) - 3 Steps

```bash
# 1. Navigate to directory
cd examples/local_gsm8k

# 2. Run setup script
./docker-run-macos.sh

# 3. Inside container, test (training will be slow on CPU)
python examples/local_gsm8k/test_grpo_docker.py
```

## Common Commands

```bash
# Start existing container
docker start -i areal-grpo-container

# Stop container
docker stop areal-grpo-container

# Remove container (start fresh)
docker rm -f areal-grpo-container

# View container logs
docker logs areal-grpo-container

# Execute command in running container
docker exec -it areal-grpo-container bash
```

## Test Script Output

✅ **All tests passed!** → Ready for training  
⚠️ **Some tests failed** → Check errors, may need to install dependencies

## Files Created

- `Dockerfile` - Docker image definition
- `docker-run-windows.ps1` - Windows setup script  
- `docker-run-macos.sh` - macOS setup script
- `test_grpo_docker.py` - Environment verification
- `DOCKER_SETUP_GUIDE.md` - Detailed guide

---

For detailed instructions, see `DOCKER_SETUP_GUIDE.md`

