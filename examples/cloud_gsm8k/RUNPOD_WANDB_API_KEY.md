# How to Set WandB API Key in RunPod

## Quick Answer

Set your WandB API key as an **Environment Variable** when creating your RunPod pod or template.

## Method 1: Using RunPod Template (Recommended)

When creating a template in RunPod:

1. **Go to "Templates"** → **"New Template"**
2. **Fill in template details**
3. **In "Environment Variables" section**, add:
   - **Key**: `WANDB_API_KEY`
   - **Value**: `your-actual-wandb-api-key-here`
4. **Save template**
5. **Deploy pod** using the template

The API key will be automatically available when the pod starts.

## Method 2: Manual Pod Creation

When creating a pod manually:

1. **Go to "Pods"** → **"Deploy"**
2. **Select GPU** and other settings
3. **Scroll to "Environment Variables"** section
4. **Click "Add Environment Variable"**
5. **Enter**:
   - **Key**: `WANDB_API_KEY`
   - **Value**: `your-actual-wandb-api-key-here`
6. **Deploy pod**

## Method 3: Set After Pod Starts (Not Recommended)

If you forgot to set it, you can set it manually inside the pod:

```bash
# Inside pod (via web terminal)
export WANDB_API_KEY=your-actual-wandb-api-key-here
```

**Note**: This only works for the current session. If the pod restarts, you'll need to set it again. **Better to use Method 1 or 2.**

## Where to Find Your WandB API Key

1. **Go to**: https://wandb.ai
2. **Login** to your account
3. **Click** on your profile (top right)
4. **Go to** "Settings"
5. **Scroll to** "API Keys" section
6. **Copy** your API key (or create a new one)

## Visual Guide: Setting in RunPod Dashboard

### In Template Creation:

```
Templates → New Template
├── Basic Info
│   ├── Name: areal-grpo-training
│   └── Container Image: ghcr.io/inclusionai/areal-runtime:v0.3.4
├── Environment Variables  ← HERE!
│   ├── WANDB_API_KEY: your-api-key-here
│   └── PYTHONPATH: /workspace/AReaL
├── Volume Mounts
└── GPU Settings
```

### In Pod Creation:

```
Pods → Deploy
├── Select GPU
├── Container Image
├── Docker Command
├── Environment Variables  ← HERE!
│   ├── Click "Add Environment Variable"
│   ├── Key: WANDB_API_KEY
│   └── Value: your-api-key-here
└── Deploy
```

## Verification

To verify your WandB API key is set correctly:

```bash
# Inside pod (via web terminal)
echo $WANDB_API_KEY
# Should show your API key (not empty)
```

If it's empty, set it using Method 3 above, or recreate the pod with the environment variable set.

## Security Best Practices

1. ✅ **Use Environment Variables**: Don't hardcode API keys in scripts
2. ✅ **Don't Commit to Git**: Never commit API keys to your repository
3. ✅ **Use RunPod Secrets** (if available): Some platforms have secret management
4. ✅ **Rotate Keys**: Regularly rotate your API keys for security

## Troubleshooting

### "WandB API key not found" Error

**Solution**: Make sure the environment variable is set:
```bash
# Check if set
echo $WANDB_API_KEY

# If empty, set it
export WANDB_API_KEY=your-api-key-here
```

### WandB Not Logging

**Check**:
1. API key is correct (no typos)
2. API key has write permissions
3. Internet connection in pod (for online mode)
4. WandB project name is correct in config

### API Key Works Locally But Not in RunPod

**Common causes**:
- Environment variable not set in RunPod
- Pod restarted and environment variable lost
- Different shell session

**Solution**: Set it in RunPod template/pod settings (Method 1 or 2)

## Example: Complete Template Setup

Here's a complete example of setting up a template with WandB API key:

**Template Name**: `areal-grpo-training`

**Container Image**: `ghcr.io/inclusionai/areal-runtime:v0.3.4`

**Docker Command**:
```bash
bash -c "cd /workspace && git clone -b DL4Math https://github.com/nexthybrid/AReaL.git && cd AReaL && pip install -e . && export WANDB_API_KEY=\$WANDB_API_KEY && bash examples/cloud_gsm8k/run_training_cloud.sh 1hour"
```

**Environment Variables**:
- `WANDB_API_KEY`: `e1adc5be02c03fd34828c84b1ece937e0c2feb6e` (your actual key)
- `PYTHONPATH`: `/workspace/AReaL`

**Volume Mounts**:
- `/workspace/outputs` → Your network volume

**GPU**: RTX 4090 (spot enabled)

## Summary

**Best Practice**: Set `WANDB_API_KEY` as an environment variable in your RunPod template or pod settings. This ensures it's available every time the pod starts, and you don't have to manually set it each time.

