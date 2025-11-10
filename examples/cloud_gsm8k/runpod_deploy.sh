#!/bin/bash
# RunPod Deployment Script
# This script helps you deploy AReaL GRPO training on RunPod
#
# Usage:
#   bash examples/cloud_gsm8k/runpod_deploy.sh
#
# Prerequisites:
#   - RunPod account with credits
#   - WandB API key
#   - RunPod CLI installed (optional, for automated deployment)

set -e

echo "=========================================="
echo "RunPod AReaL GRPO Training Deployment"
echo "=========================================="
echo ""

# Check for WandB API key
if [ -z "$WANDB_API_KEY" ]; then
    echo "ERROR: WANDB_API_KEY environment variable not set"
    echo "Please set it: export WANDB_API_KEY=your-api-key"
    exit 1
fi

echo "WandB API key: ${WANDB_API_KEY:0:10}..."
echo ""

# Configuration
TRAINING_CONFIG="${1:-1hour}"  # fast, 1hour, 3hour, or full
GPU_TYPE="${GPU_TYPE:-RTX 4090}"  # RTX 4090, A100, etc.
USE_SPOT="${USE_SPOT:-true}"  # Use spot instances for savings

echo "Configuration:"
echo "  Training: $TRAINING_CONFIG"
echo "  GPU: $GPU_TYPE"
echo "  Spot Instance: $USE_SPOT"
echo ""

# Docker command based on training config
case "$TRAINING_CONFIG" in
    fast)
        TRAIN_CMD="bash examples/cloud_gsm8k/run_training_cloud.sh fast"
        ;;
    1hour)
        TRAIN_CMD="bash examples/cloud_gsm8k/run_training_cloud.sh 1hour"
        ;;
    3hour)
        TRAIN_CMD="bash examples/cloud_gsm8k/run_training_cloud.sh 3hour"
        ;;
    full)
        TRAIN_CMD="bash examples/cloud_gsm8k/run_training_cloud.sh full"
        ;;
    *)
        echo "ERROR: Unknown training config: $TRAINING_CONFIG"
        echo "Valid options: fast, 1hour, 3hour, full"
        exit 1
        ;;
esac

echo "=========================================="
echo "RunPod Deployment Instructions"
echo "=========================================="
echo ""
echo "Option 1: Use RunPod Web Dashboard (Recommended)"
echo "---------------------------------------------------"
echo "1. Go to https://runpod.io"
echo "2. Navigate to 'Pods' → 'Deploy'"
echo "3. Use these settings:"
echo ""
echo "   Container Image: ghcr.io/inclusionai/areal-runtime:v0.3.4"
echo "   Docker Command: /bin/bash"
echo "   Environment Variables:"
echo "     WANDB_API_KEY=$WANDB_API_KEY"
echo "     PYTHONPATH=/workspace/AReaL"
echo "   Volume Mounts:"
echo "     /workspace/outputs → Your network volume (50GB+)"
echo "   GPU: $GPU_TYPE"
echo "   Spot Instance: $USE_SPOT"
echo ""
echo "4. Deploy pod"
echo "5. Connect via web terminal"
echo "6. Run setup commands (see below)"
echo ""
echo "Option 2: Use RunPod Template"
echo "---------------------------------------------------"
echo "1. Go to 'Templates' in RunPod dashboard"
echo "2. Create new template with:"
echo "   - Name: areal-grpo-training"
echo "   - Image: ghcr.io/inclusionai/areal-runtime:v0.3.4"
echo "   - Docker Command:"
echo "     bash -c \"set -e && pip config set global.index-url https://pypi.org/simple && pip config set global.extra-index-url '' && cd /workspace && if [ -d AReaL/.git ]; then cd AReaL && git fetch origin && git checkout -B DL4Math origin/DL4Math 2>/dev/null || git checkout -B DL4Math origin/DL4Math 2>/dev/null || (cd .. && rm -rf AReaL && git clone -b DL4Math https://github.com/nexthybrid/AReaL.git); else rm -rf AReaL && git clone -b DL4Math https://github.com/nexthybrid/AReaL.git; fi && cd /workspace/AReaL && pip install -e . && export WANDB_API_KEY=\\$WANDB_API_KEY && $TRAIN_CMD\""
echo "   - Environment: WANDB_API_KEY=$WANDB_API_KEY, PYTHONPATH=/workspace/AReaL"
echo "   - Volume: /workspace/outputs → your network volume"
echo "3. Deploy using template"
echo ""
echo "=========================================="
echo "Setup Commands (if using manual pod)"
echo "=========================================="
echo ""
echo "Once inside the pod (via web terminal):"
echo ""
echo "cd /workspace"
echo "pip config set global.index-url https://pypi.org/simple"
echo "pip config set global.extra-index-url \"\""
echo "git clone -b DL4Math https://github.com/nexthybrid/AReaL.git"
echo "cd AReaL"
echo "pip install -e ."
echo "export WANDB_API_KEY=$WANDB_API_KEY"
echo "$TRAIN_CMD"
echo ""
echo "=========================================="
echo "Cost Estimate"
echo "=========================================="
case "$TRAINING_CONFIG" in
    fast)
        EST_HOURS=0.5
        ;;
    1hour)
        EST_HOURS=2
        ;;
    3hour)
        EST_HOURS=4
        ;;
    full)
        EST_HOURS=120
        ;;
esac

if [ "$GPU_TYPE" = "RTX 4090" ]; then
    COST_PER_HOUR=0.29
    COST_SPOT=0.09
elif [ "$GPU_TYPE" = "A100" ]; then
    COST_PER_HOUR=1.39
    COST_SPOT=0.42
else
    COST_PER_HOUR=1.0
    COST_SPOT=0.3
fi

REGULAR_COST=$(echo "$EST_HOURS * $COST_PER_HOUR" | bc -l)
SPOT_COST=$(echo "$EST_HOURS * $COST_SPOT" | bc -l)

echo "Estimated training time: $EST_HOURS hours"
echo "GPU: $GPU_TYPE"
echo ""
if [ "$USE_SPOT" = "true" ]; then
    echo "Estimated cost (Spot): \$$(printf "%.2f" $SPOT_COST)"
    echo "Estimated cost (Regular): \$$(printf "%.2f" $REGULAR_COST)"
    echo "Savings: \$$(printf "%.2f" $(echo "$REGULAR_COST - $SPOT_COST" | bc -l))"
else
    echo "Estimated cost: \$$(printf "%.2f" $REGULAR_COST)"
    echo "Tip: Enable spot instances to save 50-70%!"
fi
echo ""
echo "=========================================="
echo "Monitoring"
echo "=========================================="
echo ""
echo "1. RunPod Dashboard: View GPU utilization and logs"
echo "2. WandB Dashboard: https://wandb.ai (project: gsm8k-grpo-local)"
echo "3. Checkpoints: /workspace/outputs/grpo/checkpoints/"
echo ""
echo "=========================================="
echo "Next Steps"
echo "=========================================="
echo ""
echo "1. Deploy pod using instructions above"
echo "2. Monitor training in WandB"
echo "3. Download results from network volume"
echo "4. Stop pod when done (save costs!)"
echo ""

