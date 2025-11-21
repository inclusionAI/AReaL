#!/bin/bash
# Script to stop RunPod pod after training completes
# This prevents the container from restarting and wasting money

# Try to stop the pod using RunPod API if available
# RunPod provides environment variables for pod management

if [ -n "$RUNPOD_POD_ID" ]; then
    echo "Attempting to stop RunPod pod: $RUNPOD_POD_ID"
    # RunPod API endpoint (if available)
    if command -v curl >/dev/null 2>&1; then
        # Note: This requires RunPod API token - may not be available in all setups
        # For now, we'll just create a completion marker
        echo "Pod ID detected: $RUNPOD_POD_ID"
    fi
fi

# Create completion marker to prevent re-running
COMPLETION_MARKER="/workspace/outputs/training_completed_$(date +%Y%m%d_%H%M%S).marker"
echo "Training completed at $(date)" > "$COMPLETION_MARKER"
echo "Created completion marker: $COMPLETION_MARKER"

# Print instructions for manual pod stop
echo ""
echo "=========================================="
echo "⚠️  IMPORTANT: Stop the pod manually!"
echo "=========================================="
echo "Training has completed. To save costs:"
echo "1. Go to RunPod dashboard: https://www.runpod.io/console/pods"
echo "2. Find your pod and click 'Stop'"
echo ""
echo "Your checkpoints are safe in the network volume!"
echo "They will persist even after the pod stops."
echo "=========================================="
echo ""

# Exit with code 0 to indicate successful completion
# This will cause the container to exit
# If RunPod has auto-restart enabled, it will restart, but the completion
# marker check in run_training_cloud.sh will prevent re-running
exit 0

