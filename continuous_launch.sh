#!/bin/bash

# Continuous launch script for AReaL training
# This script will continuously attempt to launch the training job,
# retrying after failures or every 30 minutes

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
MAX_RUNTIME=1800  # 30 minutes in seconds
RETRY_DELAY=10    # Delay before retry after failure (seconds)

# Command to execute
EXPERIMENT_NAME="zzy-4b-test-52"
TRIAL_NAME="0.3-latency"
CMD="python3 -m areal.launcher.slurm examples/math/parallel_grpo.py \
    --config examples/math/openr1_rl.yaml \
    experiment_name=${EXPERIMENT_NAME} \
    trial_name=${TRIAL_NAME} \
    cluster.n_gpus_per_node=8"

# Counter for attempts
ATTEMPT=0

echo -e "${GREEN}Starting continuous launch script${NC}"
echo -e "${GREEN}Command: ${CMD}${NC}"
echo -e "${GREEN}Max runtime per attempt: ${MAX_RUNTIME}s (30 minutes)${NC}"
echo -e "${GREEN}Press Ctrl+C to stop${NC}\n"

# Trap Ctrl+C to exit gracefully
trap 'echo -e "\n${RED}Interrupted by user. Exiting...${NC}"; exit 0' INT

while true; do
    ATTEMPT=$((ATTEMPT + 1))
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo -e "${YELLOW}=======================================${NC}"
    echo -e "${YELLOW}Attempt #${ATTEMPT} at ${TIMESTAMP}${NC}"
    echo -e "${YELLOW}=======================================${NC}\n"
    
    # Run the command with timeout
    timeout ${MAX_RUNTIME} ${CMD}
    EXIT_CODE=$?
    
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Check exit code
    if [ $EXIT_CODE -eq 124 ]; then
        # Timeout occurred (30 minutes reached)
        echo -e "\n${YELLOW}[${TIMESTAMP}] Timeout reached (30 minutes). Restarting...${NC}\n"
    elif [ $EXIT_CODE -eq 0 ]; then
        # Success
        echo -e "\n${GREEN}[${TIMESTAMP}] Job completed successfully!${NC}\n"
        echo -e "${GREEN}Exiting continuous launch script.${NC}"
        exit 0
    else
        # Error occurred
        echo -e "\n${RED}[${TIMESTAMP}] Job failed with exit code ${EXIT_CODE}${NC}"
        echo -e "${RED}Likely cause: No machines available or timeout waiting for servers${NC}"
        echo -e "${YELLOW}Retrying in ${RETRY_DELAY} seconds...${NC}\n"
        sleep ${RETRY_DELAY}
    fi
    
    # Small delay before next attempt
    sleep 2
done
