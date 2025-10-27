#!/bin/bash

# Cleanup script for SGLang parallel inference automation

# Define directories
RESULTS_DIR="../results"
LOGS_DIR="../logs"

# Get current date and time for timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create a backup of the results and logs before cleanup
if [ -d "$RESULTS_DIR" ]; then
    mv "$RESULTS_DIR" "${RESULTS_DIR}_backup_$TIMESTAMP"
fi

if [ -d "$LOGS_DIR" ]; then
    mv "$LOGS_DIR" "${LOGS_DIR}_backup_$TIMESTAMP"
fi

# Recreate the results and logs directories
mkdir "$RESULTS_DIR"
mkdir "$LOGS_DIR"

echo "Cleanup completed. Old results and logs have been backed up with timestamp: $TIMESTAMP"