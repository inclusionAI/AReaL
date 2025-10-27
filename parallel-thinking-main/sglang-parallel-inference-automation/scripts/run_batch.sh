#!/bin/bash

# Create a timestamp for the results directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="results/run_$TIMESTAMP"

# Create the results directory
mkdir -p "$RESULTS_DIR"

# Run the batch runner 8 times
for i in {1..8}
do
    echo "Running inference batch $i..."
    python3 src/batch_runner.py --output_dir "$RESULTS_DIR" --iteration $i
done

echo "All batches completed. Results stored in $RESULTS_DIR."