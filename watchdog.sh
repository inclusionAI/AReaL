#!/usr/bin/env bash

# Check interval (seconds)
INTERVAL=60
# Threshold for number of idle nodes
THRESHOLD=3

# Command to run when idle nodes exceed threshold
CMD="python -m areal.launcher.slurm examples/lite/werewolf_grpo.py \
  --config examples/lite/configs/werewolf_grpo.yaml \
  stats_logger.wandb.mode=online \
  experiment_name=xmy-werewolf-sppo-15 \
  trial_name=villager1-t32b"

while true; do
    idle_nodes=$(sinfo -h -t idle -o "%D" | awk '{s+=$1} END{print s+0}')
    echo "$(date '+%F %T') Idle nodes: $idle_nodes"

    if (( idle_nodes > THRESHOLD )); then
        echo "$(date '+%F %T') Idle > $THRESHOLD, launching job..."
        eval "$CMD"
        echo "$(date '+%F %T') Job launched."
        exit 0
    else
        sleep "$INTERVAL"
    fi
done
