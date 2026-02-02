#!/bin/bash
# Tau2 RL training script using Single Controller mode with Proxy Server
#
# Key differences from SPMD mode (tau2/tau2_rl.sh):
# 1. Uses `python3 script.py` directly instead of `python3 -m areal.launcher.slurm`
# 2. Uses `scheduler.type=slurm` to enable single-controller mode
# 3. Uses `allocation_mode=proxy+archon:...` instead of `sglang:...+archon:...`
# 4. Uses AgentWorkflow (receives base_url) instead of RolloutWorkflow (receives engine)

set -e

current_time=$(date +"%y%m%d-%H%M%S") && echo tau2_proxy_$current_time.log

# Prerequisites (run in container or install beforehand):
# pip install tenacity torch_memory_saver /storage/openpsi/users/donghonghua.dhh/workspace/tau2-bench transformers==4.57.1
# export TAU2_DATA_DIR=/storage/openpsi/users/donghonghua.dhh/workspace/tau2-bench/data

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Single Controller mode: run script directly with scheduler.type=slurm
python3 "$SCRIPT_DIR/tau2_train.py" \
    "$SCRIPT_DIR/config.yaml" \
    scheduler.type=slurm \
    experiment_name=xss-tau2-proxy trial_name=run-grpo-airline-qwen1.7B-proxy \
    cluster.n_nodes=2 \
    cluster.n_gpus_per_node=8 \
    allocation_mode=proxy+archon:d4p2t1 \
    rollout.openai.mode=proxy \
    rollout.max_concurrent_rollouts=512 \
    gconfig.n_samples=16 \
    train_dataset.batch_size=16 \
    gconfig.max_new_tokens=512 \
    econfig.max_steps=50 \
    econfig.domain=airline \
    econfig.save_trajectories=true \
    actor.path=/storage/openpsi/models/Qwen__Qwen3-1.7B \
    "$@"
    # For larger model:
    # actor.path=/storage/openpsi/models/Qwen3-30B-A3B
