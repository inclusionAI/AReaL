#!/bin/bash
# Tau2 RL training script using Single Controller mode with AgentWorkflow
#
# Key differences from SPMD mode (tau2/tau2_rl.sh):
# 1. Runs script directly (not via launcher) to enable single-controller mode
# 2. Uses AgentWorkflow which receives base_url instead of RolloutWorkflow which receives engine
# 3. Single-controller mode is determined by NOT using areal.launcher.slurm

set -e

# Set TAU2_DATA_DIR for tau2-bench data
export TAU2_DATA_DIR=/storage/openpsi/users/donghonghua.dhh/workspace/tau2-bench/data

current_time=$(date +"%y%m%d-%H%M%S") && echo tau2_proxy_$current_time.log

# Prerequisites (run in container or install beforehand):
# pip install tenacity torch_memory_saver /storage/openpsi/users/donghonghua.dhh/workspace/tau2-bench transformers==4.57.1

# Use Python 3.12+ (required for AReaL's type syntax)
PYTHON=${PYTHON:-python}

# Single Controller mode: run script directly with scheduler.type=slurm
# This does NOT use areal.launcher.slurm, which enables single-controller mode
$PYTHON examples/tau2-proxy/tau2_train.py \
    --config examples/tau2-proxy/config.yaml \
    scheduler.type=slurm \
    experiment_name=hcy-tau2-proxy trial_name=run-grpo-airline-qwen1.7B-proxy \
    cluster.n_nodes=2 \
    cluster.n_gpus_per_node=8 \
    allocation_mode=sglang:d4t2+archon:d4p2t1 \
    rollout.openai.mode=inline \
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
