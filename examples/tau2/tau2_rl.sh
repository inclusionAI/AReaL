#!/bin/bash
# Tau2 RL training script using archon backend

current_time=$(date +"%y%m%d-%H%M%S") && echo tau2_$current_time.log

# Prerequisites (run in container or install beforehand):
# pip install tenacity torch_memory_saver /storage/openpsi/users/donghonghua.dhh/workspace/tau2-bench transformers==4.57.1
# export TAU2_DATA_DIR=/storage/openpsi/users/donghonghua.dhh/workspace/tau2-bench/data

python3 -m areal.launcher.slurm \
    examples/tau2/tau2_train.py \
    --config examples/tau2/config.yaml \
    experiment_name=xss-tau2 trial_name=run-grpo-airline-qwen1.7B-archon \
    cluster.n_nodes=2 \
    cluster.n_gpus_per_node=8 \
    allocation_mode=sglang:d4t2+archon:d4p2t1 \
    rollout.max_concurrent_rollouts=512 \
    gconfig.n_samples=16 \
    train_dataset.batch_size=16 \
    gconfig.max_new_tokens=512 \
    econfig.max_steps=50 \
    econfig.domain=airline \
    actor.path=/storage/openpsi/models/Qwen__Qwen3-1.7B
    # For larger model:
    # actor.path=/storage/openpsi/models/Qwen3-30B-A3B
