#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0,1,2
N_GPU=2
EXP_NAME=gsm8k-dapo
TRIAL_NAME=trial1
FILE_ROOT=/data/yl/AReaL/tmp/areal/experiments
ACTOR_PATH=/data/yl/model/Qwen/Qwen2.5-1.5B-Instruct

TOTAL_TRAIN_EPOCHS=1

python3 -m areal.launcher.local \
    examples/experimental/dapo/gsm8k_dapo.py \
    --config examples/experimental/dapo/gsm8k_dapo.yaml \
    experiment_name="$EXP_NAME" \
    trial_name="$TRIAL_NAME" \
    total_train_epochs="$TOTAL_TRAIN_EPOCHS" \
    allocation_mode=sglang.d1p1t1+d1p1t1 \
    cluster.n_nodes=1 \
    cluster.n_gpus_per_node="$N_GPU" \
    cluster.fileroot="$FILE_ROOT" \
    actor.path="$ACTOR_PATH" \
    actor.optimizer.lr=1e-6 \
    actor.optimizer.weight_decay=0.1 \
    actor.overlong_reward_penalty=false 