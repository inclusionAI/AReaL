#!/bin/bash

expr_name=mzy-tau2
trial_name=$(date +%Y%m%d%H%M%S)-flatten-tree-0
folder_name=${expr_name}_${trial_name}

rsync -av --delete --exclude='.git' --exclude='*.pyc' /home/admin/meizhiyu.mzy/inclusionAI/AReaL /storage/openpsi/users/meizhiyu.mzy/workspace/$folder_name

# Change enable_tree_training and pad_to_maximum to False in packed seq baseline
# Change user_llm_base_url to actual server address
srun --mpi=pmi2 -J mzy-tau2-server --chdir /storage/openpsi/users/meizhiyu.mzy/workspace/$folder_name/AReaL --cpus-per-task=128 --gres=gpu:8 --mem-per-cpu=12G --pty \
    singularity exec --no-home --writable-tmpfs --nv --pid --bind /storage:/storage \
    --env PYTHONPATH=/storage/openpsi/users/meizhiyu.mzy/workspace/$folder_name/AReaL \
    --env WANDB_BASE_URL=http://8.150.1.98:8080 \
    --env WANDB_API_KEY=local-8726acfd20a41b5ab8ac35fd51014b15169fa20a \
    --env HF_ENDPOINT=https://hf-mirror.com \
    --env TAU2_DATA_DIR=/storage/openpsi/users/donghonghua.dhh/workspace/tau2-bench/data \
    --env TREE_PACK_DUMP_PATH=/storage/openpsi/users/meizhiyu.mzy/zeta/tree-data-visualize/tau2-16k-small \
    /storage/openpsi/images/areal-latest.sif \
    bash -c "pip install git+https://github.com/dhh1995/tau2-bench.git@dhh/async-and-custom-completion; \
        python3 -m areal.launcher.local examples/tau2/tau2_train.py \
        --config .legacy/tau2_grpo.yaml \
        experiment_name=$expr_name \
        trial_name=$trial_name \
        cluster.n_nodes=1 \
        cluster.n_gpus_per_node=8 \
        allocation_mode=sglang:d6+d2 \
        actor.path=/storage/openpsi/models/Qwen__Qwen3-1.7B \
        stats_logger.wandb.mode=online \
        rollout.enable_rollout_tracing=True \
        econfig.user_llm_base_url=http://33.180.161.30:30000/v1/ \
        econfig.user_llm=openai/hosted \
        econfig.max_steps=20 \
        actor.mb_spec.max_tokens_per_mb=16384 \
        ref.mb_spec.max_tokens_per_mb=16384 \
        gconfig.max_tokens=16383 \
        gconfig.max_new_tokens=2048 \
        gconfig.n_samples=4 \
        train_dataset.batch_size=8 \
        +total_train_steps=3 \
        +actor.enable_tree_training=True \
        +actor.pad_to_maximum=True"

