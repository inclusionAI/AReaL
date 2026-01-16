#!/bin/bash

expr_name=mzy-asearcher
trial_name=$(date +%Y%m%d%H%M%S)-flatten-tree-0
folder_name=${expr_name}_${trial_name}

rsync -av --delete --exclude='.git' --exclude='*.pyc' /home/admin/meizhiyu.mzy/inclusionAI/AReaL /storage/openpsi/users/meizhiyu.mzy/workspace/$folder_name

# Change enable_tree_training and pad_to_maximum to False in packed seq baseline
# Change user_llm_base_url to actual server address
srun --mpi=pmi2 -J mzy-asearcher --chdir /storage/openpsi/users/meizhiyu.mzy/workspace/$folder_name/AReaL --cpus-per-task=128 --gres=gpu:8 --mem-per-cpu=12G --pty \
    singularity exec --no-home --writable-tmpfs --nv --pid --bind /storage:/storage \
    --env PYTHONPATH=/storage/openpsi/users/meizhiyu.mzy/workspace/$folder_name/AReaL \
    --env WANDB_BASE_URL=http://8.150.1.98:8080 \
    --env WANDB_API_KEY=local-8726acfd20a41b5ab8ac35fd51014b15169fa20a \
    --env HF_ENDPOINT=https://hf-mirror.com \
    --env RAG_SERVER_ADDR_DIR=/storage/openpsi/users/wanghuaijie.whj/jwhj/ASearcher/rag_addr \
    /storage/openpsi/images/areal-latest.sif \
    bash -c "python3 -m areal.launcher.local examples/search_agent/asearcher/train.py \
        --config experiments/asearcher/config.yaml \
        experiment_name=$expr_name \
        trial_name=$trial_name \
        stats_logger.wandb.mode=online \
        cluster.fileroot=/storage/openpsi/users/meizhiyu.mzy/experiments \
        cluster.name_resolve.type=nfs \
        cluster.name_resolve.nfs_record_root=/storage/openpsi/users/meizhiyu.mzy/experiments/name_resolve \
        cluster.n_nodes=1 \
        cluster.n_gpus_per_node=8 \
        allocation_mode=sglang:d6+d2 \
        actor.path=/storage/openpsi/models/Qwen__Qwen2-1.5B-Instruct/ \
        train_dataset.path=/storage/openpsi/users/meizhiyu.mzy/mydata/asearcher/ASearcher-LRM-35k.jsonl \
        judge_engine.experiment_name=mzy-asearcher-qwen72b-inst-server-only \
        judge_engine.trial_name=run0 \
        +total_train_steps=1000 \
        rollout.enable_rollout_tracing=True \
        +actor.enable_tree_training=True \
        actor.pad_to_maximum=True"

