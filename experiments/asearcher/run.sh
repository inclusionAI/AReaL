#!/bin/bash

expr_name=mzy-asearcher
trial_name=$(date +%Y%m%d%H%M%S)-flatten-tree-0
folder_name=${expr_name}_${trial_name}

rsync -av --delete --exclude='.git' --exclude='*.pyc' /home/admin/meizhiyu.mzy/inclusionAI/AReaL /storage/openpsi/users/meizhiyu.mzy/workspace/$folder_name

# Change enable_tree_training and pad_to_maximum to False in packed seq baseline
# Change user_llm_base_url to actual server address
srun --mpi=pmi2 -J mzy-asearcher --chdir /storage/openpsi/users/meizhiyu.mzy/workspace/$folder_name/AReaL --cpus-per-task=128 --gres=gpu:8 --mem-per-cpu=12G \
    singularity exec --no-home --writable-tmpfs --nv --pid --bind /storage:/storage \
    --env PYTHONPATH=/storage/openpsi/users/meizhiyu.mzy/workspace/$folder_name/AReaL \
    --env WANDB_BASE_URL=http://8.150.1.98:8080 \
    --env WANDB_API_KEY=local-8726acfd20a41b5ab8ac35fd51014b15169fa20a \
    --env HF_ENDPOINT=https://hf-mirror.com \
    --env RAG_SERVER_ADDR_DIR=/storage/openpsi/users/meizhiyu.mzy/workspace/rag_addr \
    /storage/openpsi/images/areal-latest.sif \
    bash -c "python3 -m areal.launcher.local examples/search_agent/asearcher/train.py \
        --config experiments/asearcher/config.yaml \
        experiment_name=$expr_name \
        trial_name=$trial_name"
