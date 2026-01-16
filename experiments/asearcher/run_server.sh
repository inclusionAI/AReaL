#!/bin/bash
folder_name="asearcher_server"
rsync -av --delete --exclude='.git' --exclude='*pyc' /home/admin/meizhiyu.mzy/inclusionAI/AReaL /storage/openpsi/users/meizhiyu.mzy/workspace/$folder_name

cd /storage/openpsi/users/meizhiyu.mzy/workspace/$folder_name/AReaL && \
    python3 -m areal.launcher.slurm examples/search_agent/asearcher/train.py \
        --config experiments/asearcher/config.yaml \
        cluster.n_nodes=1 \
        cluster.n_gpus_per_node=8 \
        cluster.fileroot=/storage/openpsi/users/meizhiyu.mzy/experiments \
        cluster.name_resolve.nfs_record_root=/storage/openpsi/users/meizhiyu.mzy/experiments/name_resolve \
        experiment_name=mzy-asearcher-qwen72b-inst-server-only \
        trial_name=run0 cluster.n_nodes=1 allocation_mode=sglang:d2t4 actor.path=/storage/openpsi/models/Qwen__Qwen2.5-72B-Instruct 