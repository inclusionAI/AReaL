#!/bin/bash
folder_name="asearcher_server"
rsync -av --delete --exclude='.git' --exclude='*pyc' /home/admin/meizhiyu.mzy/inclusionAI/AReaL /storage/openpsi/users/meizhiyu.mzy/workspace/$folder_name

cd /storage/openpsi/users/meizhiyu.mzy/workspace/$folder_name/AReaL && \
    python3 -m areal.launcher.slurm examples/search_agent/asearcher/train.py \
        --config experiments/asearcher/config.yaml \
        experiment_name=mzy-asearcher-qwen72b-inst-server-only \
        trial_name=run0 cluster.n_nodes=1 allocation_mode=sglang.d2t4p1 actor.path=/storage/openpsi/models/Qwen__Qwen2.5-72B-Instruct 