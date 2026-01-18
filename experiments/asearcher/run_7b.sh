#!/bin/bash
expr_name=mzy-asearcher
trial_name=$(date +%Y%m%d%H%M%S)-7b-flatten-tree-0
folder_name=${expr_name}_${trial_name}
rsync -av --delete --exclude='.git' --exclude='*pyc' /home/admin/meizhiyu.mzy/inclusionAI/AReaL /storage/openpsi/users/meizhiyu.mzy/workspace/$folder_name

cd /storage/openpsi/users/meizhiyu.mzy/workspace/$folder_name/AReaL && \
    python3 -m areal.launcher.slurm examples/search_agent/asearcher/train.py \
        --config experiments/asearcher/config_7b.yaml \
        experiment_name=$expr_name \
        trial_name=$trial_name