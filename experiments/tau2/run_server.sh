#!/bin/bash

folder_name="tau2_server"

rsync -av --delete --exclude='.git' --exclude='*.pyc' /home/admin/meizhiyu.mzy/inclusionAI/AReaL /storage/openpsi/users/meizhiyu.mzy/workspace/$folder_name

srun --mpi=pmi2 -J mzy-tau2-server --chdir /storage/openpsi/users/meizhiyu.mzy/workspace/$folder_name/AReaL --cpus-per-task=128 --gres=gpu:8 --mem-per-cpu=12G \
    singularity exec --no-home --writable-tmpfs --nv --pid --bind /storage:/storage \
    --env PYTHONPATH=/storage/openpsi/users/meizhiyu.mzy/workspace/$folder_name/AReaL \
    --env WANDB_BASE_URL=http://8.150.1.98:8080 \
    --env WANDB_API_KEY=local-8726acfd20a41b5ab8ac35fd51014b15169fa20a \
    --env HF_ENDPOINT=https://hf-mirror.com \
    --env TAU2_DATA_DIR=/storage/openpsi/users/donghonghua.dhh/workspace/tau2-bench/data \
    /storage/openpsi/images/areal-latest.sif \
    bash -c "python3 -m sglang.launch_server --model-path /storage/openpsi/models/Qwen__Qwen2.5-72B-Instruct --host 0.0.0.0 --tool-call-parser qwen25 --chat-template /storage/openpsi/users/meizhiyu.mzy/qwen3_nonthinking.jinja --dp-size 2 --tp-size 4"