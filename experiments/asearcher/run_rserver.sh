#!/bin/bash

folder_name=retrieval_server
PORT=8000
RAG_SERVER_ADDR_DIR=/storage/openpsi/users/meizhiyu.mzy/workspace/rag_addr

rsync -av --delete --exclude='.git' --exclude='*.pyc' /home/admin/meizhiyu.mzy/inclusionAI/AReaL /storage/openpsi/users/meizhiyu.mzy/workspace/$folder_name

srun --mpi=pmi2 -J mzy-asearcher-rserver --chdir /storage/openpsi/users/meizhiyu.mzy/workspace/$folder_name/AReaL --cpus-per-task=128 --gres=gpu:8 --mem-per-cpu=12G --pty \
    singularity exec --no-home --writable-tmpfs --nv --pid --bind /storage:/storage \
    --env PYTHONPATH=/storage/openpsi/users/meizhiyu.mzy/workspace/$folder_name/AReaL \
    --env WANDB_BASE_URL=http://8.150.1.98:8080 \
    --env WANDB_API_KEY=local-8726acfd20a41b5ab8ac35fd51014b15169fa20a \
    --env HF_ENDPOINT=https://hf-mirror.com \
    --env RAG_SERVER_ADDR_DIR=$RAG_SERVER_ADDR_DIR \
    --env PORT=$PORT \
    /storage/openpsi/users/gjx/data/gjx-images/gpu-infer-faiss-gpu1.8.0.sif \
    bash 
    # bash experiments/asearcher/retrieval_server/launch_local_server.sh $PORT $RAG_SERVER_ADDR_DIR
