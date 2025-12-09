#!/bin/bash

uid="$(date +%Y%m%d-%H%M%S)"

srun --job-name=XXX \
    --nodes=2 \
    --ntasks-per-node=1 \
    --account=YYY \
    --partition=ZZZ \
    --gpus-per-task=8 \
    --mem=256G \
    --time=3-00:00:00 \
    ./train/sft_slurm.sh \
        --uid ${uid} \
        > log/sft_${uid}.txt 2>&1 &